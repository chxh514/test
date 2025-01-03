import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from sklearn import metrics
import time

# Constants
TRAIN_RATIO = 0.85
TEST_RATIO = 0.15
ROUND_DIGITS = 2
SKIP_ROWS = 1
CLASS_COLUMN = 9

# 初始化全局變數
data = {
    'A': None,
    'B': None,
    'C': None,
    'IdT': None,
    'ClassT': None
}
specific_instances = None

def zscore(v):
    return 0.0 if v == 0.0 else v

def find_patterns(instances):
    """找出實例中的模式"""
    patterns = defaultdict(lambda: [0, set()])
    
    for i in range(len(instances)):
        for j in range(i, len(instances)):
            intersection = tuple(sorted(set(instances[i]) & set(instances[j])))
            if intersection:
                patterns[intersection][0] += len(intersection)**2
                patterns[intersection][1].update([i, j])
    
    return {k: (v[0], v[1]) for k, v in patterns.items()}

def find_pure_patterns(patterns, other_instances):
    """找出純淨模式"""
    pure_patterns = {}
    other_sets = [set(instance) for instance in other_instances]
    
    for pattern, (score, indices) in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = (score, indices)
    
    return pure_patterns

def get_instance_score(instance, patterns):
    """計算實例的分數"""
    score = 0
    matched_patterns = []
    instance_set = set(instance)
    
    for pattern, (pattern_score, _) in patterns.items():
        if set(pattern).issubset(instance_set):
            score += pattern_score
            matched_patterns.append((set(pattern), pattern_score))
    
    return score, matched_patterns

def find_specific_instances(instances, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    """找出特定實例"""
    specific_instances = []
    
    for instance in instances:
        score_A = get_instance_score(instance, patterns_A)
        score_B = get_instance_score(instance, patterns_B)
        pure_score_A = get_instance_score(instance, pure_patterns_A)
        pure_score_B = get_instance_score(instance, pure_patterns_B)
        
        if ((score_A[0] > score_B[0] and pure_score_A[0] < pure_score_B[0]) or 
            (score_A[0] < score_B[0] and pure_score_A[0] > pure_score_B[0])):
            specific_instances.append((instance, score_A, score_B, pure_score_A, pure_score_B))
    
    return specific_instances

def preprocess_data(df):
    """數據預處理"""
    acc = 0
    itr1 = int(df.shape[0] * TRAIN_RATIO)
    D_IdClass = df.iloc[:, CLASS_COLUMN].to_dict()
    L = df.T.values.tolist()
    R = []
    U = []
    
    for c in [L[l] for l in range(len(L)-1)]:
        D_Num, D_Cat = {}, {}
        L_All = [0] * len(c)
        
        for e, val in enumerate(c):
            if isinstance(val, str):
                D_Cat[e] = val
            else:
                D_Num[e] = val
                
        if D_Num:
            L_Num = list(D_Num.values())
            ave, std = np.mean(L_Num), np.std(L_Num)
            for n, v in D_Num.items():
                L_All[n] = zscore(round((v - ave) / std, ROUND_DIGITS))
        
        for i, v in D_Cat.items():
            L_All[i] = v
            
        R.append(L_All)
    
    for r in R:
        unique_vals = set(r)
        index_map = {e: i + acc for i, e in enumerate(unique_vals)}
        acc += len(unique_vals)
        U.append(index_map)
    
    R = [tuple(U[i][e] for e in r) for i, r in enumerate(R)]
    df_indexed = pd.DataFrame(R).T
    
    V = [tuple(v) for v in df_indexed.values.tolist()]
    W = defaultdict(set)
    for v_idx, v in enumerate(V):
        W[v].add(D_IdClass[v_idx])
    
    N = [(v_idx, v, D_IdClass[v_idx]) for v_idx, v in enumerate(V) if len(W[v]) == 1]
    
    G = list(set(i[2] for i in N))
    A = [i[1] for i in N if i[0] < itr1 and i[2] == G[0]]
    B = [i[1] for i in N if i[0] < itr1 and i[2] == G[1]]
    C = [i[1] for i in N if i[0] >= itr1]
    IdT = [i[0] for i in N if i[0] >= itr1]
    ClassT = [i[2] for i in N if i[0] >= itr1]
    
    return A, B, C, IdT, ClassT

def create_sankey_diagram(instance_data):
    """創建 Sankey 圖"""
    instance, score_A, score_B, pure_score_A, pure_score_B = instance_data
    
    source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
    target = ([1, 2] + 
             list(range(3, 3 + len(score_A[1]))) + 
             list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1]))))
    value = [score_A[0], score_B[0]] + [p[1] for p in score_A[1]] + [p[1] for p in score_B[1]]
    
    labels = (['Instance', 'Class A', 'Class B'] + 
             [f'Pattern A{i}' for i in range(len(score_A[1]))] + 
             [f'Pattern B{i}' for i in range(len(score_B[1]))])
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=['lightblue'] * len(labels)
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(title_text="Pattern Analysis", font_size=10)
    return fig

def main():
    global data, specific_instances
    
    st.title('Pattern Analysis Tool')
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None, skiprows=SKIP_ROWS)
        
        if st.button('Process Data'):
            with st.spinner('Processing data...'):
                # 預處理數據
                data['A'], data['B'], data['C'], data['IdT'], data['ClassT'] = preprocess_data(df)
                
                # 查找模式
                patterns_A = find_patterns(data['A'])
                patterns_B = find_patterns(data['B'])
                
                # 查找純淨模式
                pure_patterns_A = find_pure_patterns(patterns_A, data['B'])
                pure_patterns_B = find_pure_patterns(patterns_B, data['A'])
                
                # 找出特定實例
                specific_instances = find_specific_instances(
                    data['C'], patterns_A, patterns_B, pure_patterns_A, pure_patterns_B
                )
                
                st.success(f'Found {len(specific_instances)} specific instances')
        
        if specific_instances is not None:
            st.subheader('Analysis Results')
            
            # 顯示特定實例的詳細信息
            for i, instance_data in enumerate(specific_instances):
                with st.expander(f'Instance {i+1}'):
                    st.write(f'Pattern scores A: {instance_data[1][0]}')
                    st.write(f'Pattern scores B: {instance_data[2][0]}')
                    st.write(f'Pure pattern scores A: {instance_data[3][0]}')
                    st.write(f'Pure pattern scores B: {instance_data[4][0]}')
                    
                    # 顯示 Sankey 圖
                    fig = create_sankey_diagram(instance_data)
                    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
