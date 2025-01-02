import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
import os
from plotly.subplots import make_subplots
import plotly.express as px
from collections import Counter, defaultdict

# Set page configuration with improved styling
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="🏥",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #1f77b4, #4a90e2);
    }
    .risk-high {
        background-color: #ff4c4c;
        padding: 5px;
        border-radius: 3px;
    }
    .risk-medium {
        background-color: #ffd966;
        padding: 5px;
        border-radius: 3px;
    }
    .risk-low {
        background-color: #c6efce;
        padding: 5px;
        border-radius: 3px;
    }
    </style>
""", unsafe_allow_html=True)

def Zscore(v):
    """計算 Z-score，處理零值的特殊情況"""
    if v == 0.0:
        return 0.0
    else:
        return v

def DataPreprocessing(uploaded_file):
    """數據預處理函數"""
    start_time = time.time()

    # 定義預處理參數
    rat1 = 0.85  # 訓練集比例
    rat2 = 0.15  # 測試集比例
    rof = 2      # 四捨五入位數
    ski = 1      # 跳過的行數

    try:
        # 讀取 CSV 文件
        df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=ski)
        
        st.write(f"DataFrame shape: {df.shape}")
        st.write("DataFrame preview:", df.head())

        cla = 9  # 分類列的索引

        if cla >= df.shape[1]:
            st.error(f"Error: Column index {cla} exceeds the number of columns ({df.shape[1]})")
            return None

        # 處理缺失值
        df.fillna(value='NotNumber', inplace=True)
        st.write(f"Process-1_Missing Value: Row={df.shape[0]}, Column={df.shape[1]}")
        st.write(f"DataFrame (with column-{cla} 'Class')", df.head())
        
        # 初始化變量
        acc = 0  # 累計唯一項目數
        itr1 = int(df.shape[0] * rat1)  # 訓練集大小
        itr2 = int(df.shape[0] * rat2)  # 測試集大小
        D_IdClass = df.iloc[:, cla].to_dict()  # 類別字典
        
        # 轉置數據並準備處理
        L = df.T.values.tolist()
        R = []  # 存儲處理後的數據
        U = []  # 存儲唯一值映射

        # 處理每一列數據
        for c in [L[l] for l in range(len(L)-1)]:
            D_Num = {}  # 數值型數據
            D_Cat = {}  # 分類型數據
            L_All = [0 for _ in c]  # 初始化結果列表
            
            # 分離數值型和分類型數據
            for e in range(len(c)):
                if isinstance(c[e], (int, float)) and not isinstance(c[e], bool):
                    D_Num[e] = c[e]
                else:
                    D_Cat[e] = c[e]
            
            # 處理數值型數據 - 計算 Z-score
            if D_Num:
                L_Num = list(D_Num.values())
                ave = np.mean(L_Num)
                std = np.std(L_Num) if len(L_Num) > 1 else 1
                
                for n in {v[0]: Zscore(round((v[1]-ave)/std, rof)) for v in D_Num.items()}.items():
                    L_All[n[0]] = n[1]
            
            # 處理分類型數據
            for i in D_Cat.items():
                L_All[i[0]] = i[1]
            
            R.append(L_All)
        
        # 創建數據框
        df = pd.DataFrame(R).T
        st.write(f"Process-2_Z-Score: Row={df.shape[0]}, Column={df.shape[1]}")
        st.write("DataFrame", df.head())

        # 創建分類值的索引映射
        for r in R:
            u = set(r)
            d = {e: i + acc for i, e in enumerate(u)}
            acc += len(u)
            U.append(d)
        
        # 應用索引映射
        R = [tuple(U[i][e] for e in r) for i, r in enumerate(R)]
        df = pd.DataFrame(R).T
        st.write(f"Process-3_Index for Efficiency: UniqueItems={acc}, Row={df.shape[0]}, Column={df.shape[1]}")
        st.write("DataFrame", df.head())

        # 處理類別信息
        V = [tuple(v) for v in df.values.tolist()]
        W = {w[0]: set() for w in pd.DataFrame(R).T.iloc[:, :].value_counts().to_dict().items()}
        for v in range(len(V)):
            W[V[v]].add(D_IdClass[v])
        
        # 分離正常實例和矛盾實例
        N = [(v, V[v], D_IdClass[v]) for v in range(len(V)) if len(W[V[v]]) == 1]
        E = [(v, V[v], D_IdClass[v]) for v in range(len(V)) if len(W[V[v]]) > 1]
        st.write(f"Process-4_Contradiction (C) Instance (I): Normal-I={len(N)}, C-I={len(E)}")

        # 分離訓練集和測試集
        G = list(set([i[2] for i in N]))
        A = [i for i in N if i[0] < itr1 and i[2] == G[0]]
        B = [i for i in N if i[0] < itr1 and i[2] == G[1]]
        C = [i for i in N if i[0] >= itr2]
        
        end_time = time.time()
        st.write(f"Classes: {set([i[2] for i in C])}")
        st.write(f"I for TEST with Respective Classes: {dict(Counter([i[2] for i in C]))}")
        st.write(f"DataPreprocessing Complete! Time={end_time - start_time:.2f}s")

        return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]

    except Exception as e:
        st.error(f"Error in DataPreprocessing: {str(e)}")
        return None

def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (sum([v[0]]), set(v[1])) for k, v in pattern_counts.items()}

def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

def create_sankey_data(instance_data, score_A, score_B, pure_score_A, pure_score_B, index):
    source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
    target = [1, 2] + list(range(3, 3 + len(score_A[1]))) + list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1])))
    value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]
    label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + [f'P{i[0]}' for i in score_A[1]] + [f'N{i[0]}' for i in score_B[1]]
    
    return {
        'source': source,
        'target': target,
        'values': value,
        'labels': label,
        'node_colors': ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1]),
        'link_colors': ['#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])
    }

def main():
    st.markdown("""
        <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; margin-bottom: 30px">
            <h1 style="color: white; text-align: center">Misdiagnosis Detection Tool</h1>
            <p style="color: white; text-align: center">Advanced analysis for medical diagnosis validation</p>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📤 Upload Files",
        "📊 Data Analysis",
        "🔍 Misdiagnosis Detection",
        "📈 Visualization",
        "⚙️ Settings"
    ])

    # Global state management
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Upload Files Tab
    with tabs[0]:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                result = DataPreprocessing(uploaded_file)
                if result is not None:
                    acc, A, B, C, IdT, ClassT = result
                    st.session_state.processed_data = {
                        'acc': acc, 'A': A, 'B': B, 'C': C, 
                        'IdT': IdT, 'ClassT': ClassT
                    }
                    Method(acc, A, B, C, IdT, ClassT)
                    st.success("Data processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Data Analysis Tab
    with tabs[1]:
        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            
            st.header("Data Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(data['C']))
            with col2:
                st.metric("Unique Classes", len(set(data['ClassT'])))
            with col3:
                st.metric("Features", len(data['C'][0]) if data['C'] else 0)

            # ROC Curve
            ANS = np.array(data['ClassT'])
            ScoreA = np.array([get_score_of_instance(c, find_patterns_updated(data['A']))[0] for c in data['C']])
            ScoreB = np.array([get_score_of_instance(c, find_patterns_updated(data['B']))[0] for c in data['C']])
            
            fpr_A, tpr_A, _ = metrics.roc_curve(ANS, ScoreA, pos_label=2)
            fpr_B, tpr_B, _ = metrics.roc_curve(ANS, ScoreB, pos_label=4)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name='Score A'))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name='Score B'))
            fig.update_layout(title='ROC Curve Analysis')
            st.plotly_chart(fig)

    # Misdiagnosis Detection Tab
    with tabs[2]:
        if st.session_state.processed_data is not None:
            st.header("Misdiagnosis Detection")
            
            data = st.session_state.processed_data
            patterns_A = find_patterns_updated(data['A'])
            patterns_B = find_patterns_updated(data['B'])
            pure_patterns_A = find_pure_patterns(patterns_A, data['B'])
            pure_patterns_B = find_pure_patterns(patterns_B, data['A'])
            
            specific_instances = find_specific_instances(data['C'], 
                                                       patterns_A, patterns_B,
                                                       pure_patterns_A, pure_patterns_B)
            
            st.metric("Detected Risk Cases", len(specific_instances))
            
            risk_df = pd.DataFrame([{
                'ID': idx,
                'Risk Score': max(instance[3][0], instance[4][0]),
                'Class': data['ClassT'][idx]
            } for idx, instance in enumerate(specific_instances)])
            
            st.dataframe(
