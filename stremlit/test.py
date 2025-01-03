import streamlit as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool

# Global variables
specific_instances_C = None
ClassT = None

# Page Configuration
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="m.jpg",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Constants
TRAIN_RATIO = 0.85
TEST_RATIO = 0.15
ROUND_DIGITS = 2
SKIP_ROWS = 1
CLASS_COLUMN = 9

def zscore(v):
    return 0.0 if v == 0.0 else v

def tuples_to_boolean_arrays(tuples, max_value):
    return np.array([np.isin(range(max_value), t) for t in tuples])

def calculate_score(instance, pure_sets):
    score = 0
    for ps in pure_sets:
        if set(ps).issubset(set(instance)):
            score += len(ps)**2
    return score

def calculate_scores_parallel(instances, pure_sets, num_processes=4):
    with Pool(num_processes) as pool:
        scores = pool.starmap(calculate_score, [(instance, pure_sets) for instance in instances])
    return scores

def identify_pure_sets_numpy(intersections, other_bool, max_value):
    pure_sets = []
    for intersection in intersections:
        intersection_bool = np.isin(range(max_value), intersection)
        if not np.any(np.all(intersection_bool <= other_bool, axis=-1)):
            pure_sets.append(intersection)
    return pure_sets

def calculate_unique_intersections_single(array):
    intersections = np.bitwise_and(array[:, None, :], array)
    return {tuple(np.where(intersections[i, j])[0]) 
            for i in range(intersections.shape[0]) 
            for j in range(intersections.shape[1])}

def calculate_unique_intersections_parallel(bool_arrays, num_processes=4):
    with Pool(num_processes) as pool:
        results = pool.map(calculate_unique_intersections_single, [bool_arrays])
    return set.union(*results)

def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (v[0], v[1]) for k, v in pattern_counts.items()}

def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

def get_score_of_instance(instance, patterns):
    score = 0
    pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_in_.append([set(pattern), data[0]])
    return score, pattern_in_

def get_pure_score_of_instance(instance, pure_patterns):
    return get_score_of_instance(instance, pure_patterns)

def find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    satisfying_instances = []
    for c in C:
        score_A = get_score_of_instance(c, patterns_A)
        score_B = get_score_of_instance(c, patterns_B)
        pure_score_A = get_pure_score_of_instance(c, pure_patterns_A)
        pure_score_B = get_pure_score_of_instance(c, pure_patterns_B)
        
        if ((score_A[0] > score_B[0] or pure_score_A[0] < pure_score_B[0]) or 
            (score_A[0] < score_B[0] or pure_score_A[0] > pure_score_B[0])):
            satisfying_instances.append((c, score_A, score_B, pure_score_A, pure_score_B))
    
    return satisfying_instances

def create_sankey_diagram(instance_data, title):
    c, score_A, score_B, pure_score_A, pure_score_B = instance_data
    
    source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
    target = ([1, 2] + 
             list(range(3, 3 + len(score_A[1]))) + 
             list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1]))))
    value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]
    
    labels = (['PATIENT', 'Positive P', 'Negative N'] + 
             [f'P{i[0]}' for i in score_A[1]] + 
             [f'N{i[0]}' for i in score_B[1]])
    
    node_colors = (['#ECEFF1', '#F8BBD0', '#DCEDC8'] + 
                  ['#FFEBEE'] * len(score_A[1]) + 
                  ['#F1F8E9'] * len(score_B[1]))
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#37474F", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=(node_colors[1:2] + node_colors[2:3] + 
                  ['#FFEBEE'] * len(score_A[1]) + 
                  ['#F1F8E9'] * len(score_B[1]))
        )
    )])
    
    fig.update_layout(title=title)
    return fig

def highlight_risk(row):
    risk = row.get('Misdiagnosis Risk', '')
    colors = {
        "Very High": '#ff4c4c',
        "High": '#ffd966',
        "Low": '#c6efce',
        "Very Low": '#aec6cf'
    }
    return [f'background-color: {colors.get(risk, "")}'] * len(row)

def preprocess_data(uploaded_file):
    global ClassT
    start_time = time.time()
    
    df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=SKIP_ROWS)
    
    if CLASS_COLUMN >= df.shape[1]:
        st.error(f"Error: Class column ({CLASS_COLUMN}) exceeds DataFrame columns ({df.shape[1]})")
        return None
    
    df.fillna(value='NotNumber', inplace=True)
    st.write(f"Process-1_Missing Value: Row={df.shape[0]}, Column={df.shape[1]}")
    
    acc = 0
    itr1 = int(df.shape[0] * TRAIN_RATIO)
    itr2 = int(df.shape[0] * TEST_RATIO)
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
    A = [i for i in N if i[0] < itr1 and i[2] == G[0]]
    B = [i for i in N if i[0] < itr1 and i[2] == G[1]]
    C = [i for i in N if i[0] >= itr2]
    
    ClassT = [i[2] for i in C]
    
    end_time = time.time()
    st.write(f"Data preprocessing completed in {end_time - start_time:.2f} seconds")
    
    return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], ClassT

def Method(acc, A, B, C, IdT, ClassT):
    start_time = time.time()
    
    A_bool = tuples_to_boolean_arrays(A, acc)
    B_bool = tuples_to_boolean_arrays(B, acc)
    
    A_intersections = calculate_unique_intersections_parallel(A_bool)
    B_intersections = calculate_unique_intersections_parallel(B_bool)
    
    pure_sets_A = identify_pure_sets_numpy(A_intersections, B_bool, acc)
    pure_sets_B = identify_pure_sets_numpy(B_intersections, A_bool, acc)
    
    SA = calculate_scores_parallel(C, pure_sets_A)
    SB = calculate_scores_parallel(C, pure_sets_B)
    
    result_df = pd.DataFrame({
        'ID': IdT,
        'Class': ClassT,
        'A': SA,
        'B': SB,
        'Items': C
    })
    
    end_time = time.time()
    st.write(f"Method completed in {end_time - start_time:.2f} seconds")
    st.write("Result DataFrame:", result_df)
    
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='analysis_results.csv',
        mime='text/csv'
    )
    
    return result_df

def create_sankey_diagrams(instance_data):
    # Create regular Sankey diagram
    regular_fig = create_sankey_diagram(
        instance_data,
        "Regular Analysis"
    )
    st.plotly_chart(regular_fig)
    
    # Create pure Sankey diagram
    pure_data = (
        instance_data[0],
        instance_data[3],  # pure_score_A
        instance_data[4],  # pure_score_B
        instance_data[3],
        instance_data[4]
    )
    pure_fig = create_sankey_diagram(pure_data, "Pure Analysis")
    st.plotly_chart(pure_fig)

def display_risk_table():
    st.subheader("Misdiagnosis Risk Table")
    
    risk_data = []
    for idx, (c, score_A, score_B, pure_score_A, pure_score_B) in enumerate(specific_instances_C):
        risk_score = max(pure_score_A[0], pure_score_B[0])
        
        if risk_score < 1000:
            risk_level = "Very Low"
            status = ""
        elif risk_score < 2000:
            risk_level = "Low"
            status = ""
        elif risk_score < 3000:
            risk_level = "High"
            status = "⚠️"
        else:
            risk_level = "Very High"
            status = "⚠️"
        
        risk_data.append({
            "Status": status,
            "ID": idx + 1,
            "NS": pure_score_A[0],
            "PS": pure_score_B[0],
            "Label": ClassT[idx],
            "Misdiagnosis Risk": risk_level
        })
    
    risk_df = pd.DataFrame(risk_data)
    styled_risk_df = risk_df.style.apply(highlight_risk, axis=1)
    st.dataframe(styled_risk_df, use_container_width=False, height=600)

def main():
    global specific_instances_C, ClassT
    
    st.markdown("""
        <div style="background-color: #C9E2F2;padding:10px;">
        <h1 style="color: #324BD9">Misdiagnosis Detection Tool</h1>
        </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(['Upload Files', 'DataFrame', "Detection of Misdiagnosis", 
                    "Sankey diagram", "Functions"])
    
    with tabs[0]:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            result = preprocess_data(uploaded_file)
            if result:
                acc, A, B, C, IdT, ClassT = result
                patterns_A = find_patterns_updated(A)
                patterns_B = find_patterns_updated(B)
                pure_patterns_A = find_pure_patterns(patterns_A, B)
                pure_patterns_B = find_pure_patterns(patterns_B, A)
                specific_instances_C = find_specific_instances(C, patterns_A, patterns_B, 
                                                            pure_patterns_A, pure_patterns_B)
                Method(acc, A, B, C, IdT, ClassT)
    
    with tabs[1]:
        uploaded_files = st.file_uploader("**Upload data for analysis**", type=['csv'], key='tab2')
        if uploaded_files:
            df = pd.read_csv(uploaded_files, sep=',', header=None, skiprows=1)
            st.write("Data Preview:")
            st.write(df)
            
            L = df.T.values.tolist()
            ANS, ScoreA, ScoreB = np.array(L[1]), np.array(L[2]), np.array(L[3])
            
            fpr_A, tpr_A, _ = metrics.roc_curve(ANS, ScoreA, pos_label=2)
            fpr_B, tpr_B, _ = metrics.roc_curve(ANS, ScoreB, pos_label=4)
            auc_scoreA = metrics.auc(fpr_A, tpr_A)
            auc_scoreB = metrics.auc(fpr_B, tpr_B)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name=f'ScoreA (AUC = {auc_scoreA:.2f})',
                                   mode='lines', line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name=f'ScoreB (AUC = {auc_scoreB:.2f})',
                                   mode='lines', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess',
                                   mode='lines', line=dict(color='gray', dash='dash')))
            
            fig.update_layout(
                title='ROC Curve Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                showlegend=True,
                width=800,
                height=600
            )
            
            st.plotly_chart(fig)
    
    with tabs[2]:
        if specific_instances_C is not None:
            st.write(f"Number of instances meeting criteria: {len(specific_instances_C)}")
            
            st.subheader("Detailed Analysis")
            analysis_data = []
            for idx, (c, score_A, score_B, pure_score_A, pure_score_B) in enumerate(specific_instances_C):
                analysis_data.append({
                    "Instance ID": idx + 1,
                    "Score A": score_A[0],
                    "Score B": score_B[0],
                    "Pure Score A": pure_score_A[0],
                    "Pure Score B": pure_score_B[0],
                    "Class": ClassT[idx]
                })
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df)
            
            # Add visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=analysis_df["Score A"],
                y=analysis_df["Score B"],
                mode='markers',
                name='Scores',
                text=analysis_df["Instance ID"],
                marker=dict(
                    size=10,
                    color=analysis_df["Pure Score A"],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title='Score Distribution',
                xaxis_title='Score A',
                yaxis_title='Score B',
                width=800,
                height=600
            )
            
            st.plotly_chart(fig)
    
    with tabs[3]:
        if specific_instances_C is not None:
            total_instances = len(specific_instances_C)
            choices = [f"Data {i+1}" for i in range(total_instances)]
            choice = st.selectbox("Select Data", [" "] + choices)
            
            if choice != " ":
                index = int(choice.split(" ")[1]) - 1
                st.subheader("RESULT")
                create_sankey_diagrams(specific_instances_C[index])
    
    with tabs[4]:
        if specific_instances_C is not None and ClassT is not None:
            display_risk_table()
            
            # Add summary statistics
            st.subheader("Summary Statistics")
            risk_counts = defaultdict(int)
            for _, (c, _, _, pure_score_A, pure_score_B) in enumerate(specific_instances_C):
                risk_score = max(pure_score_A[0], pure_score_B[0])
                if risk_score >= 3000:
                    risk_counts["Very High"] += 1
                elif risk_score >= 2000:
                    risk_counts["High"] += 1
                elif risk_score >= 1000:
                    risk_counts["Low"] += 1
                else:
                    risk_counts["Very Low"] += 1
            
            # Create pie chart for risk distribution
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=.3,
                marker_colors=['#ff4c4c', '#ffd966', '#c6efce', '#aec6cf']
            )])
            
            fig.update_layout(
                title='Risk Distribution',
                width=600,
                height=400
            )
            
            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
