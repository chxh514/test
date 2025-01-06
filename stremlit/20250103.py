import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool
from collections import Counter, defaultdict
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor
import os
from plotly.subplots import make_subplots

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

# Helper Functions
@st.cache_data
def tuples_to_boolean_arrays(tuples, max_value):
    return np.array([np.isin(range(max_value), t) for t in tuples])

@st.cache_data
def calculate_score(instance, pure_sets):
    score = 0
    for ps in pure_sets:
        if set(ps).issubset(set(instance)):
            score += len(ps)**2
    return score

def calculate_scores_parallel(instances, pure_sets, num_processes=os.cpu_count()):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        scores = list(executor.map(lambda x: calculate_score(x, pure_sets), instances))
    return scores

@st.cache_data
def identify_pure_sets_numpy(intersections, other_bool, max_value):
    pure_sets = []
    for intersection in intersections:
        intersection_bool = np.isin(range(max_value), intersection)
        if not np.any(np.all(intersection_bool <= other_bool, axis=-1)):
            pure_sets.append(intersection)
    return pure_sets

@st.cache_data
def calculate_unique_intersections_single(array):
    intersections = np.bitwise_and(array[:, None, :], array)
    unique_intersections = set()
    for i in range(intersections.shape[0]):
        for j in range(intersections.shape[1]):
            intersection = tuple(np.where(intersections[i, j])[0])
            unique_intersections.add(intersection)
    return unique_intersections

def calculate_unique_intersections_parallel(bool_arrays, num_processes=os.cpu_count()):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(calculate_unique_intersections_single, [bool_arrays]))
    return set.union(*results)

@st.cache_data
def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (v[0], v[1]) for k, v in pattern_counts.items()}

@st.cache_data
def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

@st.cache_data
def get_score_of_instance(instance, patterns):
    score = 0
    pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_in_.append([set(pattern), data[0]])
    return score, pattern_in_

@st.cache_data
def get_pure_score_of_instance(instance, pure_patterns):
    pure_score = 0
    pure_pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in pure_patterns.items():
        if set(pattern).issubset(instance_set):
            pure_score += data[0]
            pure_pattern_in_.append([set(pattern), data[0]])
    return pure_score, pure_pattern_in_

def find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    satisfying_instances = []

    for c in C:
        score_A = get_score_of_instance(c, patterns_A)
        score_B = get_score_of_instance(c, patterns_B)
        pure_score_A = get_pure_score_of_instance(c, pure_patterns_A)
        pure_score_B = get_pure_score_of_instance(c, pure_patterns_B)

        if (score_A[0] > score_B[0] or pure_score_A[0] < pure_score_B[0]) or (score_A[0] < score_B[0] or pure_score_A[0] > pure_score_B[0]):
            satisfying_instances.append((c, score_A, score_B, pure_score_A, pure_score_B))

    return satisfying_instances

def Zscore(v):
    if v == 0.0:
        return 0.0
    else:
        return v

@st.cache_data
def DataPreprocessing(df, rat1=0.85, rat2=0.15, rof=2, ski=0, cla=9):
    start_time = time.time()
    
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"DataFrame preview:", df.head())

    if cla >= df.shape[1]:
        st.error(f"Error: `cla` ({cla}) exceeds the number of columns in the DataFrame ({df.shape[1]})")
        return None

    df.fillna(value='NotNumber', inplace=True)
    st.write(f"Process-1_Missing Value: Row={df.shape[0]}, Column={df.shape[1]}")
    st.write(f"DataFrame (with column-{cla} 'Class')", df.head())
    
    acc, itr1, itr2, D_IdClass = 0, df.shape[0] * rat1, df.shape[0] * rat2, df.iloc[:, cla].to_dict()
    L, R, U = df.T.values.tolist(), [], []    

    for c in [L[l] for l in range(len(L)-1)]:
        D_Num, D_Cat, L_All = dict(), dict(), [0 for _ in c]
        for e in range(len(c)):
            if type(c[e]) != str: D_Num[e] = c[e]
            else: D_Cat[e] = c[e]
        L_Num = list(D_Num.values())
        if L_Num:  # Add check for empty L_Num
            ave, std = np.mean(L_Num), np.std(L_Num)
            for n in {v[0]:Zscore(round((v[1]-ave)/std if std != 0 else 0, rof)) for v in D_Num.items()}.items():
                L_All[n[0]] = n[1]
        for i in D_Cat.items(): L_All[i[0]] = i[1]
        R.append(L_All)
    
    df = pd.DataFrame(R).T
    st.write(f"Process-2_Z-Score: Row={df.shape[0]}, Column={df.shape[1]}")
    st.write(f"DataFrame", df.head())

    for r in R:
        u = set(r)
        d = {e: i + acc for i, e in enumerate(u)}
        acc += len(u)
        U.append(d)
    
    R = [tuple(U[i][e] for e in r) for i, r in enumerate(R)]
    df = pd.DataFrame(R).T
    st.write(f"Process-3_Index for Efficiency: UniqueItems={acc}, Row={df.shape[0]}, Column={df.shape[1]}")
    st.write(f"DataFrame", df.head())

    V, W = [tuple(v) for v in df.values.tolist()], {w[0]: set() for w in pd.DataFrame(R).T.iloc[:, :].value_counts().to_dict().items()}
    for v in range(len(V)): W[V[v]].add(D_IdClass[v])
    
    N, E = [(v, V[v], D_IdClass[v]) for v in range(len(V)) if len(W[V[v]]) == 1], [(v, V[v], D_IdClass[v]) for v in range(len(V)) if len(W[V[v]]) > 1]
    st.write(f"Process-4_Contradiction (C) Instance (I): Amounts of Normal-I={len(N)}, Amounts of C-I={len(E)}")

    G = list(set([i[2] for i in N]))    
    A = [i for i in N if i[0] < itr1 and i[2] == G[0]]
    B = [i for i in N if i[0] < itr1 and i[2] == G[1]]
    C = [i for i in N if i[0] >= itr2]
    
    end_time = time.time()
    st.write(f"Classes: {set([i[2] for i in C])}")
    st.write(f"I for TEST with Respective Classes: {dict(Counter([i[2] for i in C]))}")
    st.write(f"DataPreprocessing Complete! Time={end_time - start_time}, Train={rat1}%(first~{itr1}), Test={1-rat1}%({itr2}~end)")

    return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]

def Method(acc, A, B, C, IdT, ClassT):
    start_time = time.time()
    A_bool, B_bool = tuples_to_boolean_arrays(A, acc), tuples_to_boolean_arrays(B, acc)
    SA = calculate_scores_parallel(C, identify_pure_sets_numpy(calculate_unique_intersections_parallel(A_bool), B_bool, acc))
    SB = calculate_scores_parallel(C, identify_pure_sets_numpy(calculate_unique_intersections_parallel(B_bool), A_bool, acc))
    
    end_time = time.time()
    
    st.write(f"Method Complete! Time={end_time - start_time}, len(scores_A)={len(SA)}, len(scores_B)={len(SB)}")
    
    result_df = pd.DataFrame({'ID': IdT, 'Class': ClassT, 'A': SA, 'B': SB, 'Items': C})
    st.write("Result DataFrame", result_df)
    
    csv = result_df.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=csv, file_name='S.csv', mime='text/csv')
    
    return result_df

def highlight_risk(row):
    risk = row.get('Misdiagnosis Risk', '')
    if risk == "Very High":
        return ['background-color: #ff4c4c'] * len(row)  # Soft red
    elif risk == "High":
        return ['background-color: #ffd966'] * len(row)  # Soft orange
    elif risk == "Low":
        return ['background-color: #c6efce'] * len(row)  # Soft green
    elif risk == "Very Low":
        return ['background-color: #aec6cf'] * len(row)  # Soft blue
    return [''] * len(row)

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'specific_instances' not in st.session_state:
        st.session_state.specific_instances = None

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
        "📈 Sankey Diagram",
        "📊 Functions"
    ])

    # Upload Files Tab
    with tabs[0]:
        st.markdown("<h2 style='font-weight:bold;'>File Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("File Upload", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None, skiprows=1)
                result = DataPreprocessing(df)
                if result is not None:
                    acc, A, B, C, IdT,
                    ClassT = result
                    st.session_state.processed_data = {
                        'acc': acc,
                        'A': A,
                        'B': B,
                        'C': C,
                        'IdT': IdT,
                        'ClassT': ClassT
                    }
                    result_df = Method(acc, A, B, C, IdT, ClassT)
                    st.session_state.analysis_results = result_df
                    st.success("Data processed successfully!")
                    
                    # Calculate patterns and specific instances
                    patterns_A = find_patterns_updated(A)
                    patterns_B = find_patterns_updated(B)
                    pure_patterns_A = find_pure_patterns(patterns_A, B)
                    pure_patterns_B = find_pure_patterns(patterns_B, A)
                    specific_instances = find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B)
                    st.session_state.specific_instances = specific_instances
                    
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
            patterns_A = find_patterns_updated(data['A'])
            patterns_B = find_patterns_updated(data['B'])
            
            ScoreA = [get_score_of_instance(c, patterns_A)[0] for c in data['C']]
            ScoreB = [get_score_of_instance(c, patterns_B)[0] for c in data['C']]
            
            fpr_A, tpr_A, _ = metrics.roc_curve(data['ClassT'], ScoreA, pos_label=2)
            fpr_B, tpr_B, _ = metrics.roc_curve(data['ClassT'], ScoreB, pos_label=4)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name='Score A'))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name='Score B'))
            fig.update_layout(title='ROC Curve Analysis')
            st.plotly_chart(fig)

    # Misdiagnosis Detection Tab
    with tabs[2]:
        if st.session_state.specific_instances is not None:
            st.header("Misdiagnosis Detection")
            
            specific_instances = st.session_state.specific_instances
            st.metric("Detected Risk Cases", len(specific_instances))
            
            risk_df = pd.DataFrame([
                {
                    'ID': idx + 1,
                    'Risk Score': max(instance[3][0], instance[4][0]),
                    'Class': st.session_state.processed_data['ClassT'][idx]
                }
                for idx, instance in enumerate(specific_instances)
            ])
            
            st.dataframe(risk_df)

    # Sankey Diagram Tab
    with tabs[3]:
        if st.session_state.specific_instances is not None:
            specific_instances_C = st.session_state.specific_instances
            total_specific_instances_C = len(specific_instances_C)
            
            choices = [f"Data {i+1}" for i in range(total_specific_instances_C)]
            choice = st.selectbox("Data", [" "] + choices)
            
            if choice != " ":
                index = int(choice.split(" ")[1]) - 1
                st.subheader("RESULT")
                
                c, score_A, score_B, pure_score_A, pure_score_B = specific_instances_C[index]

                # Standard Sankey diagram
                source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
                target = [1, 2] + list(range(3, 3 + len(score_A[1]))) + list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1])))
                value = [score_A[0], score_B[0]] + [i[1] for i in score_A[1]] + [i[1] for i in score_B[1]]
                label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + [f'P{i[0]}' for i in score_A[1]] + [f'N{i[0]}' for i in score_B[1]]
                
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="#37474F", width=0.5),
                        label=label
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=['#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])
                    )
                )])
                st.plotly_chart(fig)

                # Pure Sankey diagram
                st.subheader("Pure RESULT")
                pure_source = [0, 0] + [1] * len(pure_score_A[1]) + [2] * len(pure_score_B[1])
                pure_target = [1, 2] + list(range(3, 3 + len(pure_score_A[1]))) + list(range(3 + len(pure_score_A[1]), 3 + len(pure_score_A[1]) + len(pure_score_B[1])))
                pure_value = [pure_score_A[0], pure_score_B[0]] + [i[1] for i in pure_score_A[1]] + [i[1] for i in pure_score_B[1]]
                pure_label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + [f'P{i[0]}' for i in pure_score_A[1]] + [f'N{i[0]}' for i in pure_score_B[1]]
                
                pure_fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="#37474F", width=0.5),
                        label=pure_label
                    ),
                    link=dict(
                        source=pure_source,
                        target=pure_target,
                        value=pure_value,
                        color=['#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(pure_score_A[1]) + ['#F1F8E9'] * len(pure_score_B[1])
                    )
                )])
                st.plotly_chart(pure_fig)

    # Functions Tab
    with tabs[4]:
        if st.session_state.specific_instances is not None:
            st.subheader("Misdiagnosis Risk Table")
            
            specific_instances_C = st.session_state.specific_instances
            ClassT = st.session_state.processed_data['ClassT']
            
            data = []
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
                data.append({
                    "Status": status,
                    "ID": idx + 1,
                    "NS": pure_score_A[0],
                    "PS": pure_score_B[0],
                    "Label": ClassT[idx],
                    "Misdiagnosis Risk": risk_level
                })

            df_risk = pd.DataFrame(data)
            styled_df = df_risk.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=False, height=600)

if __name__ == '__main__':
    main()
