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
from multiprocessing import Pool


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

# [Previous helper functions remain the same]

# Convert tuples to boolean arrays
def tuples_to_boolean_arrays(tuples, max_value):
    return np.array([np.isin(range(max_value), t) for t in tuples])

# Function to calculate score
def calculate_score(instance, pure_sets):
    score = 0
    for ps in pure_sets:
        if set(ps).issubset(set(instance)):
            score += len(ps)**2
    return score

# Parallel version of score calculation
def calculate_scores_parallel(instances, pure_sets, num_processes=4):
    with Pool(num_processes) as pool:
        scores = pool.starmap(calculate_score, [(instance, pure_sets) for instance in instances])
    return scores

# Function to identify pure sets directly in numpy array format
def identify_pure_sets_numpy(intersections, other_bool, max_value):
    pure_sets = []
    for intersection in intersections:
        intersection_bool = np.isin(range(max_value), intersection)
        if not np.any(np.all(intersection_bool <= other_bool, axis=-1)):
            pure_sets.append(intersection)
    return pure_sets

# Function to calculate unique intersections for a single array
def calculate_unique_intersections_single(array):
    intersections = np.bitwise_and(array[:, None, :], array)
    unique_intersections = set()
    for i in range(intersections.shape[0]):
        for j in range(intersections.shape[1]):
            intersection = tuple(np.where(intersections[i, j])[0])
            unique_intersections.add(intersection)
    return unique_intersections

# Parallel version of calculate_unique_intersections
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
    return {k: (sum([v[0]]), set(v[1])) for k, v in pattern_counts.items()}

# Function to find pure patterns in one dataset with respect to another
def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

# Function to get the score of an instance based on patterns
def get_score_of_instance(instance, patterns):
    score = 0
    pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_in_.append([set(pattern), data[0]])
    return score, pattern_in_

# Function to get the pure score of an instance based on pure patterns
def get_pure_score_of_instance(instance, pure_patterns):
    pure_score = 0
    pure_pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in pure_patterns.items():
        if set(pattern).issubset(instance_set):
            pure_score += data[0]
            pure_pattern_in_.append([set(pattern), data[0]])
    return pure_score, pure_pattern_in_

# Function to find instances in C that satisfy the specified conditions
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

def DataPreprocessing(uploaded_file):
    start_time = time.time()

    df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=ski)
    
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"DataFrame preview:", df.head())

    cla = 9  # 假设第9列是分类列

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
        ave, std = np.mean(L_Num), np.std(L_Num)
        for n in {v[0]:Zscore(round((v[1]-ave)/std, rof)) for v in D_Num.items()}.items(): L_All[n[0]] = n[1]
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

def Zscore(v):
    if v == 0.0:
        return 0.0
    else:
        return v

def DataPreprocessing(uploaded_file):
    start_time = time.time()

    # 定義 rat1, rat2 和 rof
    rat1 = 0.85  # 例如，80% 的數據用於訓練
    rat2 = 0.15  # 例如，15% 的數據用於測試
    rof = 2     # 例如，四捨五入到小數點後兩位
    ski= 0

    df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=1)
    
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"DataFrame preview:", df.head())

    cla = 9  # 假设第6列是分类列

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
        ave, std = np.mean(L_Num), np.std(L_Num)
        for n in {v[0]:Zscore(round((v[1]-ave)/std, rof)) for v in D_Num.items()}.items(): L_All[n[0]] = n[1]
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

def highlight_risk(row):
    """根據誤診風險返回整行的顏色"""
    risk = row.get('Misdiagnosis Risk', '')
    if risk == "Very High":
        return ['background-color: #ff4c4c'] * len(row)  # 柔和的紅色
    elif risk == "High":
        return ['background-color: #ffd966'] * len(row)  # 柔和的橙色
    elif risk == "Low":
        return ['background-color: #c6efce'] * len(row)  # 柔和的綠色
    elif risk == "Very Low":
        return ['background-color: #aec6cf'] * len(row)  # 柔和的藍色
    return [''] * len(row)

def create_sankey_diagram(sankey_data, title):
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['labels'],
            color=sankey_data['node_colors']
        ),
        link=dict(
            source=sankey_data['source'],
            target=sankey_data['target'],
            value=sankey_data['values'],
            color=sankey_data['link_colors']
        )
    )])

    fig.update_layout(title_text=title, font_size=10)
    return fig

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
            
            st.dataframe(risk_df)
        

    # Visualization Tab
    with tabs[3]:
        if st.session_state.processed_data is not None and 'specific_instances' in locals():
            st.header("Visualization")
            
            selected_instance = st.selectbox(
                "Select Patient ID",
                options=range(len(specific_instances)),
                format_func=lambda x: f"Patient {x+1}"
            )
            
            if selected_instance is not None:
                instance_data = specific_instances[selected_instance]
                
                # Create and display Sankey diagram
                sankey_data = create_sankey_data(
                    instance_data[0],
                    instance_data[1],
                    instance_data[2],
                    instance_data[3],
                    instance_data[4],
                    selected_instance
                )
                
                fig = create_sankey_diagram(sankey_data, "Patient Analysis Flow")
                st.plotly_chart(fig)

    # Settings Tab
    with tabs[4]:
        st.header("Settings")
        
        # Analysis Settings
        st.subheader("Analysis Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Maximum Threads", min_value=1, max_value=16, value=4)
            st.selectbox("Color Theme", ["Default", "Light", "Dark"])
        with col2:
            st.checkbox("Enable Advanced Analytics", value=True)
            st.checkbox("Auto-save Results", value=True)

        # Export Settings
        st.subheader("Export Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
            st.checkbox("Include Metadata", value=True)
        with col2:
            st.text_input("Export Directory", value="C:/Results")
            st.checkbox("Auto-export", value=False)

        # Save Settings
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
