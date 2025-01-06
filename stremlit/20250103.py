import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration with improved styling
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="🏥",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS with optimized selectors
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
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
    .risk-high { background-color: #ff4c4c; }
    .risk-medium { background-color: #ffd966; }
    .risk-low { background-color: #c6efce; }
    .risk-status { padding: 5px; border-radius: 3px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def calculate_score(instance, pure_sets):
    return sum(len(ps)**2 for ps in pure_sets if set(ps).issubset(set(instance)))

def calculate_scores_parallel(instances, pure_sets, max_workers=None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
def calculate_unique_intersections(bool_arrays):
    intersections = np.bitwise_and(bool_arrays[:, None, :], bool_arrays)
    unique_intersections = set()
    for i in range(intersections.shape[0]):
        for j in range(i, intersections.shape[1]):
            intersection = tuple(np.where(intersections[i, j])[0])
            if intersection:
                unique_intersections.add(intersection)
    return unique_intersections

@st.cache_data
def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    data_array = np.array(data)
    
    for i in range(len(data)):
        intersections = np.array([set(data[i]) & set(data[j]) for j in range(i, len(data))])
        for j, intersection in enumerate(intersections, i):
            if intersection:
                pattern = tuple(sorted(intersection))
                pattern_counts[pattern][0] += len(intersection)**2
                pattern_counts[pattern][1].update([i, j])
    
    return {k: (v[0], v[1]) for k, v in pattern_counts.items()}

@st.cache_data
def find_pure_patterns(patterns, other_data):
    other_sets = [set(item) for item in other_data]
    return {
        pattern: data for pattern, data in patterns.items()
        if not any(set(pattern).issubset(other_set) for other_set in other_sets)
    }

@st.cache_data
def get_instance_scores(instance, patterns):
    score = 0
    pattern_info = []
    instance_set = set(instance)
    
    for pattern, (pattern_score, _) in patterns.items():
        if set(pattern).issubset(instance_set):
            score += pattern_score
            pattern_info.append([set(pattern), pattern_score])
    
    return score, pattern_info

def find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    results = []
    
    for c in C:
        score_A = get_instance_scores(c, patterns_A)
        score_B = get_instance_scores(c, patterns_B)
        pure_score_A = get_instance_scores(c, pure_patterns_A)
        pure_score_B = get_instance_scores(c, pure_patterns_B)
        
        if ((score_A[0] > score_B[0] and pure_score_A[0] < pure_score_B[0]) or 
            (score_A[0] < score_B[0] and pure_score_A[0] > pure_score_B[0])):
            results.append((c, score_A, score_B, pure_score_A, pure_score_B))
    
    return results

@st.cache_data
def preprocess_data(df, class_col=9, train_ratio=0.85):
    # Handle missing values
    df = df.fillna('NotNumber')
    
    # Calculate indices for train/test split
    total_rows = len(df)
    train_size = int(total_rows * train_ratio)
    
    # Extract class information
    class_data = df.iloc[:, class_col].to_dict()
    
    # Process numerical and categorical columns
    processed_data = []
    unique_values_count = 0
    value_mappings = []
    
    for col in df.drop(df.columns[class_col], axis=1).T.values:
        num_mask = pd.to_numeric(pd.Series(col), errors='coerce').notna()
        num_values = pd.to_numeric(pd.Series(col)[num_mask])
        
        if len(num_values) > 0:
            # Normalize numerical values
            normalized = (num_values - num_values.mean()) / num_values.std() if num_values.std() != 0 else 0
            col[num_mask] = normalized.round(2)
        
        # Create mapping for unique values
        unique_vals = pd.Series(col).unique()
        mapping = {val: i + unique_values_count for i, val in enumerate(unique_vals)}
        unique_values_count += len(unique_vals)
        value_mappings.append(mapping)
        
        # Convert values using mapping
        processed_data.append([mapping[val] for val in col])
    
    # Convert to tuple format
    processed_tuples = list(map(tuple, np.array(processed_data).T))
    
    # Split data based on class
    classes = sorted(set(class_data.values()))
    class_A = [i for i, (_, c) in enumerate(class_data.items()) if c == classes[0] and i < train_size]
    class_B = [i for i, (_, c) in enumerate(class_data.items()) if c == classes[1] and i < train_size]
    test_indices = range(train_size, total_rows)
    
    return (unique_values_count,
            [processed_tuples[i] for i in class_A],
            [processed_tuples[i] for i in class_B],
            [processed_tuples[i] for i in test_indices],
            list(test_indices),
            [class_data[i] for i in test_indices])

def create_sankey_diagram(index, instance_data, title="RESULT"):
    c, score_A, score_B, pure_score_A, pure_score_B = instance_data
    
    # Prepare data for Sankey diagram
    source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
    target = ([1, 2] + 
             list(range(3, 3 + len(score_A[1]))) + 
             list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1]))))
    value = [score_A[0], score_B[0]] + [i[1] for i in score_A[1]] + [i[1] for i in score_B[1]]
    
    label = ([f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + 
             [f'P{i[0]}' for i in score_A[1]] + 
             [f'N{i[0]}' for i in score_B[1]])
    
    colors = (['#ECEFF1', '#F8BBD0', '#DCEDC8'] + 
              ['#FFEBEE'] * len(score_A[1]) + 
              ['#F1F8E9'] * len(score_B[1]))
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="#37474F", width=0.5),
            label=label,
            color=colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=colors[1:2] + colors[2:3] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])
        )
    )])
    
    fig.update_layout(title=title)
    return fig

def main():
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
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
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None, skiprows=1)
                result = preprocess_data(df)
                if result is not None:
                    st.session_state.processed_data = {
                        'acc': result[0],
                        'A': result[1],
                        'B': result[2],
                        'C': result[3],
                        'IdT': result[4],
                        'ClassT': result[5]
                    }
                    st.success("Data processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Data Analysis Tab
    if st.session_state.processed_data is not None:
        with tabs[1]:
            data = st.session_state.processed_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(data['C']))
            with col2:
                st.metric("Unique Classes", len(set(data['ClassT'])))
            with col3:
                st.metric("Features", len(data['C'][0]) if data['C'] else 0)
            
            # Calculate and display ROC curves
            patterns_A = find_patterns_updated(data['A'])
            patterns_B = find_patterns_updated(data['B'])
            
            scores_A = [get_instance_scores(c, patterns_A)[0] for c in data['C']]
            scores_B = [get_instance_scores(c, patterns_B)[0] for c in data['C']]
            
            fpr_A, tpr_A, _ = metrics.roc_curve(data['ClassT'], scores_A, pos_label=2)
            fpr_B, tpr_B, _ = metrics.roc_curve(data['ClassT'], scores_B, pos_label=4)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name='Score A'))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name='Score B'))
            fig.update_layout(title='ROC Curve Analysis')
            st.plotly_chart(fig)
        
        # Misdiagnosis Detection Tab
        with tabs[2]:
            st.header("Misdiagnosis Detection")
            
            pure_patterns_A = find_pure_patterns(patterns_A, data['B'])
            pure_patterns_B = find_pure_patterns(patterns_B, data['A'])
            
            specific_instances = find_specific_instances(
                data['C'], patterns_A, patterns_B, pure_patterns_A, pure_patterns_B
            )
            
            st.metric("Detected Risk Cases", len(specific_instances))
            
            risk_df = pd.DataFrame([
                {
                    'ID': idx + 1,
                    'Risk Score': max(instance[3][0], instance[4][0]),
                    'Class': data['ClassT'][idx]
                }
                for idx, instance in enumerate(specific_instances)
            ])
            
            st.dataframe(risk_df)
        
        # Sankey Diagram Tab
        with tabs[3]:
            if specific_instances:
                choice = st.selectbox(
                    "Select Patient",
                    [""] + [f"Patient {i+1}" for i in range(len(specific_instances))]
                )
                
                if choice:
                    index = int(choice.split(" ")[1]) - 1
                    instance_data = specific_instances[index]
                    
                    st.plotly_chart(create_sankey_diagram(
                        index, instance_data, "Standard Analysis"
                    ))
                    st.plotly_chart(create_sankey_diagram(
                        index, instance_data, "Pure Pattern Analysis"
                    ))
        
        # Functions Tab
        with tabs[4]:
            st.subheader("Misdiagnosis Risk Analysis")
            
            risk_data = []
            for idx, instance_data in enumerate(specific_instances):
                risk_score = max(instance_data[3][0], instance_data[4][0])
                
                risk_level = "Very Low"
                status = ""
                if risk_score >= 3000:
                    risk_level = "High"
                status = "⚠️"
            else:
                risk_level = "Very High"
                status = "⚠️"
            data.append({
                "Status": status,
                "ID": idx + 1,
                "NS": pure_score_A[0],  # 使用取過 pure 的分數
                "PS": pure_score_B[0],  # 使用取過 pure 的分數
                "Label": ClassT[idx],
                "Misdiagnosis Risk": risk_level
            })

        df_risk = pd.DataFrame(data)
        styled_df = df_risk.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=False, height=600)

if __name__ == '__main__':
    main()
