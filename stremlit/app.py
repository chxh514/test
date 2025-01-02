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

# [Previous helper functions remain the same]

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
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Processing Threads", 1, 16, 4)
            st.selectbox("Risk Threshold Method", ["Standard", "Adaptive", "Custom"])
        with col2:
            st.slider("Confidence Level", 0.8, 0.99, 0.95)
            st.checkbox("Enable Advanced Analytics", True)

if __name__ == "__main__":
    main()
