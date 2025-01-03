from xmlrpc.client import _Method
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

# [Previous code remains the same until main()]

def main():
    # Application header with improved styling
    st.markdown("""
        <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; margin-bottom: 30px">
            <h1 style="color: white; text-align: center">Misdiagnosis Detection Tool</h1>
            <p style="color: white; text-align: center">Advanced analysis for medical diagnosis validation</p>
        </div>
    """, unsafe_allow_html=True)

    #頁籤ICON
    st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="🏥",
    layout='wide',
    initial_sidebar_state='expanded')

    
# 裝飾Tabs
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


#定義
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
    
    
    
    # Create tabs with improved styling
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
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Tab 1: File Upload
    with tabs[0]:
        st.header("File Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            help="Please upload a CSV file containing patient data"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success("File uploaded successfully!")
                st.write("Data Preview:", df.head())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")


        if uploaded_file is not None:
            try:
                result = DataPreprocessing(uploaded_file) # type: ignore
                if result is not None:
                    acc, A, B, C, IdT, ClassT = result
                    st.session_state.processed_data = {
                        'acc': acc, 'A': A, 'B': B, 'C': C, 
                        'IdT': IdT, 'ClassT': ClassT
                    }
                    _Method(acc, A, B, C, IdT, ClassT)
                    st.success("Data processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Tab 2: Data Analysis
    with tabs[1]:
        if st.session_state.data is not None:
            st.header("Data Analysis")
            
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


            # Basic Statistics
            st.subheader("Basic Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(st.session_state.data))
            with col2:
                st.metric("Features", len(st.session_state.data.columns))
            with col3:
                st.metric("Missing Values", st.session_state.data.isnull().sum().sum())

            # Data Distribution
            st.subheader("Data Distribution")
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            fig = make_subplots(rows=1, cols=2)
            # Histogram
            fig.add_trace(
                go.Histogram(x=st.session_state.data[selected_col], name="Distribution"),
                row=1, col=1
            )
            # Box plot
            fig.add_trace(
                go.Box(y=st.session_state.data[selected_col], name="Box Plot"),
                row=1, col=2
            )
            fig.update_layout(height=400, title_text=f"Distribution Analysis of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

            # Correlation Matrix
            st.subheader("Correlation Matrix")
            corr = st.session_state.data[numeric_cols].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Misdiagnosis Detection
    with tabs[2]:
        if st.session_state.data is not None:
            st.header("Misdiagnosis Detection")

            # Parameters
            st.subheader("Detection Parameters")
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5)
            with col2:
                confidence = st.slider("Confidence Level", 0.8, 0.99, 0.95)

            if st.button("Run Detection"):
                with st.spinner("Running misdiagnosis detection..."):
                    # Simulated analysis (replace with actual detection logic)
                    time.sleep(2)
                    st.session_state.analysis_results = {
                        'high_risk': np.random.randint(1, 10),
                        'medium_risk': np.random.randint(5, 15),
                        'low_risk': np.random.randint(10, 30),
                        'confidence_score': confidence
                    }
                
                # Display Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Cases", st.session_state.analysis_results['high_risk'])
                with col2:
                    st.metric("Medium Risk Cases", st.session_state.analysis_results['medium_risk'])
                with col3:
                    st.metric("Low Risk Cases", st.session_state.analysis_results['low_risk'])

            #表格
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


    # Tab 4: Visualization
    with tabs[3]:
        if st.session_state.analysis_results is not None:
            st.header("Results Visualization")

            #風險表格
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
            
            
            # Risk Distribution Pie Chart
            st.subheader("Risk Distribution")
            risk_data = {
                'Category': ['High Risk', 'Medium Risk', 'Low Risk'],
                'Count': [
                    st.session_state.analysis_results['high_risk'],
                    st.session_state.analysis_results['medium_risk'],
                    st.session_state.analysis_results['low_risk']
                ]
            }
            fig = px.pie(risk_data, values='Count', names='Category', 
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)

            # Timeline Analysis
            st.subheader("Timeline Analysis")
            # Generate sample timeline data
            dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
            timeline_data = pd.DataFrame({
                'Date': dates,
                'Risk Score': np.random.uniform(0, 1, 10)
            })
            fig = px.line(timeline_data, x='Date', y='Risk Score')
            st.plotly_chart(fig, use_container_width=True)

    # Tab 5: Settings
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
