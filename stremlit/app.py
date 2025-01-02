import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool
from collections import Counter, defaultdict
from plotly.subplots import make_subplots
import plotly.express as px

st.set_page_config(page_title="Misdiagnosis Detection Tool", layout='wide', initial_sidebar_state='expanded')

# [Previous helper functions remain the same]
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

# [Other helper functions from first file]

def highlight_risk(row):
    risk = row.get('Misdiagnosis Risk', '')
    if risk == "Very High":
        return ['background-color: #ff4c4c'] * len(row)
    elif risk == "High":
        return ['background-color: #ffd966'] * len(row)
    elif risk == "Low":
        return ['background-color: #c6efce'] * len(row)
    elif risk == "Very Low":
        return ['background-color: #aec6cf'] * len(row)
    return [''] * len(row)

def main():
    # Improved header styling
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
        "📈 Sankey Diagram",
        "⚙️ Risk Assessment"
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
        
        if uploaded_file is not None:
            result = DataPreprocessing(uploaded_file)
            if result is not None:
                acc, A, B, C, IdT, ClassT = result
                Method(acc, A, B, C, IdT, ClassT)

    # Tab 2: Data Analysis
    with tabs[1]:
        if st.session_state.data is not None:
            st.header("Data Analysis")
            
            # ROC Curve Analysis
            df = st.session_state.data
            L = df.T.values.tolist()
            ANS, ScoreA, ScoreB = np.array(L[1]), np.array(L[2]), np.array(L[3])
            
            fpr_A, tpr_A, thresholds_A = metrics.roc_curve(ANS, ScoreA, pos_label=2)
            auc_scoreA = metrics.auc(fpr_A, tpr_A)
            
            fpr_B, tpr_B, thresholds_B = metrics.roc_curve(ANS, ScoreB, pos_label=4)
            auc_scoreB = metrics.auc(fpr_B, tpr_B)
            
            # Create ROC curve plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, mode='lines', 
                                   name=f'ScoreA (AUC = {auc_scoreA:.2f})',
                                   line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, mode='lines', 
                                   name=f'ScoreB (AUC = {auc_scoreB:.2f})',
                                   line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Random Guess',
                                   line=dict(color='gray', dash='dash')))
            
            fig.update_layout(
                title='ROC Curve Analysis',
                xaxis_title='False Positive Rate (FPR)',
                yaxis_title='True Positive Rate (TPR)',
                width=800, height=600
            )
            
            st.plotly_chart(fig)

    # [Tabs 3-5 remain similar to first file but with improved styling]
    # [Include the Sankey diagram and risk assessment table with the same functionality]

if __name__ == '__main__':
    main()
