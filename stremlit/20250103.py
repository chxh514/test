import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool
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

# [Previous helper functions remain unchanged]
[All helper functions from both files...]

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
        "📈 Sankey Diagram",
        "📊 Functions"
    ])

    # Global state management
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'specific_instances' not in st.session_state:
        st.session_state.specific_instances = None

    # Upload Files Tab (from first file)
    with tabs[0]:
        st.markdown("<h2 style='font-weight:bold;'>File Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("File Upload", type=["csv"])
        
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

    # Data Analysis Tab (from first file)
    with tabs[1]:
        if st.session_state.processed_data is not None:
            data = st.session_state.processed_data
            
            st.header("Data Analysis")
            col1, col2, col3 = st.columns(3)
            
            [Rest of Data Analysis tab code from first file...]

    # Misdiagnosis Detection Tab (from first file)
    with tabs[2]:
        if st.session_state.processed_data is not None:
            [Rest of Misdiagnosis Detection tab code from first file...]

    # Sankey Diagram Tab (from second file)
    with tabs[3]:
        if st.session_state.processed_data is not None and st.session_state.specific_instances is not None:
            [Sankey Diagram tab code from second file...]

    # Functions Tab (from second file)
    with tabs[4]:
        if st.session_state.processed_data is not None and st.session_state.specific_instances is not None:
            [Functions tab code from second file...]

if __name__ == "__main__":
    main()
