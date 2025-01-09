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
import logging
from typing import List, Tuple, Dict, Any, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration constants
CONFIG = {
    'MAX_THREADS': min(os.cpu_count() or 4, 8),
    'BATCH_SIZE': 1000,
    'TIMEOUT': 300,  # seconds
    'DEFAULT_ROUND': 2,
    'TRAIN_RATIO': 0.85,
    'TEST_RATIO': 0.15,
    'CLASS_COLUMN': 9,  # Index of the class column
    'SKIP_ROWS': 1,
}

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

def validate_input_data(df: pd.DataFrame) -> bool:
    """
    Validate input DataFrame structure and content
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Input data is empty")
        
        # Check if class column exists
        if CONFIG['CLASS_COLUMN'] >= df.shape[1]:
            raise ValueError(f"Class column index {CONFIG['CLASS_COLUMN']} exceeds DataFrame columns")
        
        # Check for minimum required columns
        if df.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns")
        
        # Check for null values in class column
        if df.iloc[:, CONFIG['CLASS_COLUMN']].isnull().any():
            raise ValueError("Class column contains null values")
        
        return True
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise

def Zscore(v: float) -> float:
    """Calculate Z-score with safety check"""
    return 0.0 if v == 0.0 else v

def process_data_in_batches(df: pd.DataFrame, batch_size: int = CONFIG['BATCH_SIZE']):
    """Process large DataFrames in batches to prevent memory issues"""
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        yield df.iloc[start:end]

def DataPreprocessing(uploaded_file) -> Tuple[int, List, List, List, List, List]:
    """
    Preprocess the uploaded data file with improved error handling and progress tracking
    """
    try:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Read and validate data
        df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=CONFIG['SKIP_ROWS'])
        validate_input_data(df)
        
        status_text.text("Processing data...")
        progress_bar.progress(10)

        # Fill missing values
        df.fillna(value='NotNumber', inplace=True)
        
        # Calculate indices and class dictionary
        acc = 0
        itr1 = int(df.shape[0] * CONFIG['TRAIN_RATIO'])
        itr2 = int(df.shape[0] * CONFIG['TEST_RATIO'])
        D_IdClass = df.iloc[:, CONFIG['CLASS_COLUMN']].to_dict()
        
        progress_bar.progress(30)
        
        # Process data transformation
        L = df.T.values.tolist()
        R = []
        U = []

        for c in [L[l] for l in range(len(L)-1)]:
            D_Num = {e: val for e, val in enumerate(c) if not isinstance(val, str)}
            D_Cat = {e: val for e, val in enumerate(c) if isinstance(val, str)}
            L_All = [0] * len(c)
            
            if D_Num:
                L_Num = list(D_Num.values())
                ave = np.mean(L_Num)
                std = np.std(L_Num) if len(L_Num) > 1 else 1
                
                for n, v in D_Num.items():
                    L_All[n] = Zscore(round((v - ave) / std, CONFIG['DEFAULT_ROUND']))
            
            for i, v in D_Cat.items():
                L_All[i] = v
            
            R.append(L_All)
        
        progress_bar.progress(60)
        
        # Index transformation
        for r in R:
            u = set(r)
            d = {e: i + acc for i, e in enumerate(u)}
            acc += len(u)
            U.append(d)
        
        R = [tuple(U[i][e] for e in r) for i, r in enumerate(R)]
        
        progress_bar.progress(80)
        
        # Process instances
        V = [tuple(v) for v in pd.DataFrame(R).T.values.tolist()]
        W = {w[0]: set() for w in pd.DataFrame(R).T.iloc[:, :].value_counts().to_dict().items()}
        
        for v in range(len(V)):
            W[V[v]].add(D_IdClass[v])
        
        N = [(v, V[v], D_IdClass[v]) for v in range(len(V)) if len(W[V[v]]) == 1]
        
        # Split data
        G = list(set([i[2] for i in N]))
        A = [i for i in N if i[0] < itr1 and i[2] == G[0]]
        B = [i for i in N if i[0] < itr1 and i[2] == G[1]]
        C = [i for i in N if i[0] >= itr2]
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        processing_time = time.time() - start_time
        logger.info(f"Data preprocessing completed in {processing_time:.2f} seconds")
        
        return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        st.error(f"Error processing data: {str(e)}")
        return None

def Method(acc: int, A: List, B: List, C: List, IdT: List, ClassT: List):
    """
    Improved method implementation with parallel processing and progress tracking
    """
    try:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Processing patterns...")
        progress_bar.progress(30)
        
        # Calculate patterns and scores using parallel processing
        with Pool(processes=CONFIG['MAX_THREADS']) as pool:
            patterns_A = find_patterns_updated(A)
            patterns_B = find_patterns_updated(B)
            
            progress_bar.progress(50)
            
            pure_patterns_A = find_pure_patterns(patterns_A, B)
            pure_patterns_B = find_pure_patterns(patterns_B, A)
            
            progress_bar.progress(70)
            
            SA = pool.starmap(get_score_of_instance, [(c, patterns_A) for c in C])
            SB = pool.starmap(get_score_of_instance, [(c, patterns_B) for c in C])
        
        progress_bar.progress(90)
        
        # Prepare results DataFrame
        result_df = pd.DataFrame({
            'ID': IdT,
            'Class': ClassT,
            'Score_A': [s[0] for s in SA],
            'Score_B': [s[0] for s in SB],
            'Items': C
        })
        
        # Save results
        csv = result_df.to_csv(index=False)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        processing_time = time.time() - start_time
        logger.info(f"Method completed in {processing_time:.2f} seconds")
        
        return result_df, csv
        
    except Exception as e:
        logger.error(f"Error in method execution: {str(e)}")
        st.error(f"Error in analysis: {str(e)}")
        return None, None

def find_patterns_updated(data: List) -> Dict:
    """
    Updated pattern finding implementation with improved efficiency
    """
    pattern_counts = defaultdict(lambda: [0, set()])
    
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    
    return {k: (v[0], v[1]) for k, v in pattern_counts.items()}

def find_pure_patterns(patterns: Dict, other_data: List) -> Dict:
    """
    Find pure patterns with improved implementation
    """
    other_sets = [set(item) for item in other_data]
    return {
        pattern: data
        for pattern, data in patterns.items()
        if not any(set(pattern).issubset(other_set) for other_set in other_sets)
    }

def get_score_of_instance(instance: Tuple, patterns: Dict) -> Tuple[float, List]:
    """
    Calculate instance score with pattern analysis
    """
    score = 0
    pattern_in = []
    instance_set = set(instance)
    
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_in.append([set(pattern), data[0]])
    
    return score, pattern_in

def create_visualizations(data: Dict, specific_instances: List):
    """
    Create visualization components with improved styling
    """
    try:
        # Create ROC curve
        ANS = np.array(data['ClassT'])
        ScoreA = np.array([get_score_of_instance(c, find_patterns_updated(data['A']))[0] for c in data['C']])
        ScoreB = np.array([get_score_of_instance(c, find_patterns_updated(data['B']))[0] for c in data['C']])
        
        fpr_A, tpr_A, _ = metrics.roc_curve(ANS, ScoreA, pos_label=2)
        fpr_B, tpr_B, _ = metrics.roc_curve(ANS, ScoreB, pos_label=4)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name='Score A'))
        fig_roc.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name='Score B'))
        fig_roc.update_layout(title='ROC Curve Analysis')
        
        return fig_roc
        
    except Exception as e:
        logger.error(f"Error in visualization creation: {str(e)}")
        st.error(f"Error creating visualizations: {str(e)}")
        return None

def main():
    """
    Main application function with improved structure and error handling
    """
    st.markdown("""
        <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; margin-bottom: 30px">
            <h1 style="color: white; text-align: center">Misdiagnosis Detection Tool</h1>
            <p style="color: white; text-align: center">Advanced analysis for medical diagnosis validation</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    # Create tabs
    tabs = st.tabs([
        "📤 Upload Files",
        "📊 Data Analysis",
        "🔍 Misdiagnosis Detection",
        "📈 Visualization",
        "⚙️ Settings"
    ])

    # Upload Files Tab
    with tabs[0]:
        st.markdown("<h2 style='font-weight:bold;'>File Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Processing data...'):
                    result = DataPreprocessing(uploaded_file)
                    if result is not None:
                        acc, A, B, C, IdT, ClassT = result
                        st.session_state.processed_data = {
                            'acc': acc, 'A': A, 'B': B, 'C': C,
                            'IdT': IdT, 'ClassT': ClassT
                        }
                        result_df, csv = Method(acc, A, B, C, IdT, ClassT)
                        if result_df is not None:
                            st.session_state.analysis_
                            st.session_state.analysis_results = result_df
                            
                            # Provide download button for results
                            st.download_button(
                                label="Download Results CSV",
                                data=csv,
                                file_name='analysis_results.csv',
                                mime='text/csv'
                            )
                            st.success("Data processed successfully!")
            except Exception as e:
                logger.error(f"Error in file processing: {str(e)}")
                st.error(f"Error processing file: {str(e)}")

    # Data Analysis Tab
    with tabs[1]:
        if st.session_state.processed_data is not None:
            st.header("Data Analysis")
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Total Records",
                    len(st.session_state.processed_data['C']),
                    help="Total number of processed records"
                )
            with col2:
                st.metric(
                    "Unique Classes",
                    len(set(st.session_state.processed_data['ClassT'])),
                    help="Number of unique diagnosis classes"
                )
            with col3:
                st.metric(
                    "Features",
                    len(st.session_state.processed_data['C'][0]) if st.session_state.processed_data['C'] else 0,
                    help="Number of features analyzed"
                )
            
            # Display detailed analysis
            st.subheader("Detailed Analysis")
            try:
                fig = create_visualizations(
                    st.session_state.processed_data,
                    st.session_state.analysis_results if st.session_state.analysis_results is not None else []
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display class distribution
                class_dist = pd.DataFrame(
                    Counter(st.session_state.processed_data['ClassT']).items(),
                    columns=['Class', 'Count']
                )
                st.write("Class Distribution:")
                st.dataframe(class_dist)
                
            except Exception as e:
                logger.error(f"Error in data analysis visualization: {str(e)}")
                st.error("Error generating analysis visualizations")

    # Misdiagnosis Detection Tab
    with tabs[2]:
        if st.session_state.processed_data is not None:
            st.header("Misdiagnosis Detection")
            
            try:
                # Calculate patterns and scores
                patterns_A = find_patterns_updated(st.session_state.processed_data['A'])
                patterns_B = find_patterns_updated(st.session_state.processed_data['B'])
                pure_patterns_A = find_pure_patterns(patterns_A, st.session_state.processed_data['B'])
                pure_patterns_B = find_pure_patterns(patterns_B, st.session_state.processed_data['A'])
                
                # Find specific instances
                specific_instances = find_specific_instances(
                    st.session_state.processed_data['C'],
                    patterns_A, patterns_B,
                    pure_patterns_A, pure_patterns_B
                )
                
                # Display results
                st.metric("Detected Risk Cases", len(specific_instances))
                
                # Create risk analysis DataFrame
                risk_df = pd.DataFrame([{
                    'ID': st.session_state.processed_data['IdT'][idx],
                    'Class': st.session_state.processed_data['ClassT'][idx],
                    'Risk Score': max(instance[3][0], instance[4][0]),
                    'Pattern Count': len(instance[1][1]) + len(instance[2][1]),
                    'Risk Level': get_risk_level(max(instance[3][0], instance[4][0]))
                } for idx, instance in enumerate(specific_instances)])
                
                # Apply styling and display
                st.dataframe(
                    risk_df.style.apply(highlight_risk, axis=1),
                    height=400
                )
                
                # Add detailed analysis for selected cases
                if len(risk_df) > 0:
                    selected_case = st.selectbox(
                        "Select case for detailed analysis",
                        risk_df['ID'].tolist()
                    )
                    
                    if selected_case:
                        show_detailed_analysis(
                            selected_case,
                            specific_instances,
                            st.session_state.processed_data
                        )
                
            except Exception as e:
                logger.error(f"Error in misdiagnosis detection: {str(e)}")
                st.error("Error performing misdiagnosis detection")

    # Visualization Tab
    with tabs[3]:
        if st.session_state.processed_data is not None:
            st.header("Advanced Visualizations")
            
            try:
                # Create visualization options
                viz_type = st.selectbox(
                    "Select Visualization Type",
                    ["Pattern Distribution", "Risk Score Distribution", "Feature Correlation"]
                )
                
                if viz_type == "Pattern Distribution":
                    show_pattern_distribution(st.session_state.processed_data)
                elif viz_type == "Risk Score Distribution":
                    show_risk_distribution(st.session_state.analysis_results)
                else:
                    show_feature_correlation(st.session_state.processed_data)
                
            except Exception as e:
                logger.error(f"Error in visualization: {str(e)}")
                st.error("Error generating visualizations")

    # Settings Tab
    with tabs[4]:
        st.header("Settings")
        
        # Analysis Settings
        st.subheader("Analysis Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            new_max_threads = st.number_input(
                "Maximum Threads",
                min_value=1,
                max_value=16,
                value=CONFIG['MAX_THREADS']
            )
            new_batch_size = st.number_input(
                "Batch Size",
                min_value=100,
                max_value=10000,
                value=CONFIG['BATCH_SIZE']
            )
            
        with col2:
            new_train_ratio = st.slider(
                "Training Data Ratio",
                min_value=0.1,
                max_value=0.9,
                value=CONFIG['TRAIN_RATIO']
            )
            new_timeout = st.number_input(
                "Processing Timeout (seconds)",
                min_value=60,
                max_value=900,
                value=CONFIG['TIMEOUT']
            )
        
        # Update configuration
        if st.button("Save Settings"):
            try:
                CONFIG.update({
                    'MAX_THREADS': new_max_threads,
                    'BATCH_SIZE': new_batch_size,
                    'TRAIN_RATIO': new_train_ratio,
                    'TIMEOUT': new_timeout
                })
                st.success("Settings updated successfully!")
                
            except Exception as e:
                logger.error(f"Error updating settings: {str(e)}")
                st.error("Error saving settings")

def get_risk_level(score: float) -> str:
    """Determine risk level based on score"""
    if score >= 0.8:
        return "Very High"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.2:
        return "Low"
    return "Very Low"

def highlight_risk(row):
    """Apply color highlighting based on risk level"""
    risk_colors = {
        "Very High": "#ff4c4c",
        "High": "#ffd966",
        "Medium": "#fff2cc",
        "Low": "#c6efce",
        "Very Low": "#deebf7"
    }
    
    color = risk_colors.get(row.get('Risk Level', ""), "")
    return [f"background-color: {color}"] * len(row)

def show_detailed_analysis(case_id: int, specific_instances: List, data: Dict):
    """Show detailed analysis for a selected case"""
    st.subheader(f"Detailed Analysis for Case {case_id}")
    
    # Find case details
    case_details = next(
        (instance for instance in specific_instances 
         if data['IdT'][specific_instances.index(instance)] == case_id),
        None
    )
    
    if case_details:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Score Analysis")
            score_df = pd.DataFrame({
                'Metric': ['Score A', 'Score B', 'Pure Score A', 'Pure Score B'],
                'Value': [
                    case_details[1][0],
                    case_details[2][0],
                    case_details[3][0],
                    case_details[4][0]
                ]
            })
            st.dataframe(score_df)
        
        with col2:
            st.write("Pattern Analysis")
            pattern_df = pd.DataFrame({
                'Type': ['Patterns A', 'Patterns B'],
                'Count': [
                    len(case_details[1][1]),
                    len(case_details[2][1])
                ]
            })
            st.dataframe(pattern_df)
        
        # Create visualization
        create_case_visualization(case_details)

def create_case_visualization(case_details: Tuple):
    """Create detailed visualization for case analysis"""
    try:
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Case", "Score A", "Score B", "Pure A", "Pure B"],
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            ),
            link=dict(
                source=[0, 0, 0, 0],
                target=[1, 2, 3, 4],
                value=[
                    case_details[1][0],
                    case_details[2][0],
                    case_details[3][0],
                    case_details[4][0]
                ]
            )
        )])
        
        fig.update_layout(title_text="Case Analysis Flow")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating case visualization: {str(e)}")
        st.error("Error generating case visualization")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again.")
