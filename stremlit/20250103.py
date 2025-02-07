import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
import plotly.express as px
from collections import Counter, defaultdict
from multiprocessing import Pool
import functools

# Performance optimization: Cache heavy computations
@functools.lru_cache(maxsize=128)
def calculate_score(instance, pure_sets):
    score = 0
    for ps in pure_sets:
        if set(ps).issubset(set(instance)):
            score += len(ps)**2
    return score

# Improved UI Configuration
st.set_page_config(
    page_title="Medical Diagnosis Analysis Tool",
    page_icon="🏥",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Enhanced UI Styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4361ee;
        color: white;
        box-shadow: 0 4px 6px rgba(67, 97, 238, 0.1);
    }
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #4361ee, #3f37c9);
    }
    .risk-high {
        background-color: #ef233c;
        color: white;
        padding: 6px;
        border-radius: 4px;
    }
    .risk-medium {
        background-color: #ffd60a;
        padding: 6px;
        border-radius: 4px;
    }
    .risk-low {
        background-color: #52b788;
        color: white;
        padding: 6px;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Optimized data preprocessing
def preprocess_data(uploaded_file):
    train_ratio = 0.85
    test_ratio = 0.15
    rounding = 2
    skip_rows = 1
    
    start_time = time.time()
    
    # Performance optimization: Use chunks for large files
    df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=skip_rows, chunksize=10000)
    df = pd.concat(df, ignore_index=True)
    
    class_col = 9
    
    if class_col >= df.shape[1]:
        st.error(f"Error: Column index {class_col} exceeds DataFrame dimensions ({df.shape[1]})")
        return None
        
    # Vectorized operations for better performance
    df_processed = df.fillna('NotNumber')
    class_dict = df_processed.iloc[:, class_col].to_dict()
    
    # Parallel processing for Z-score calculation
    with ProcessPoolExecutor() as executor:
        processed_columns = list(executor.map(
            lambda col: process_column(col, rounding),
            [df_processed.iloc[:, i] for i in range(df_processed.shape[1]-1)]
        ))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return process_results(processed_columns, df_processed, class_dict, train_ratio, test_ratio, processing_time)

# Performance optimization: Parallel pattern finding
def find_patterns_parallel(data, chunk_size=1000):
    with ProcessPoolExecutor() as executor:
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        partial_results = list(executor.map(find_patterns_chunk, chunks))
    
    return merge_pattern_results(partial_results)

# Main application function with improved UI
def main():
    st.markdown("""
        <div style="background-color: #4361ee; padding: 30px; border-radius: 12px; margin-bottom: 40px">
            <h1 style="color: white; text-align: center; font-size: 2.5rem">Medical Diagnosis Analysis Tool</h1>
            <p style="color: white; text-align: center; font-size: 1.2rem">Advanced analytics for medical diagnosis validation</p>
        </div>
    """, unsafe_allow_html=True)

# Optimized helper functions
def process_zscore(data):
    """Vectorized Z-score calculation"""
    numeric_mask = pd.to_numeric(data, errors='coerce').notna()
    result = data.copy()
    if numeric_mask.any():
        numeric_data = pd.to_numeric(data[numeric_mask])
        result[numeric_mask] = (numeric_data - numeric_data.mean()) / numeric_data.std()
    return result

def create_sankey_visualization(data, title, node_colors=None):
    """Create optimized Sankey diagram"""
    if node_colors is None:
        node_colors = px.colors.qualitative.Set3
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="#475569", width=0.5),
            label=data['labels'],
            color=node_colors[:len(data['labels'])],
        ),
        link=dict(
            source=data['source'],
            target=data['target'],
            value=data['values'],
            color=[f"rgba{tuple(int(c * 255) for c in px.colors.hex_to_rgb(color)) + (0.6,)}" 
                  for color in node_colors[:len(data['values'])]]
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color="#1e293b")
        ),
        font=dict(size=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    return fig

class RiskAnalyzer:
    """Optimized risk analysis with caching"""
    def __init__(self):
        self.cache = {}
    
    @functools.lru_cache(maxsize=1024)
    def calculate_risk_score(self, instance_data):
        score_a = self._get_score(instance_data, 'positive')
        score_b = self._get_score(instance_data, 'negative')
        return max(score_a, score_b)
    
    def get_risk_level(self, score):
        if score < 1000:
            return "Very Low", "#94a3b8"  # Slate 400
        elif score < 2000:
            return "Low", "#4ade80"  # Green 400
        elif score < 3000:
            return "High", "#fb923c"  # Orange 400
        return "Very High", "#ef4444"  # Red 500

def create_analysis_dashboard():
    """Create main analysis dashboard with tabs"""
    tabs = st.tabs([
        "📤 Data Upload",
        "📊 Analysis",
        "🔍 Risk Detection",
        "📈 Visualization",
        "📋 Results"
    ])
    
    with tabs[0]:
        create_upload_section()
    
    with tabs[1]:
        if 'processed_data' in st.session_state:
            create_analysis_section()
    
    with tabs[2]:
        if 'processed_data' in st.session_state:
            create_risk_detection_section()
    
    with tabs[3]:
        if 'processed_data' in st.session_state:
            create_visualization_section()
    
    with tabs[4]:
        if 'processed_data' in st.session_state:
            create_results_section()

def create_upload_section():
    """Enhanced file upload section"""
    st.markdown("""
        <div class="upload-container">
            <h2>Data Upload</h2>
            <p>Upload your medical diagnosis data file (CSV format)</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing medical diagnosis data"
    )
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            try:
                result = preprocess_data(uploaded_file)
                if result:
                    st.session_state.processed_data = result
                    st.success("✅ Data processed successfully!")
                    show_data_preview()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def create_analysis_section():
    """Enhanced analysis section with metrics"""
    st.header("Data Analysis")
    
    data = st.session_state.processed_data
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Total Records",
            len(data['records']),
            "📊"
        )
    
    with col2:
        create_metric_card(
            "Risk Cases",
            len(data['risk_cases']),
            "⚠️"
        )
    
    with col3:
        create_metric_card(
            "Processing Time",
            f"{data['processing_time']:.2f}s",
            "⚡"
        )
    
    with col4:
        create_metric_card(
            "Accuracy",
            f"{data['accuracy']:.1f}%",
            "🎯"
        )
    
    # Create ROC curve
    st.subheader("ROC Curve Analysis")
    fig = create_roc_curve(data)
    st.plotly_chart(fig, use_container_width=True)

def create_metric_card(title, value, icon):
    """Create a styled metric card"""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

def create_risk_detection_section():
    """Enhanced risk detection visualization"""
    st.header("Risk Detection")
    
    risk_analyzer = RiskAnalyzer()
    data = st.session_state.processed_data
    
    # Create risk distribution chart
    risk_distribution = calculate_risk_distribution(data, risk_analyzer)
    fig = create_risk_distribution_chart(risk_distribution)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed risk analysis
    show_risk_analysis(data, risk_analyzer)

def create_visualization_section():
    """Enhanced visualization section with interactive charts"""
    st.header("Visualization & Analysis")
    
    if not st.session_state.processed_data:
        st.warning("Please upload and process data first.")
        return
        
    data = st.session_state.processed_data
    
    # Create visualization options
    viz_type = st.radio(
        "Select Visualization Type",
        ["Sankey Diagram", "Pattern Analysis", "Risk Distribution"],
        horizontal=True
    )
    
    if viz_type == "Sankey Diagram":
        create_enhanced_sankey(data)
    elif viz_type == "Pattern Analysis":
        create_pattern_analysis(data)
    else:
        create_risk_distribution(data)

def create_enhanced_sankey(data):
    """Create an enhanced Sankey diagram with tooltips and interactivity"""
    patterns_A = find_patterns_parallel(data['A'])
    patterns_B = find_patterns_parallel(data['B'])
    pure_patterns_A = find_pure_patterns(patterns_A, data['B'])
    pure_patterns_B = find_pure_patterns(patterns_B, data['A'])
    
    # Allow user to select specific case
    case_id = st.selectbox(
        "Select Case ID",
        options=list(range(1, len(data['C']) + 1)),
        format_func=lambda x: f"Case #{x}"
    )
    
    if case_id:
        idx = case_id - 1
        instance = data['C'][idx]
        scores = calculate_instance_scores(
            instance, 
            patterns_A, 
            patterns_B,
            pure_patterns_A,
            pure_patterns_B
        )
        
        fig = create_interactive_sankey(scores, f"Analysis Flow for Case #{case_id}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed scores
        show_detailed_scores(scores)

def create_pattern_analysis(data):
    """Create interactive pattern analysis visualization"""
    st.subheader("Pattern Analysis")
    
    # Create tabs for different pattern views
    pattern_tabs = st.tabs(["Frequency", "Correlation", "Time Series"])
    
    with pattern_tabs[0]:
        create_pattern_frequency_chart(data)
    
    with pattern_tabs[1]:
        create_pattern_correlation_matrix(data)
    
    with pattern_tabs[2]:
        create_pattern_timeline(data)

def create_results_section():
    """Enhanced results section with filtering and export options"""
    st.header("Analysis Results")
    
    if not st.session_state.processed_data:
        st.warning("Please upload and process data first.")
        return
    
    data = st.session_state.processed_data
    risk_analyzer = RiskAnalyzer()
    
    # Create filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Risk Level",
            ["Very High", "High", "Low", "Very Low"],
            default=["Very High", "High"]
        )
    
    with col2:
        score_range = st.slider(
            "Score Range",
            min_value=0,
            max_value=5000,
            value=(0, 5000)
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Risk Level", "Score", "ID"]
        )
    
    # Create and display filtered results
    filtered_results = create_filtered_results(
        data,
        risk_analyzer,
        risk_filter,
        score_range,
        sort_by
    )
    
    # Display results table with styling
    st.markdown("""
        <style>
        .risk-table {
            font-family: 'Inter', sans-serif;
        }
        .risk-table th {
            background-color: #f8fafc;
            padding: 12px;
        }
        .risk-table td {
            padding: 8px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        filtered_results.style.apply(style_risk_levels, axis=1),
        height=500,
        use_container_width=True
    )
    
    # Export options
    export_format = st.radio(
        "Export Format",
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )
    
    if st.button("Export Results"):
        export_results(filtered_results, export_format)

def style_risk_levels(row):
    """Apply conditional styling to risk levels"""
    risk_colors = {
        "Very High": "#fee2e2",  # Red-50
        "High": "#fff7ed",       # Orange-50
        "Low": "#f0fdf4",        # Green-50
        "Very Low": "#f8fafc"    # Slate-50
    }
    
    risk_level = row["Risk Level"]
    color = risk_colors.get(risk_level, "#ffffff")
    
    return [f"background-color: {color}"] * len(row)

def export_results(results_df, format_type):
    """Export results in selected format"""
    try:
        if format_type == "CSV":
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv_data,
                "diagnosis_analysis.csv",
                "text/csv"
            )
        elif format_type == "Excel":
            excel_buffer = io.BytesIO()
            results_df.to_excel(excel_buffer, index=False)
            st.download_button(
                "Download Excel",
                excel_buffer.getvalue(),
                "diagnosis_analysis.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            json_data = results_df.to_json(orient="records")
            st.download_button(
                "Download JSON",
                json_data,
                "diagnosis_analysis.json",
                "application/json"
            )
            
        st.success("Export completed successfully!")
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def create_filtered_results(data, risk_analyzer, risk_filter, score_range, sort_by):
    """Create filtered and sorted results dataframe"""
    results = []
    
    for idx, instance in enumerate(data['C']):
        risk_score = risk_analyzer.calculate_risk_score(tuple(instance))
        risk_level, _ = risk_analyzer.get_risk_level(risk_score)
        
        if (risk_level in risk_filter and 
            score_range[0] <= risk_score <= score_range[1]):
            results.append({
                "ID": idx + 1,
                "Risk Level": risk_level,
                "Risk Score": risk_score,
                "Class": data['ClassT'][idx]
            })
    
    results_df = pd.DataFrame(results)
    
    if sort_by == "Risk Level":
        risk_order = ["Very High", "High", "Low", "Very Low"]
        results_df["Risk Level"] = pd.Categorical(
            results_df["Risk Level"],
            categories=risk_order,
            ordered=True
        )
    
    return results_df.sort_values(by=sort_by, ascending=sort_by=="ID")

if __name__ == "__main__":
    main()
