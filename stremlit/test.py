import streamlit as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool
from stremlit.test import find_specific_instances

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

# Utility Functions
def zscore(value):
    """Calculate Z-score, return 0 if value is 0"""
    return 0.0 if value == 0.0 else value

def tuples_to_boolean_arrays(tuples, max_value):
    """Convert tuples to boolean arrays"""
    return np.array([np.isin(range(max_value), t) for t in tuples])

def calculate_score(instance, pure_sets):
    """Calculate score for a single instance"""
    return sum(len(ps)**2 for ps in pure_sets if set(ps).issubset(set(instance)))

def calculate_scores_parallel(instances, pure_sets, num_processes=4):
    """Calculate scores in parallel"""
    with Pool(num_processes) as pool:
        return pool.starmap(calculate_score, [(instance, pure_sets) for instance in instances])

def identify_pure_sets_numpy(intersections, other_bool, max_value):
    """Identify pure sets using numpy arrays"""
    pure_sets = []
    for intersection in intersections:
        intersection_bool = np.isin(range(max_value), intersection)
        if not np.any(np.all(intersection_bool <= other_bool, axis=-1)):
            pure_sets.append(intersection)
    return pure_sets

def calculate_unique_intersections_single(array):
    """Calculate unique intersections for a single array"""
    intersections = np.bitwise_and(array[:, None, :], array)
    return {tuple(np.where(intersections[i, j])[0]) 
            for i in range(intersections.shape[0]) 
            for j in range(intersections.shape[1])}

def calculate_unique_intersections_parallel(bool_arrays, num_processes=4):
    """Calculate unique intersections in parallel"""
    with Pool(num_processes) as pool:
        results = pool.map(calculate_unique_intersections_single, [bool_arrays])
    return set.union(*results)

def find_patterns_updated(data):
    """Find patterns in the data"""
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (v[0], v[1]) for k, v in pattern_counts.items()}

def find_pure_patterns(patterns, other_data):
    """Find pure patterns in one dataset with respect to another"""
    other_sets = [set(item) for item in other_data]
    return {pattern: data for pattern, data in patterns.items()
            if not any(set(pattern).issubset(other_set) for other_set in other_sets)}

def get_score_of_instance(instance, patterns):
    """Get score of an instance based on patterns"""
    score = 0
    pattern_info = []
    instance_set = set(instance)
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_info.append([set(pattern), data[0]])
    return score, pattern_info

def get_pure_score_of_instance(instance, pure_patterns):
    """Get pure score of an instance based on pure patterns"""
    return get_score_of_instance(instance, pure_patterns)

def find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    """Find instances in C that satisfy specified conditions"""
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

def preprocess_data(uploaded_file):
    """Preprocess the uploaded data file"""
    start_time = time.time()
    
    # Read data
    df = pd.read_csv(uploaded_file, sep=',', header=None, skiprows=SKIP_ROWS)
    if CLASS_COLUMN >= df.shape[1]:
        st.error(f"Error: Class column ({CLASS_COLUMN}) exceeds DataFrame columns ({df.shape[1]})")
        return None
    
    # Handle missing values
    df.fillna(value='NotNumber', inplace=True)
    
    # Initialize variables
    acc = 0
    itr1 = df.shape[0] * TRAIN_RATIO
    itr2 = df.shape[0] * TEST_RATIO
    D_IdClass = df.iloc[:, CLASS_COLUMN].to_dict()
    L = df.T.values.tolist()
    R = []
    U = []
    
    # Process each column
    for c in [L[l] for l in range(len(L)-1)]:
        D_Num, D_Cat = {}, {}
        L_All = [0] * len(c)
        
        # Separate numeric and categorical values
        for e, val in enumerate(c):
            if isinstance(val, str):
                D_Cat[e] = val
            else:
                D_Num[e] = val
        
        # Calculate Z-scores for numeric values
        if D_Num:
            L_Num = list(D_Num.values())
            ave, std = np.mean(L_Num), np.std(L_Num)
            for n, v in D_Num.items():
                L_All[n] = zscore(round((v - ave) / std, ROUND_DIGITS))
        
        # Add categorical values
        for i, v in D_Cat.items():
            L_All[i] = v
            
        R.append(L_All)
    
    # Create index mapping
    for r in R:
        unique_vals = set(r)
        index_map = {e: i + acc for i, e in enumerate(unique_vals)}
        acc += len(unique_vals)
        U.append(index_map)
    
    # Convert to indexed format
    R = [tuple(U[i][e] for e in r) for i, r in enumerate(R)]
    
    # Identify normal and contradiction instances
    V = [tuple(v) for v in pd.DataFrame(R).T.values.tolist()]
    W = defaultdict(set)
    for v_idx, v in enumerate(V):
        W[v].add(D_IdClass[v_idx])
    
    N = [(v_idx, v, D_IdClass[v_idx]) for v_idx, v in enumerate(V) if len(W[v]) == 1]
    
    # Split data into training and testing sets
    G = list(set(i[2] for i in N))
    A = [i for i in N if i[0] < itr1 and i[2] == G[0]]
    B = [i for i in N if i[0] < itr1 and i[2] == G[1]]
    C = [i for i in N if i[0] >= itr2]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]

def create_sankey_diagram(instance_data, title):
    """Create a Sankey diagram for the given instance data"""
    c, score_A, score_B, pure_score_A, pure_score_B = instance_data
    
    # Prepare data for Sankey diagram
    source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
    target = ([1, 2] + 
             list(range(3, 3 + len(score_A[1]))) + 
             list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1]))))
    value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]
    
    # Create node labels
    labels = (['PATIENT', 'Positive P', 'Negative N'] + 
             [f'P{i[0]}' for i in score_A[1]] + 
             [f'N{i[0]}' for i in score_B[1]])
    
    # Define colors
    node_colors = (['#ECEFF1', '#F8BBD0', '#DCEDC8'] + 
                  ['#FFEBEE'] * len(score_A[1]) + 
                  ['#F1F8E9'] * len(score_B[1]))
    
    # Create Sankey diagram
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
            color=node_colors[1:2] + node_colors[2:3] + 
                  ['#FFEBEE'] * len(score_A[1]) + 
                  ['#F1F8E9'] * len(score_B[1])
        )
    )])
    
    fig.update_layout(title=title)
    return fig

def highlight_risk(row):
    """Apply conditional formatting based on risk level"""
    risk = row.get('Misdiagnosis Risk', '')
    colors = {
        "Very High": '#ff4c4c',
        "High": '#ffd966',
        "Low": '#c6efce',
        "Very Low": '#aec6cf'
    }
    return [f'background-color: {colors.get(risk, "")}'] * len(row)

def main():
    """Main application function"""
    # Header
    st.markdown("""
        <div style="background-color: #C9E2F2;padding:10px;">
        <h1 style="color: #324BD9">Misdiagnosis Detection Tool</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs(['Upload Files', 'DataFrame', "Detection of Misdiagnosis", 
                    "Sankey diagram", "Functions"])
    
    with tabs[0]:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            result = preprocess_data(uploaded_file)
            if result:
                acc, A, B, C, IdT, ClassT = result
                Method(acc, A, B, C, IdT, ClassT)
    
    with tabs[1]:
        uploaded_files = st.file_uploader("**Upload data for analysis**", type=['csv'])
        if uploaded_files:
            df = pd.read_csv(uploaded_files, sep=',', header=None, skiprows=1)
            
            # Display data preview and analysis
            st.write("Data Preview:")
            st.write(df)
            
            # Calculate and display ROC curves
            L = df.T.values.tolist()
            ANS, ScoreA, ScoreB = np.array(L[1]), np.array(L[2]), np.array(L[3])
            
            # Calculate ROC curves and AUC scores
            fpr_A, tpr_A, _ = metrics.roc_curve(ANS, ScoreA, pos_label=2)
            fpr_B, tpr_B, _ = metrics.roc_curve(ANS, ScoreB, pos_label=4)
            auc_scoreA = metrics.auc(fpr_A, tpr_A)
            auc_scoreB = metrics.auc(fpr_B, tpr_B)
            
            # Create ROC curve plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, name=f'ScoreA (AUC = {auc_scoreA:.2f})',
                                   mode='lines', line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, name=f'ScoreB (AUC = {auc_scoreB:.2f})',
                                   mode='lines', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess',
                                   mode='lines', line=dict(color='gray', dash='dash')))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate (FPR)',
                yaxis_title='True Positive Rate (TPR)',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                width=800, height=600
            )
            
            st.plotly_chart(fig)
    
    with tabs[2]:
        if 'specific_instances_C' in locals():
            st.write(f"Number of instances meeting criteria: {len(specific_instances_C)}")
    
    with tabs[3]:
        if 'specific_instances_C' in locals():
            choices = [f"Data {i+1}" for i in range(len(specific_instances_C))]
            choice = st.selectbox("Select Data", [" "] + choices)
            
            if choice != " ":
                index = int(choice.split(" ")[1]) - 1
                st.subheader("RESULT")
                
                # Create and display Sankey diagrams
                regular_fig = create_sankey_diagram(
                    specific_instances_C[index],
                    "Regular Analysis"
                )
                st.plotly_chart(regular_fig)
                
# Continue from previous Sankey diagram code...
                pure_fig = create_sankey_diagram(
                    (specific_instances_C[index][0],
                     specific_instances_C[index][3],  # pure_score_A
                     specific_instances_C[index][4],  # pure_score_B
                     specific_instances_C[index][3],
                     specific_instances_C[index][4]),
                    "Pure Analysis"
                )
                st.plotly_chart(pure_fig)
    
    with tabs[4]:
        st.subheader("Misdiagnosis Risk Table")
        
        if 'specific_instances_C' in locals() and 'ClassT' in locals():
            # Prepare risk assessment data
            risk_data = []
            for idx, (c, score_A, score_B, pure_score_A, pure_score_B) in enumerate(specific_instances_C):
                # Calculate risk score based on pure scores
                risk_score = max(pure_score_A[0], pure_score_B[0])
                
                # Determine risk level and status
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
                
                # Add data to risk assessment table
                risk_data.append({
                    "Status": status,
                    "ID": idx + 1,
                    "NS": pure_score_A[0],
                    "PS": pure_score_B[0],
                    "Label": ClassT[idx],
                    "Misdiagnosis Risk": risk_level
                })
            
            # Create and display styled risk assessment table
            risk_df = pd.DataFrame(risk_data)
            styled_risk_df = risk_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_risk_df, use_container_width=False, height=600)

def Method(acc, A, B, C, IdT, ClassT):
    """Process and analyze data using the specified method"""
    start_time = time.time()
    
    # Convert data to boolean arrays
    A_bool = tuples_to_boolean_arrays(A, acc)
    B_bool = tuples_to_boolean_arrays(B, acc)
    
    # Calculate intersections and pure sets
    A_intersections = calculate_unique_intersections_parallel(A_bool)
    B_intersections = calculate_unique_intersections_parallel(B_bool)
    
    # Identify pure sets
    pure_sets_A = identify_pure_sets_numpy(A_intersections, B_bool, acc)
    pure_sets_B = identify_pure_sets_numpy(B_intersections, A_bool, acc)
    
    # Calculate scores
    SA = calculate_scores_parallel(C, pure_sets_A)
    SB = calculate_scores_parallel(C, pure_sets_B)
    
    # Create results DataFrame
    result_df = pd.DataFrame({
        'ID': IdT,
        'Class': ClassT,
        'A': SA,
        'B': SB,
        'Items': C
    })
    
    # Display results and processing time
    end_time = time.time()
    st.write(f"Method Complete! Time={end_time - start_time:.2f}s")
    st.write("Result DataFrame:", result_df)
    
    # Provide download option
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='analysis_results.csv',
        mime='text/csv'
    )
    
    return result_df

if __name__ == '__main__':
    main()
