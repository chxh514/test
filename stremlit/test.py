import streamlit as st
from collections import Counter, defaultdict
#from PP import tuples_to_boolean_arrays, calculate_unique_intersections_parallel, calculate_scores_parallel, identify_pure_sets_numpy
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool
#from collections import defaultdict
from itertools import combinations

st.set_page_config(page_title="BBBBB", page_icon="h.png", layout='wide', initial_sidebar_state='auto')

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

##tabs[2]
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

def to_bool_arr(tuples, size=9):
    return np.array([np.isin(range(size), t) for t in tuples])

def SS(arr, sets):
    return any(np.all(np.logical_and(arr, s)) for s in sets)

def SSB(ins, b):
    return np.all(np.logical_or(ins == b, ins == False))
    
# Default Parameters
cla, rat1, rat2, rof, ski = 9, 0.85, 0.15, 0, 0  # 调整后的 cla

#st.title("Data Preprocessing and Method Execution")


def main():
    """簡單的誤診檢測工具"""
    html_templ = """
    <div style="background-color: #C9E2F2;padding:10px;">
    <h1 style="color: #324BD9">Misdiagnosis Detection Tool</h1>
    </div>
    """
    st.markdown(html_templ, unsafe_allow_html=True)

    tabs = st.tabs(['Upload Files', '**DataFrame**',"Detection of Misdiagnosis","Sankey diagram","Functions"])

    with tabs[0]:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            result = DataPreprocessing(uploaded_file)
            if result is not None:
                acc, A, B, C, IdT, ClassT = result
                Method(acc, A, B, C, IdT, ClassT)

    with tabs[1]:
        uploaded_files = st.file_uploader("**上傳所要檢測的數據**", type = ['csv'])

        if uploaded_files is not None:
            df = pd.read_csv(uploaded_files, sep=',', header=None, skiprows=1)

            st.write("Data Preview:")
            st.write(df)

            L = df.T.values.tolist()
            
            # 獲取不同分數和標籤
            ANS, ScoreA, ScoreB = np.array(L[1]), np.array(L[2]), np.array(L[3])
            
            # 計算 ScoreA 的 ROC 曲線和 AUC
            fpr_A, tpr_A, thresholds_A = metrics.roc_curve(ANS, ScoreA, pos_label=2)
            auc_scoreA = metrics.auc(fpr_A, tpr_A)
            st.write(f"AUC_ScoreA = {auc_scoreA}")
            
            # 計算 ScoreB 的 ROC 曲線和 AUC
            fpr_B, tpr_B, thresholds_B = metrics.roc_curve(ANS, ScoreB, pos_label=4)
            auc_scoreB = metrics.auc(fpr_B, tpr_B)
            st.write(f"AUC_ScoreB = {auc_scoreB}")

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=fpr_A, y=tpr_A, mode='lines', name=f'ScoreA (AUC = {auc_scoreA:.2f})',
                             line=dict(color='red', width=2)))

            # ScoreB 的 ROC 曲線
            fig.add_trace(go.Scatter(x=fpr_B, y=tpr_B, mode='lines', name=f'ScoreB (AUC = {auc_scoreB:.2f})',
                                    line=dict(color='blue', width=2)))

            # 添加一條對角線代表隨機猜測
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess',
                                    line=dict(color='gray', dash='dash')))

            # 設定圖表標題和軸標籤
            fig.update_layout(title='ROC Curve',
                            xaxis_title='False Positive Rate (FPR)',
                            yaxis_title='True Positive Rate (TPR)',
                            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                            width=800, height=600)

            st.plotly_chart(fig)


    with tabs[2]:
        #uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
      
        # Analyze patterns and find pure patterns for A and B
            patterns_A = find_patterns_updated(A)
            patterns_B = find_patterns_updated(B)
            pure_patterns_A = find_pure_patterns(patterns_A, B)
            pure_patterns_B = find_pure_patterns(patterns_B, A)

        # Find instances in C that satisfy the specified conditions
            specific_instances_C = find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B)

        # Adjusted print statement
            st.write("Instances in C that satisfy the specified conditions:")
        #for instance in specific_instances_C:
        #    c, score_A, score_B, pure_score_A, pure_score_B = instance
        #    st.write(f"Instance in C: {c},  Score_A ({score_A}),  Score_B ({score_B}),  Pure_Score_A ({pure_score_A}),  Pure_Score_B ({pure_score_B})")
#        for i in specific_instances_C:
#            print(i)
#            print("\n")

    with tabs[4]:
#        choice = st.selectbox("Functions", [" ", "Sankey diagram", "ROC/AUC", "Precision-Recall"])
#        if choice == "Sankey diagram":
        choice = st.selectbox("Functions", [" ", "Data 1", "Data 2", "Data 3"])
        if choice == "Data 1":
            st.subheader("RESULT")
            # Define the source, target, and value arrays for the Sankey diagram
            #c, score_A, score_B, pure_score_A, pure_score_B = ((3, 5, 8, 15, 17, 29, 33, 38, 42), (6917, [[{42}, 1498], [{33}, 86], [{33, 42}, 1572], [{8, 33, 42}, 162], [{8, 42}, 400], [{33, 42, 38}, 189], [{8, 33, 42, 38}, 16], [{42, 38}, 156], [{33, 42, 3}, 1044], [{3}, 12], [{17, 42}, 136], [{17}, 13], [{33, 42, 17}, 27], [{8, 33}, 44], [{8, 33, 42, 3}, 96], [{42, 3}, 980], [{33, 42, 3, 29}, 64], [{8}, 26], [{8, 42, 3}, 144], [{8, 42, 38}, 18], [{8, 3, 38}, 9], [{17, 42, 3}, 72], [{42, 3, 29}, 27], [{17, 42, 3, 29}, 16], [{38}, 4], [{42, 29}, 16], [{8, 17, 42}, 45], [{8, 17}, 8], [{8, 17, 42, 3}, 16], [{3, 29}, 4], [{17, 42, 38}, 9], [{8, 3}, 4], [{33, 17}, 4]]), (6416, [[{42}, 885], [{33, 42}, 432], [{33, 42, 38}, 288], [{8}, 61], [{8, 33, 38}, 18], [{33}, 156], [{8, 29, 38}, 9], [{8, 42}, 320], [{29}, 7], [{8, 33, 42}, 108], [{38}, 75], [{42, 38}, 172], [{42, 29}, 12], [{33, 38, 8, 42, 29}, 25], [{33, 42, 29}, 27], [{5}, 95], [{15}, 56], [{5, 15}, 20], [{33, 38}, 44], [{42, 5}, 220], [{33, 42, 17}, 207], [{17}, 111], [{33, 17}, 72], [{17, 5}, 72], [{17, 42}, 564], [{33, 42, 5}, 45], [{17, 42, 5}, 117], [{33, 42, 5, 17}, 64], [{8, 38, 15}, 36], [{8, 33}, 44], [{5, 38}, 28], [{8, 5}, 16], [{3, 5}, 12], [{8, 42, 3, 5}, 16], [{17, 29}, 8], [{8, 17, 42, 29}, 16], [{8, 17, 42}, 126], [{33, 5}, 56], [{33, 5, 17}, 27], [{42, 3, 5}, 18], [{17, 42, 3}, 27], [{33, 42, 3}, 18], [{3}, 9], [{33, 3, 5, 42, 17}, 25], [{5, 29}, 4], [{8, 38}, 40], [{8, 33, 42, 38}, 32], [{33, 42, 29, 17}, 16], [{8, 17}, 80], [{17, 38}, 16], [{33, 38, 17}, 99], [{8, 33, 38, 17}, 48], [{33, 42, 15}, 45], [{8, 33, 5}, 18], [{8, 17, 5}, 36], [{17, 5, 38}, 18], [{33, 5, 38}, 18], [{8, 5, 38}, 18], [{33, 5, 38, 17}, 64], [{33, 5, 38, 8, 17}, 25], [{42, 3}, 16], [{33, 3}, 8], [{33, 42, 3, 38}, 16], [{8, 3}, 4], [{8, 42, 3}, 9], [{8, 42, 29}, 9], [{33, 38, 8, 17, 29}, 50], [{33, 29}, 8], [{42, 15}, 60], [{42, 3, 15}, 9], [{42, 3, 5, 15}, 16], [{8, 33, 42, 5}, 16], [{8, 15}, 24], [{17, 5, 15}, 18], [{33, 15}, 40], [{8, 42, 15}, 54], [{33, 5, 15}, 18], [{17, 42, 15}, 27], [{8, 42, 5}, 9], [{17, 42, 5, 38}, 16], [{8, 17, 42, 38}, 16], [{8, 29, 38, 15}, 16], [{33, 42, 38, 17}, 48], [{8, 33, 17}, 27], [{33, 38, 8, 42, 17}, 50], [{3, 8, 42, 15, 17}, 25], [{8, 42, 38}, 27], [{42, 5, 38}, 18], [{33, 5, 8, 42, 15}, 25], [{42, 5, 15}, 18], [{33, 42, 5, 38}, 32], [{33, 5, 38, 42, 17}, 50], [{33, 3, 15}, 9], [{33, 3, 38, 8, 42, 15}, 36], [{42, 38, 15}, 27], [{33, 3, 5, 15}, 16], [{8, 3, 15}, 9], [{3, 38}, 4], [{8, 17, 42, 15}, 16], [{33, 3, 5, 38, 8, 42, 15, 17, 29}, 81], [{8, 33, 42, 17}, 32], [{17, 42, 5, 15}, 16], [{17, 42, 38}, 9], [{5, 38, 42, 15, 17}, 25], [{17, 42, 3, 5}, 16]]), (0, []), (1344, [[{8, 29, 38}, 9], [{33, 38, 8, 42, 29}, 25], [{5, 15}, 20], [{17, 5}, 72], [{17, 42, 5}, 117], [{33, 42, 5, 17}, 64], [{8, 38, 15}, 36], [{8, 5}, 16], [{3, 5}, 12], [{8, 42, 3, 5}, 16], [{33, 5, 17}, 27], [{42, 3, 5}, 18], [{33, 3, 5, 42, 17}, 25], [{5, 29}, 4], [{8, 33, 38, 17}, 48], [{33, 42, 15}, 45], [{8, 33, 5}, 18], [{8, 17, 5}, 36], [{17, 5, 38}, 18], [{8, 5, 38}, 18], [{33, 5, 38, 17}, 64], [{33, 5, 38, 8, 17}, 25], [{33, 38, 8, 17, 29}, 50], [{42, 3, 15}, 9], [{42, 3, 5, 15}, 16], [{8, 33, 42, 5}, 16], [{17, 5, 15}, 18], [{33, 15}, 40], [{33, 5, 15}, 18], [{8, 42, 5}, 9], [{17, 42, 5, 38}, 16], [{8, 29, 38, 15}, 16], [{33, 38, 8, 42, 17}, 50], [{3, 8, 42, 15, 17}, 25], [{33, 5, 8, 42, 15}, 25], [{42, 5, 15}, 18], [{33, 5, 38, 42, 17}, 50], [{33, 3, 15}, 9], [{33, 3, 38, 8, 42, 15}, 36], [{42, 38, 15}, 27], [{33, 3, 5, 15}, 16], [{8, 3, 15}, 9], [{33, 3, 5, 38, 8, 42, 15, 17, 29}, 81], [{17, 42, 5, 15}, 16], [{5, 38, 42, 15, 17}, 25], [{17, 42, 3, 5}, 16]]))  
            c, score_A, score_B, pure_score_A, pure_score_B = specific_instances_C[150]
            source = [0 for _ in range(2)] + [1 for _ in range(len(score_A[1]))] + [2 for _ in range(len(score_B[1]))]

            target = [i for i in range(1, len(source)+1)]

            value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]

            # Define labels for the nodes
            label = ['PATIENT:488', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in score_A[1]] + ['N'+str(i[0]) for i in score_B[1]]

            # Define node colors
            node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * 4 + ['#F1F8E9'] * 4 + ['#ECEFF1']

            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="#37474F", width=0.5),label=label),link=dict(source=source,target=target,value=value,color=node_colors[1:2] + node_colors[2:3] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])))])  # Use colors for links similar to node colors

            #fig.update_layout(title_text="Sankey Diagram of Test Results", font_size=6)
            #fig.show()
            st.plotly_chart(fig)

        if choice == "Data 2":
            st.subheader("RESULT")
            # Define the source, target, and value arrays for the Sankey diagram
            #c, score_A, score_B, pure_score_A, pure_score_B = ((3, 5, 8, 15, 17, 29, 33, 38, 42), (6917, [[{42}, 1498], [{33}, 86], [{33, 42}, 1572], [{8, 33, 42}, 162], [{8, 42}, 400], [{33, 42, 38}, 189], [{8, 33, 42, 38}, 16], [{42, 38}, 156], [{33, 42, 3}, 1044], [{3}, 12], [{17, 42}, 136], [{17}, 13], [{33, 42, 17}, 27], [{8, 33}, 44], [{8, 33, 42, 3}, 96], [{42, 3}, 980], [{33, 42, 3, 29}, 64], [{8}, 26], [{8, 42, 3}, 144], [{8, 42, 38}, 18], [{8, 3, 38}, 9], [{17, 42, 3}, 72], [{42, 3, 29}, 27], [{17, 42, 3, 29}, 16], [{38}, 4], [{42, 29}, 16], [{8, 17, 42}, 45], [{8, 17}, 8], [{8, 17, 42, 3}, 16], [{3, 29}, 4], [{17, 42, 38}, 9], [{8, 3}, 4], [{33, 17}, 4]]), (6416, [[{42}, 885], [{33, 42}, 432], [{33, 42, 38}, 288], [{8}, 61], [{8, 33, 38}, 18], [{33}, 156], [{8, 29, 38}, 9], [{8, 42}, 320], [{29}, 7], [{8, 33, 42}, 108], [{38}, 75], [{42, 38}, 172], [{42, 29}, 12], [{33, 38, 8, 42, 29}, 25], [{33, 42, 29}, 27], [{5}, 95], [{15}, 56], [{5, 15}, 20], [{33, 38}, 44], [{42, 5}, 220], [{33, 42, 17}, 207], [{17}, 111], [{33, 17}, 72], [{17, 5}, 72], [{17, 42}, 564], [{33, 42, 5}, 45], [{17, 42, 5}, 117], [{33, 42, 5, 17}, 64], [{8, 38, 15}, 36], [{8, 33}, 44], [{5, 38}, 28], [{8, 5}, 16], [{3, 5}, 12], [{8, 42, 3, 5}, 16], [{17, 29}, 8], [{8, 17, 42, 29}, 16], [{8, 17, 42}, 126], [{33, 5}, 56], [{33, 5, 17}, 27], [{42, 3, 5}, 18], [{17, 42, 3}, 27], [{33, 42, 3}, 18], [{3}, 9], [{33, 3, 5, 42, 17}, 25], [{5, 29}, 4], [{8, 38}, 40], [{8, 33, 42, 38}, 32], [{33, 42, 29, 17}, 16], [{8, 17}, 80], [{17, 38}, 16], [{33, 38, 17}, 99], [{8, 33, 38, 17}, 48], [{33, 42, 15}, 45], [{8, 33, 5}, 18], [{8, 17, 5}, 36], [{17, 5, 38}, 18], [{33, 5, 38}, 18], [{8, 5, 38}, 18], [{33, 5, 38, 17}, 64], [{33, 5, 38, 8, 17}, 25], [{42, 3}, 16], [{33, 3}, 8], [{33, 42, 3, 38}, 16], [{8, 3}, 4], [{8, 42, 3}, 9], [{8, 42, 29}, 9], [{33, 38, 8, 17, 29}, 50], [{33, 29}, 8], [{42, 15}, 60], [{42, 3, 15}, 9], [{42, 3, 5, 15}, 16], [{8, 33, 42, 5}, 16], [{8, 15}, 24], [{17, 5, 15}, 18], [{33, 15}, 40], [{8, 42, 15}, 54], [{33, 5, 15}, 18], [{17, 42, 15}, 27], [{8, 42, 5}, 9], [{17, 42, 5, 38}, 16], [{8, 17, 42, 38}, 16], [{8, 29, 38, 15}, 16], [{33, 42, 38, 17}, 48], [{8, 33, 17}, 27], [{33, 38, 8, 42, 17}, 50], [{3, 8, 42, 15, 17}, 25], [{8, 42, 38}, 27], [{42, 5, 38}, 18], [{33, 5, 8, 42, 15}, 25], [{42, 5, 15}, 18], [{33, 42, 5, 38}, 32], [{33, 5, 38, 42, 17}, 50], [{33, 3, 15}, 9], [{33, 3, 38, 8, 42, 15}, 36], [{42, 38, 15}, 27], [{33, 3, 5, 15}, 16], [{8, 3, 15}, 9], [{3, 38}, 4], [{8, 17, 42, 15}, 16], [{33, 3, 5, 38, 8, 42, 15, 17, 29}, 81], [{8, 33, 42, 17}, 32], [{17, 42, 5, 15}, 16], [{17, 42, 38}, 9], [{5, 38, 42, 15, 17}, 25], [{17, 42, 3, 5}, 16]]), (0, []), (1344, [[{8, 29, 38}, 9], [{33, 38, 8, 42, 29}, 25], [{5, 15}, 20], [{17, 5}, 72], [{17, 42, 5}, 117], [{33, 42, 5, 17}, 64], [{8, 38, 15}, 36], [{8, 5}, 16], [{3, 5}, 12], [{8, 42, 3, 5}, 16], [{33, 5, 17}, 27], [{42, 3, 5}, 18], [{33, 3, 5, 42, 17}, 25], [{5, 29}, 4], [{8, 33, 38, 17}, 48], [{33, 42, 15}, 45], [{8, 33, 5}, 18], [{8, 17, 5}, 36], [{17, 5, 38}, 18], [{8, 5, 38}, 18], [{33, 5, 38, 17}, 64], [{33, 5, 38, 8, 17}, 25], [{33, 38, 8, 17, 29}, 50], [{42, 3, 15}, 9], [{42, 3, 5, 15}, 16], [{8, 33, 42, 5}, 16], [{17, 5, 15}, 18], [{33, 15}, 40], [{33, 5, 15}, 18], [{8, 42, 5}, 9], [{17, 42, 5, 38}, 16], [{8, 29, 38, 15}, 16], [{33, 38, 8, 42, 17}, 50], [{3, 8, 42, 15, 17}, 25], [{33, 5, 8, 42, 15}, 25], [{42, 5, 15}, 18], [{33, 5, 38, 42, 17}, 50], [{33, 3, 15}, 9], [{33, 3, 38, 8, 42, 15}, 36], [{42, 38, 15}, 27], [{33, 3, 5, 15}, 16], [{8, 3, 15}, 9], [{33, 3, 5, 38, 8, 42, 15, 17, 29}, 81], [{17, 42, 5, 15}, 16], [{5, 38, 42, 15, 17}, 25], [{17, 42, 3, 5}, 16]]))  
            c, score_A, score_B, pure_score_A, pure_score_B = specific_instances_C[2]
            source = [0 for _ in range(2)] + [1 for _ in range(len(score_A[1]))] + [2 for _ in range(len(score_B[1]))]

            target = [i for i in range(1, len(source)+1)]

            value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]

            # Define labels for the nodes
            label = ['PATIENT:488', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in score_A[1]] + ['N'+str(i[0]) for i in score_B[1]]

            # Define node colors
            node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * 4 + ['#F1F8E9'] * 4 + ['#ECEFF1']

            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="#37474F", width=0.5),label=label),link=dict(source=source,target=target,value=value,color=node_colors[1:2] + node_colors[2:3] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])))])  # Use colors for links similar to node colors

            #fig.update_layout(title_text="Sankey Diagram of Test Results", font_size=6)
            #fig.show()
            st.plotly_chart(fig)



            # 顯示 ROC 曲線圖表
            #st.line_chart({"FPR (ScoreA)": fpr_A, "TPR (ScoreA)": tpr_A, "FPR (ScoreB)": fpr_B, "TPR (ScoreB)": tpr_B})

#        if uploaded_file:
#            if uploaded_file.name.endswith('.csv'):
#                df = pd.read_csv(uploaded_file)
#                st.success("檔案上傳成功！")

                
#        else:
#            df = pd.read_excel(uploaded_file)
        #df=pd.read_csv(uploaded_file, sep=',', header=None, skiprows=1)
        #st.write(df.head())

                #L=df.T.values.tolist()
                #ANS, ScoreA, ScoreB,=np.array(L[1]), np.array(L[2]), np.array(L[3])
                #fpr, tpr, thresholds = metrics.roc_curve(ANS, ScoreA, pos_label=2)
                #st.write(f"AUC_ScoreA={metrics.auc(fpr, tpr)}")
                #fpr, tpr, thresholds = metrics.roc_curve(ANS, ScoreB, pos_label=4)
                #st.write(f"AUC_ScoreB={metrics.auc(fpr, tpr)}")
        
    


#uploaded_file = st.sidebar.file_uploader("上傳 CSV 或 Excel 檔案", type=['csv', 'xlsx'])

#if uploaded_file:
#    if uploaded_file.name.endswith('.csv'):
#        df = pd.read_csv(uploaded_file)
#    else:
#        df = pd.read_excel(uploaded_file)
        
#    st.sidebar.success("檔案上傳成功！")

#if st.sidebar.button("顯示 DataFrame"):
#    if 'df' in locals():
#            
        #styled_df = df.style.apply(highlight_risk, axis=1)
#        st.subheader("上傳的數據")
        #st.write(styled_df)
#        st.write(df)
        #AgGrid(df)
            

#    else:
#        st.warning("請先上傳檔案！")

if __name__ == '__main__':
    main()
