import streamlit as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool

st.set_page_config(page_title="Misdiagnosis Detection Tool", page_icon="m.jpg", layout='wide', initial_sidebar_state='expanded')

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
        st.error(f"Error: cla ({cla}) exceeds the number of columns in the DataFrame ({df.shape[1]})")
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
        st.error(f"Error: cla ({cla}) exceeds the number of columns in the DataFrame ({df.shape[1]})")
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
    
    R = [tuple(U[i][e] for e in r) for i,
