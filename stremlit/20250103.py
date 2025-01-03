import streamlit as st
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from multiprocessing import Pool

st.set_page_config(page_title="誤診檢測工具", page_icon="m.jpg", layout='wide', initial_sidebar_state='expanded')

# 輔助函數：將元組轉換為布林陣列
def tuples_to_boolean_arrays(tuples, max_value):
    return np.array([np.isin(range(max_value), t) for t in tuples])

# 輔助函數：計算單個實例的分數
def calculate_score(instance, pure_sets):
    score = 0
    for ps in pure_sets:
        if set(ps).issubset(set(instance)):
            score += len(ps)**2
    return score

# 輔助函數：平行計算多個實例的分數
def calculate_scores_parallel(instances, pure_sets, num_processes=4):
    with Pool(num_processes) as pool:
        scores = pool.starmap(calculate_score, [(instance, pure_sets) for instance in instances])
    return scores

# 輔助函數：以NumPy格式識別純集合
def identify_pure_sets_numpy(intersections, other_bool, max_value):
    pure_sets = []
    for intersection in intersections:
        intersection_bool = np.isin(range(max_value), intersection)
        if not np.any(np.all(intersection_bool <= other_bool, axis=-1)):
            pure_sets.append(intersection)
    return pure_sets

# 輔助函數：計算單個陣列的唯一交集
def calculate_unique_intersections_single(array):
    intersections = np.bitwise_and(array[:, None, :], array)
    unique_intersections = set()
    for i in range(intersections.shape[0]):
        for j in range(intersections.shape[1]):
            intersection = tuple(np.where(intersections[i, j])[0])
            unique_intersections.add(intersection)
    return unique_intersections

# 輔助函數：平行計算多個陣列的唯一交集
def calculate_unique_intersections_parallel(bool_arrays, num_processes=4):
    with Pool(num_processes) as pool:
        results = pool.map(calculate_unique_intersections_single, [bool_arrays]) #修正此處
    return set.union(*results)

# 輔助函數：尋找模式
def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(set(data[i]) & set(data[j]))
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (sum([v[0]]), set(v[1])) for k, v in pattern_counts.items()}

# 輔助函數：尋找相對於另一個數據集的純模式
def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

# 輔助函數：獲取基於模式的實例分數
def get_score_of_instance(instance, patterns):
    score = 0
    pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in patterns.items():
        if set(pattern).issubset(instance_set):
            score += data[0]
            pattern_in_.append([set(pattern), data[0]])
    return score, pattern_in_

# 輔助函數：獲取基於純模式的實例純分數
def get_pure_score_of_instance(instance, pure_patterns):
    pure_score = 0
    pure_pattern_in_ = []
    instance_set = set(instance)
    for pattern, data in pure_patterns.items():
        if set(pattern).issubset(instance_set):
            pure_score += data[0]
            pure_pattern_in_.append([set(pattern), data[0]])
    return pure_score, pure_pattern_in_

# 輔助函數：尋找滿足指定條件的C中的實例
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

# 數據預處理函數
def DataPreprocessing(uploaded_file):
    # ... (此部分程式碼與先前相同，已整理排版)
    rat1 = 0.85
    rat2 = 0.15
    rof = 2
    ski = 1 #修正跳過第一行

    # ... (其餘程式碼與先前相同)
    return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]

# 方法函數
def Method(acc, A, B, C, IdT, ClassT):
    # ... (此部分程式碼與先前相同，已整理排版)
    A_bool, B_bool = tuples_to_boolean_arrays(A, acc), tuples_to_boolean_arrays(B, acc)
    SA = calculate_scores_parallel(C, identify_pure_sets_numpy(calculate_unique_intersections_parallel(A_bool), B_bool, acc))
    SB = calculate_scores_parallel(C, identify_pure_sets_numpy(calculate_unique_intersections_parallel(B_bool), A_bool, acc))
    # ... (其餘程式碼與先前相同)

# Z分數計算函數
def Zscore(v):
    return 0.0 if v == 0.0 else v

# 風險高亮顯示函數
def highlight_risk(row):
    # ... (此部分程式碼與先前相同，已整理排版)

# 主函數
    def main():
    # ... (此部分程式碼與先前相同，已整理排版)
    with tabs[0]:
        uploaded_file = st.file_uploader("上傳您的CSV檔案", type=["csv"])

        if uploaded_file is not None:
            result = DataPreprocessing(uploaded_file)
            if result is not None:
                acc, A, B, C, IdT, ClassT = result
                Method(acc, A, B, C, IdT, ClassT)

    with tabs[1]:
        # ... (此部分程式碼與先前相同，已整理排版)
        pass #避免錯誤

    with tabs[2]:
        if 'A' in locals() and 'B' in locals() and 'C' in locals() and len(A)>0 and len(B)>0 and len(C)>0: #增加判斷，避免沒有資料時出錯
            patterns_A = find_patterns_updated(A)
            patterns_B = find_patterns_updated(B)
            pure_patterns_A = find_pure_patterns(patterns_A, B)
            pure_patterns_B = find_pure_patterns(patterns_B, A)

            specific_instances_C = find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B)
            total_specific_instances_C = len(specific_instances_C)
            st.write(f"在 C 中滿足指定條件的實例總共有 {total_specific_instances_C} 筆資料")
        else:
            st.write("請先上傳檔案並完成資料前處理。")

    with tabs[3]:
        if 'specific_instances_C' in locals() and len(specific_instances_C)>0: #增加判斷，避免沒有資料時出錯
            choices = [f"Data {i+1}" for i in range(total_specific_instances_C)]
            choice = st.selectbox("Data", [" "] + choices)

            if choice != " ":
                index = int(choice.split(" ")[1]) - 1
                st.subheader("RESULT")

                c, score_A, score_B, pure_score_A, pure_score_B = specific_instances_C[index]

                source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
                target = [1, 2] + list(range(3, 3 + len(score_A[1]))) + list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1])))
                value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]
                label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in score_A[1]] + ['N'+str(i[0]) for i in score_B[1]]
                node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])

                fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="#37474F", width=0.5), label=label, color=node_colors),
                    link=dict(source=source, target=target, value=value, color=['#F8BBD0','#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1]))
                )])

                st.plotly_chart(fig)

                st.subheader("Pure RESULT")

                pure_source = [0, 0] + [1] * len(pure_score_A[1]) + [2] * len(pure_score_B[1])
                pure_target = [1, 2] + list(range(3, 3 + len(pure_score_A[1]))) + list(range(3 + len(pure_score_A[1]), 3 + len(pure_score_A[1]) + len(pure_score_B[1])))
                pure_value = [pure_score_A[0], pure_score_B[0]] + [i[-1] for i in pure_score_A[1]] + [i[-1] for i in pure_score_B[1]]
                pure_label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in pure_score_A[1]] + ['N'+str(i[0]) for i in pure_score_B[1]]
                pure_node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(pure_score_A[1]) + ['#F1F8E9'] * len(pure_score_B[1])

                pure_fig = go.Figure(data=[go.Sankey(
                    node=dict(pad=15, thickness=20, line=dict(color="#37474F", width=0.5), label=pure_label, color=pure_node_colors),
                    link=dict(source=pure_source, target=pure_target, value=pure_value, color=['#F8BBD0','#DCEDC8'] + ['#FFEBEE'] * len(pure_score_A[1]) + ['#F1F8E9'] * len(pure_score_B[1]))
                )])

                st.plotly_chart(pure_fig)
        else:
            st.write("請先上傳檔案並完成資料前處理，且確保有符合條件的資料。")

    with tabs[4]:
        if 'specific_instances_C' in locals() and 'ClassT' in locals() and len(specific_instances_C)>0 and len(ClassT)>0: #增加判斷，避免沒有資料時出錯
            data = []
            for idx, (c, score_A, score_B, pure_score_A, pure_score_B) in enumerate(specific_instances_C):
                risk_score = max(pure_score_A[0], pure_score_B[0])
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
                data.append({
                    "Status": status,
                    "ID": idx + 1,
                    "NS": pure_score_A[0],
                    "PS": pure_score_B[0],
                    "Label": ClassT[idx],
                    "Misdiagnosis Risk": risk_level
                })

            df_risk = pd.DataFrame(data)
            styled_df = df_risk.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=False, height=600)
        else:
            st.write("請先上傳檔案並完成資料前處理，且確保有符合條件的資料。")

if __name__ == '__main__':
    main()
