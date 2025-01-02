import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from collections import Counter, defaultdict

# 定義 Z-score 計算和數據預處理的輔助函數

def Zscore(v):
    """計算 Z-score，處理零值的特殊情況"""
    if v == 0.0:
        return 0.0
    else:
        return v

def DataPreprocessing(uploaded_file):
    # ... (此部分程式碼與之前相同，故省略以節省篇幅)
    # ... (請參考之前的程式碼)
    return acc, [i[1] for i in A], [i[1] for i in B], [i[1] for i in C], [i[0] for i in C], [i[2] for i in C]

def find_patterns_updated(data):
    pattern_counts = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersection = tuple(sorted(set(data[i]) & set(data[j]))) # 加入排序
            if intersection:
                pattern_counts[intersection][0] += len(intersection)**2
                pattern_counts[intersection][1].update([i, j])
    return {k: (sum([v[0]]), sorted(list(v[1]))) for k, v in pattern_counts.items()} # 加入排序

def find_pure_patterns(patterns, other_data):
    pure_patterns = {}
    other_sets = [set(item) for item in other_data]
    for pattern, data in patterns.items():
        pattern_set = set(pattern)
        if not any(pattern_set.issubset(other_set) for other_set in other_sets):
            pure_patterns[pattern] = data
    return pure_patterns

def get_score_of_instance(instance, patterns):
    score = 0
    matched_patterns = []
    for pattern, pattern_data in patterns.items():
        if set(pattern).issubset(set(instance)):
            score += pattern_data[0]
            matched_patterns.append((pattern, pattern_data[0]))
    return score, matched_patterns

def create_sankey_data(index, instance_data, score_A_data, score_B_data):
    score_A, matched_patterns_A = score_A_data
    score_B, matched_patterns_B = score_B_data
    source = [0, 0] + [1] * len(matched_patterns_A) + [2] * len(matched_patterns_B)
    target = [1, 2] + list(range(3, 3 + len(matched_patterns_A))) + list(range(3 + len(matched_patterns_A), 3 + len(matched_patterns_A) + len(matched_patterns_B)))
    value = [score_A, score_B] + [len(i[0])**2 for i in matched_patterns_A] + [len(i[0])**2 for i in matched_patterns_B] # 修改value計算方式
    label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + [f'P{i[0]}' for i in matched_patterns_A] + [f'N{i[0]}' for i in matched_patterns_B]
    return {'source': source, 'target': target, 'values': value, 'labels': label}

def create_sankey_diagram(data, title):
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = data['labels'],
          color = data['node_colors'] if 'node_colors' in data else "blue"
        ),
        link = dict(
          source = data['source'],
          target = data['target'],
          value = data['values'],
          color = data['link_colors'] if 'link_colors' in data else "rgba(50, 100, 200, 0.3)"
      ))])
    fig.update_layout(title_text=title, font_size=10)
    return fig

def find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B):
    specific_instances = []
    for idx, instance in enumerate(C):
        score_A = get_score_of_instance(instance, patterns_A)
        score_B = get_score_of_instance(instance, patterns_B)
        pure_score_A = get_score_of_instance(instance, pure_patterns_A)
        pure_score_B = get_score_of_instance(instance, pure_patterns_B)
        specific_instances.append((idx, instance, (score_A), (score_B), (pure_score_A), (pure_score_B)))
    return specific_instances


# Streamlit 應用程式主程式
def main():
    # ... (Streamlit 介面程式碼與之前大致相同)
    # ... (請參考之前的程式碼)
    # ... 這裡會用到上面定義的函數

if __name__ == "__main__":
    main()
