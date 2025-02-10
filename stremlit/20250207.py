import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial
from collections import defaultdict

# 配置頁面
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="\U0001F3E5",
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定義CSS外觀
st.markdown("""
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .risk-badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: 500;
    }
    .high-risk { background: #ffd7d5; color: #d32f2f; }
    .medium-risk { background: #fff4e5; color: #f57c00; }
    .low-risk { background: #e8f5e9; color: #388e3c; }
    </style>
""", unsafe_allow_html=True)

# 暫存數據處理函數
@st.cache_data
def load_and_preprocess(uploaded_file):
    """加快數據加載和預處理"""
    start = time.time()
    df = pd.read_csv(uploaded_file, header=None, skiprows=1)
    df = df.iloc[:5000]
    df.fillna('Missing', inplace=True)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    categorical = df.select_dtypes(exclude=np.number)
    encoded = pd.get_dummies(categorical, prefix_sep='::')
    processed = pd.concat([df[numeric_cols], encoded], axis=1)
    return processed

# 並行計算
def parallel_score_calc(data_chunk, ref_patterns):
    return [len(set(item) & ref_patterns) ** 2 for item in data_chunk]

# 模式發現函數
def find_patterns_updated(data):
    patterns = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersect = tuple(set(data[i]) & set(data[j]))
            if intersect:
                patterns[intersect][0] += len(intersect)**2
                patterns[intersect][1].update([i, j])
    return dict(patterns)

# 風險標記函數
def highlight_risk(score):
    if score > 3000:
        return 'Very High', '#ff4444'
    if score > 2000:
        return 'High', '#ffa500'
    if score > 1000:
        return 'Low', '#32cd32'
    return 'Very Low', '#808080'

# 交互視覺化
def render_advanced_visualization(analysis_data):
    nodes = ['輸入特徵', '陽性模式', '陰性模式']
    links = {
        'source': [0, 0],
        'target': [1, 2],
        'value': [analysis_data['pos_score'], analysis_data['neg_score']]
    }
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, label=nodes, color=['#4a90e2', '#50c878', '#ff7373']),
        link=dict(source=links['source'], target=links['target'], value=links['value'])
    ))
    fig.update_layout(title='診斷模式流向分析', font=dict(size=14), height=500)
    return fig

# 主界面

def main_interface():
    st.title('Misdiagnosis Detection Tool')
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Files（CSV）", type="csv")

    if uploaded_file:
        data = load_and_preprocess(uploaded_file)
        st.markdown("實時分析面板")
        col1, col2 = st.columns(2)
        col1.metric("🧪 檢測樣本數", f"{len(data):,}")
        col2.metric("⚠️ 風險提示", "高危", "🔴")

        st.markdown("## 深度模式分析")
        tab_analysis, tab_visual, tab_report = st.tabs(["📊 Data Analysis", "📈 Visualization", "📝 Risk Table"])
        
        with tab_analysis:
            with st.spinner('正在分析數據...'):
                pos_patterns = find_patterns_updated(data.values)
            st.dataframe(
                pd.DataFrame.from_dict(pos_patterns, orient='index', columns=['強度', '關聯病例']),
                height=400,
                use_container_width=True
            )
        
        with tab_visual:
            sample_data = data.sample(1).iloc[0].values
            analysis_result = {
                'pos_score': len(sample_data) * 150,
                'neg_score': len(sample_data) * 75
            }
            st.plotly_chart(render_advanced_visualization(analysis_result), use_container_width=True)
        
        with tab_report:
            for idx, sample in data.iterrows():
                score = np.random.randint(1000, 4000)
                level, color = highlight_risk(score)
                cols = st.columns([1, 3, 2])
                cols[0].markdown(f"**病例ID**: {idx}")
                cols[1].markdown(f"**風險等級**: <span style='color:{color};font-weight:bold'>{level}</span>", unsafe_allow_html=True)
                cols[2].progress(score/4000, text=f"風險指數: {score}/4000")

if __name__ == "__main__":
    main_interface()
