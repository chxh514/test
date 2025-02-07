import streamlit as st
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import plotly.graph_objects as go

# 配置頁面
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
    page_icon="🏥",
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

# 數據處理函數
@st.cache_data
def load_and_preprocess(uploaded_file):
    start = time.time()
    df = pd.read_csv(uploaded_file, header=None, skiprows=1)
    df = df.iloc[:5000]  # 示例數據限制
    df.fillna('Missing', inplace=True)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    categorical = df.select_dtypes(exclude=np.number)
    encoded = pd.get_dummies(categorical, prefix_sep='::')
    processed = pd.concat([df[numeric_cols], encoded], axis=1)
    print(f"Data processed in {time.time()-start:.2f}s")
    return processed

# 核心分析邏輯
class DiagnosisAnalyzer:
    def __init__(self, data):
        self.data = data

    @st.cache_data
    def find_patterns(self, _self, class_type):  # 添加下劃線
        patterns = defaultdict(lambda: [0, set()])
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                intersect = tuple(set(self.data[i]) & set(self.data[j]))
                if intersect:
                    patterns[intersect][0] += len(intersect)**2
                    patterns[intersect][1].update([i, j])
        return dict(patterns)

    def get_risk_level(self, score):
        if score > 3000: return 'Very High', '#ff4444'
        if score > 2000: return 'High', '#ffa500'
        if score > 1000: return 'Low', '#32cd32'
        return 'Very Low', '#808080'

# 查找特定實例的函數
def find_specific_instances(data_col, patterns_A, patterns_B):
    return [(i, patterns_A, patterns_B) for i in range(len(data_col)) if data_col[i] == 'SomeCondition']

# 交互式視覺化组件
def render_sankey(analysis_data):
    nodes = ['輸入特徵', '陽性模式', '陰性模式']
    links = {
        'source': [0, 0],
        'target': [1, 2],
        'value': [analysis_data['pos_score'], analysis_data['neg_score']]
    }
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=nodes,
            color=['#4a90e2', '#50c878', '#ff7373']
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value']
        )
    ))
    fig.update_layout(
        title='診斷模式流向分析',
        font=dict(size=14),
        height=500
    )
    return fig

# 主界面布局
def main_interface():
    st.title('Misdiagnosis Detection Tool')
    st.markdown("---")

    # 文件上傳
    with st.expander("📁 Upload Files", expanded=True):
        uploaded_file = st.file_uploader("Upload Files（CSV）", type="csv")

    if uploaded_file:
        data = load_and_preprocess(uploaded_file)
        st.session_state.processed_data = data  # 儲存處理後的數據
        analyzer = DiagnosisAnalyzer(data.values)

        # 實時分析
        st.markdown("實時分析面板")
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container():
                st.markdown("🧪 檢測樣本數")
                st.markdown(f'<div class="metric-card">{len(data):,}</div>', unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown("⚠️ 風險提示")
                risk_sample = data.sample(1).iloc[0]
                st.markdown(f'''
                    <div class="metric-card">
                        <div>最近識別病例：</div>
                        <div class="risk-badge high-risk">高危</div>
                    </div>
                ''', unsafe_allow_html=True)

        # 核心分析流程
        st.markdown("## 深度模式分析")
        tabs = st.tabs(["📊 Data Analysis", "📈 Visualization", "🔍 Misdiagnosis Detection", "📊 Misdiagnosis Risk Table"])

        # Data Analysis Tab
        with tabs[0]:
            with st.spinner('正在分析數據...'):
                pos_patterns = analyzer.find_patterns(analyzer, 'positive')  # 傳遞實例
                neg_patterns = analyzer.find_patterns(analyzer, 'negative')

            st.dataframe(
                pd.DataFrame.from_dict(pos_patterns, orient='index', columns=['強度', '關聯病例']),
                height=400,
                use_container_width=True
            )

        # Visualization Tab
        with tabs[1]:
            sample_data = data.sample(1).iloc[0].values
            analysis_result = {
                'pos_score': len(sample_data) * 150,
                'neg_score': len(sample_data) * 75
            }
            st.plotly_chart(render_sankey(analysis_result), use_container_width=True)

     
if __name__ == "__main__":
    main_interface()
