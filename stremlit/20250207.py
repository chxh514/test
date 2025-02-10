import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import plotly.graph_objects as go
from collections import defaultdict

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

# 暫存數據處理函數
@st.cache_data
def load_and_preprocess(uploaded_file):
    """加快數據加載和預處理"""
    start = time.time()

    # 數據加載
    df = pd.read_csv(uploaded_file, header=None, skiprows=1)
    df = df.iloc[:5000]  # 示例數據限制

    # 數據清洗
    df.fillna('Missing', inplace=True)

    # 特徵工程
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # 數據編碼
    categorical = df.select_dtypes(exclude=np.number)
    encoded = pd.get_dummies(categorical, prefix_sep='::')

    # 合併數據集
    processed = pd.concat([df[numeric_cols], encoded], axis=1)

    print(f"Data processed in {time.time()-start:.2f}s")
    return processed

# 核心分析邏輯
class DiagnosisAnalyzer:
    def __init__(self, data):
        self.data = data
        self.pattern_cache = {}

    @st.cache_data
    def find_patterns(self, class_type):
        """帶緩存的模式發現"""
        patterns = defaultdict(lambda: [0, set()])
        for i in range(len(self.data)):
            for j in range(i, len(self.data)):
                intersect = tuple(set(self.data[i]) & set(self.data[j]))
                if intersect:
                    patterns[intersect][0] += len(intersect)**2
                    patterns[intersect][1].update([i, j])
        return dict(patterns)

    def get_risk_level(self, score):
        """動態風險評估"""
        if score > 3000: return 'Very High', '#ff4444'
        if score > 2000: return 'High', '#ffa500'
        if score > 1000: return 'Low', '#32cd32'
        return 'Very Low', '#808080'

# 交互式視覺化组件
def render_sankey(analysis_data):
    """動態生成"""
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

# 繪製 ROC 曲線
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    roc_auc = metrics.auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC 曲線'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='隨機猜測'))
    fig_roc.update_layout(title=f'ROC 曲線 (AUC = {roc_auc:.2f})', xaxis_title='假陽性率', yaxis_title='真陽性率')
    return fig_roc

# 主界面布局
def main_interface():
    st.title('Misdiagnosis Detection Tool')
    st.markdown("---")

    # 文件上傳
    with st.expander("📁 Upload Files", expanded=True):
        uploaded_file = st.file_uploader("Upload Files（CSV）", type="csv")

    if uploaded_file:
        data = load_and_preprocess(uploaded_file)
        analyzer = DiagnosisAnalyzer(data.values)

        # 添加下拉式選單
        patient_ids = data.index.tolist()
        selected_patient = st.selectbox("選擇病患", patient_ids)

        # 實時分析顯示
        st.markdown("實時分析面板")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("🧪 檢測樣本數")
            st.markdown(f'<div class="metric-card">{len(data):,}</div>', unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown("⚠️ 風險提示")
        
            # 計算四個風險程度的人數
            risk_counts = {'Very High': 0, 'High': 0, 'Low': 0, 'Very Low': 0}
            for idx, sample in data.iterrows():
                score = np.random.randint(1000, 4000)  # 示例分數計算
                level, _ = analyzer.get_risk_level(score)
                risk_counts[level] += 1
        
            # 顯示風險提示
            st.markdown(f'''
                <div class="metric-card">
                    <div>風險分佈：</div>
                    <div class="risk-badge high-risk">Very High: {risk_counts['Very High']}</div>
                    <div class="risk-badge medium-risk">High: {risk_counts['High']}</div>
                    <div class="risk-badge low-risk">Low: {risk_counts['Low']}</div>
                    <div class="risk-badge very-low-risk">Very Low: {risk_counts['Very Low']}</div>
                </div>
            ''', unsafe_allow_html=True)

        # 根據選擇的病患顯示桑基圖
        sample_data = data.loc[selected_patient].values
        analysis_result = {
            'pos_score': len(sample_data) * 150,
            'neg_score': len(sample_data) * 75
        }
        st.plotly_chart(render_sankey(analysis_result), use_container_width=True)

        # 計算和顯示 ROC 曲線
        y_true = np.random.randint(0, 2, size=len(data))  # 假設的真實標籤
        y_scores = np.random.rand(len(data))  # 假設的預測分數

        fig_roc = plot_roc_curve(y_true, y_scores)
        st.plotly_chart(fig_roc, use_container_width=True)
        # 核心分析流程
        st.markdown("## 深度模式分析")
        tab_analysis, tab_visual, tab_report = st.tabs(["📊 Data Analysis", "📈 Visualization", "📝 Risk Table"])

        with tab_analysis:
            with st.spinner('正在分析數據...'):
                pos_patterns = analyzer.find_patterns('positive')
                neg_patterns = analyzer.find_patterns('negative')

            st.dataframe(
                pd.DataFrame.from_dict(pos_patterns, orient='index', columns=['强度', '關聯病例']),
                height=400,
                use_container_width=True
            )

        with tab_visual:
            sample_data = data.sample(1).iloc[0].values
            analysis_result = {
                'pos_score': len(sample_data) * 150,
                'neg_score': len(sample_data) * 75
            }
            st.plotly_chart(render_sankey(analysis_result), use_container_width=True)

        with tab_report:
            for idx, sample in data.iterrows():
                score = np.random.randint(1000, 4000)
                level, color = analyzer.get_risk_level(score)

                with st.container():
                    cols = st.columns([1, 3, 2])
                    cols[0].markdown(f"**病例ID**: {idx}")
                    cols[1].markdown(f"**風險等級**: <span style='color:{color};font-weight:bold'>{level}</span>", unsafe_allow_html=True)
                    cols[2].progress(score/4000, text=f"風險指數: {score}/4000")


if __name__ == "__main__":
    main_interface()


if __name__ == "__main__":
    main_interface()
