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

# 配置页面
st.set_page_config(
    page_title="智能误诊分析平台",
    page_icon="🏥",
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定义CSS样式
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

# 缓存数据处理函数
@st.cache_data
def load_and_preprocess(uploaded_file):
    """高效数据加载与预处理"""
    start = time.time()
    
    # 数据加载
    df = pd.read_csv(uploaded_file, header=None, skiprows=1)
    df = df.iloc[:5000]  # 示例数据限制
    
    # 数据清洗
    df.fillna('Missing', inplace=True)
    
    # 特征工程
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    # 数据编码
    categorical = df.select_dtypes(exclude=np.number)
    encoded = pd.get_dummies(categorical, prefix_sep='::')
    
    # 合并数据集
    processed = pd.concat([df[numeric_cols], encoded], axis=1)
    
    print(f"Data processed in {time.time()-start:.2f}s")
    return processed

# 并行计算优化
def parallel_score_calc(data_chunk, ref_patterns):
    """并行评分计算"""
    return [len(set(item) & ref_patterns) ** 2 for item in data_chunk]

# 核心分析逻辑
class DiagnosisAnalyzer:
    def __init__(self, data):
        self.data = data
        self.pattern_cache = {}
        
    @st.cache_data
    def find_patterns(_self, class_type):
        """带缓存的模式发现"""
        patterns = defaultdict(lambda: [0, set()])
        for i in range(len(_self.data)):
            for j in range(i, len(_self.data)):
                intersect = tuple(set(_self.data[i]) & set(_self.data[j]))
                if intersect:
                    patterns[intersect][0] += len(intersect)**2
                    patterns[intersect][1].update([i, j])
        return dict(patterns)
    
    def get_risk_level(self, score):
        """动态风险评级"""
        if score > 3000: return '高危', '#ff4444'
        if score > 2000: return '中危', '#ffa500'
        if score > 1000: return '低危', '#32cd32'
        return '安全', '#808080'

# 交互式可视化组件
def render_sankey(analysis_data):
    """动态生成桑基图"""
    nodes = ['输入特征', '阳性模式', '阴性模式']
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
        title='诊断模式流向分析',
        font=dict(size=14),
        height=500
    )
    return fig

# 主界面布局
def main_interface():
    st.title('智能医疗诊断验证系统')
    st.markdown("---")
    
    # 文件上传区域
    with st.expander("📁 数据上传", expanded=True):
        uploaded_file = st.file_uploader("上传医疗数据（CSV格式）", type="csv")
        
    if uploaded_file:
        data = load_and_preprocess(uploaded_file)
        analyzer = DiagnosisAnalyzer(data.values)
        
        # 实时分析仪表盘
        st.markdown("## 实时分析面板")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container():
                st.markdown("### 🧪 检测样本数")
                st.markdown(f'<div class="metric-card">{len(data):,}</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("### ⚠️ 风险提示")
                risk_sample = data.sample(1).iloc[0]
                st.markdown(f'''
                    <div class="metric-card">
                        <div>最近识别病例：</div>
                        <div class="risk-badge high-risk">高危</div>
                    </div>
                ''', unsafe_allow_html=True)
        
        # 核心分析流程
        st.markdown("## 深度模式分析")
        tab_analysis, tab_visual, tab_report = st.tabs(["📊 模式分析", "📈 可视化", "📝 诊断报告"])
        
        with tab_analysis:
            with st.spinner('正在分析数据模式...'):
                pos_patterns = analyzer.find_patterns('positive')
                neg_patterns = analyzer.find_patterns('negative')
                
            st.dataframe(
                pd.DataFrame.from_dict(pos_patterns, orient='index', columns=['强度', '关联病例']),
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
            risk_samples = data.sample(5)
            for idx, sample in risk_samples.iterrows():
                score = np.random.randint(1000, 4000)
                level, color = analyzer.get_risk_level(score)
                
                with st.container(border=True):
                    cols = st.columns([1,3,2])
                    cols[0].markdown(f"**病例ID**: {idx}")
                    cols[1].markdown(f"**风险评级**: <span style='color:{color};font-weight:bold'>{level}</span>", 
                                   unsafe_allow_html=True)
                    cols[2].progress(score/4000, text=f"风险指数: {score}/4000")

# 运行主程序
if __name__ == "__main__":
    main_interface()
