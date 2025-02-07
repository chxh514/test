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

# 並行計算
def parallel_score_calc(data_chunk, ref_patterns):
    """并行評分計算"""
    return [len(set(item) & ref_patterns) ** 2 for item in data_chunk]

# 核心分析邏輯
class DiagnosisAnalyzer:
    def __init__(self, data):
        self.data = data
        self.pattern_cache = {}
        
    @st.cache_data
    def find_patterns(_self, class_type):
        """帶緩存的模式發現"""
        patterns = defaultdict(lambda: [0, set()])
        for i in range(len(_self.data)):
            for j in range(i, len(_self.data)):
                intersect = tuple(set(_self.data[i]) & set(_self.data[j]))
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
            # 初始化 patterns_A, patterns_B, pure_patterns_A, pure_patterns_B 變量
        patterns_A = find_patterns_updated(st.session_state.processed_data['A'])
        patterns_B = find_patterns_updated(st.session_state.processed_data['B'])
        pure_patterns_A = find_pure_patterns(patterns_A, st.session_state.processed_data['B'])
        pure_patterns_B = find_pure_patterns(patterns_B, st.session_state.processed_data['A'])
        
        # 查找滿足條件的 C 中的實例
        specific_instances_C = find_specific_instances(C, patterns_A, patterns_B, pure_patterns_A, pure_patterns_B)

        # 計算 specific_instances_C 的資料筆數並儲存為變數
        total_specific_instances_C = len(specific_instances_C)
        
        specific_instances = find_specific_instances(
            st.session_state.processed_data['C'],
            find_patterns_updated(st.session_state.processed_data['A']),
            find_patterns_updated(st.session_state.processed_data['B']),
            find_pure_patterns(find_patterns_updated(st.session_state.processed_data['A']), st.session_state.processed_data['B']),
            find_pure_patterns(find_patterns_updated(st.session_state.processed_data['B']), st.session_state.processed_data['A'])
        )

        total_specific_instances = len(specific_instances)
        choices = [f"Data {i+1}" for i in range(total_specific_instances)]
        choice = st.selectbox("Data", [" "] + choices)

        if choice != " ":
            index_str = choice.split(" ")[1]
            if index_str.isdigit():
                index = int(index_str) - 1  # 轉換選擇為索引
                c, score_A, score_B, pure_score_A, pure_score_B = specific_instances_C[index]
            else:
                st.error("Invalid selection. Please choose a valid option.")
            return

            st.subheader("RESULT")

            # 定義 Sankey 圖的 source, target 和 value 陣列
            source = [0, 0] + [1] * len(score_A[1]) + [2] * len(score_B[1])
            target = [1, 2] + list(range(3, 3 + len(score_A[1]))) + list(range(3 + len(score_A[1]), 3 + len(score_A[1]) + len(score_B[1])))
            value = [score_A[0], score_B[0]] + [i[-1] for i in score_A[1]] + [i[-1] for i in score_B[1]]

            # 定義節點標籤，PATIENT 標籤將顯示所選資料的 PATIENT_ID
            label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in score_A[1]] + ['N'+str(i[0]) for i in score_B[1]]

            # Define node colors
            node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])

            # Create the Sankey diagram
            fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="#37474F", width=0.5),label=label),link=dict(source=source,target=target,value=value,color=node_colors[1:2] + ['#FFEBEE'] * len(score_A[1]) + ['#F1F8E9'] * len(score_B[1])))])

            # 在 Streamlit 中顯示 Sankey 圖
            st.plotly_chart(fig)

            # 顯示取過 pure 的桑基圖
            st.subheader("Pure RESULT")
                    
            # 定義 pure Sankey 圖的 source, target 和 value 陣列
            pure_source = [0, 0] + [1] * len(pure_score_A[1]) + [2] * len(pure_score_B[1])
            pure_target = [1, 2] + list(range(3, 3 + len(pure_score_A[1]))) + list(range(3 + len(pure_score_A[1]), 3 + len(pure_score_A[1]) + len(pure_score_B[1])))
            pure_value = [pure_score_A[0], pure_score_B[0]] + [i[-1] for i in pure_score_A[1]] + [i[-1] for i in pure_score_B[1]]
                    
            # 定義 pure 節點標籤
            pure_label = [f'PATIENT:{index+1}', 'Positive P', 'Negative N'] + ['P'+str(i[0]) for i in pure_score_A[1]] + ['N'+str(i[0]) for i in pure_score_B[1]]

            # Define pure node colors
            pure_node_colors = ['#ECEFF1', '#F8BBD0', '#DCEDC8'] + ['#FFEBEE'] * len(pure_score_A[1]) + ['#F1F8E9'] * len(pure_score_B[1])

            # Create the pure Sankey diagram
            pure_fig = go.Figure(data=[go.Sankey(node=dict(pad=15,thickness=20,line=dict(color="#37474F", width=0.5),label=pure_label),link=dict(source=pure_source,target=pure_target,value=pure_value,color=pure_node_colors))])

            # 在 Streamlit 中顯示 pure Sankey 圖
            st.plotly_chart(pure_fig)

        
        with tab_report:
            risk_samples = data.sample(5)
            for idx, sample in risk_samples.iterrows():
                score = np.random.randint(1000, 4000)
                level, color = analyzer.get_risk_level(score)
                
                with st.container(border=True):
                    cols = st.columns([1,3,2])
                    cols[0].markdown(f"**病例ID**: {idx}")
                    cols[1].markdown(f"**風險等級**: <span style='color:{color};font-weight:bold'>{level}</span>", 
                                   unsafe_allow_html=True)
                    cols[2].progress(score/4000, text=f"風險指數: {score}/4000")


if __name__ == "__main__":
    main_interface()
