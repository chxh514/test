import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial

# 页面配置
st.set_page_config(
    page_title="Misdiagnosis Detection Tool",
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

@st.cache_data
def load_and_preprocess(uploaded_file):
    """加载和预处理数据"""
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

def find_patterns_updated(data):
    """改进后的模式发现算法"""
    patterns = defaultdict(lambda: [0, set()])
    for i in range(len(data)):
        for j in range(i, len(data)):
            intersect = tuple(set(data[i]) & set(data[j]))
            if intersect:
                patterns[intersect][0] += len(intersect)**2
                patterns[intersect][1].update([i, j])
    return dict(patterns)

def find_pure_patterns(patterns, opposite_data):
    """寻找纯净模式"""
    pure_patterns = {}
    for pattern, (score, cases) in patterns.items():
        pure = True
        for case in cases:
            if pattern in opposite_data[case]:
                pure = False
                break
        if pure:
            pure_patterns[pattern] = (score, cases)
    return pure_patterns

def find_specific_instances(C, patterns_A, patterns_B, pure_A, pure_B):
    """识别特定实例"""
    results = []
    for idx, c in enumerate(C):
        score_A = max((len(set(c)&set(p))**2 for p in patterns_A), default=0)
        score_B = max((len(set(c)&set(p))**2 for p in patterns_B), default=0)
        pure_score_A = max((len(set(c)&set(p))**2 for p in pure_A), default=0)
        pure_score_B = max((len(set(c)&set(p))**2 for p in pure_B), default=0)
        results.append((c, score_A, score_B, pure_score_A, pure_score_B))
    return sorted(results, key=lambda x: x[3]+x[4], reverse=True)

def parallel_score_calc(data_chunk, ref_patterns):
    """并行评分计算"""
    return [len(set(item) & ref_patterns) ** 2 for item in data_chunk]

class DiagnosisAnalyzer:
    def __init__(self, data):
        self.data = data
        self.patterns_cache = {}
        self.pure_patterns_cache = {}

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
        """动态风险评估"""
        if score > 3000: return 'Very High', '#ff4444'
        if score > 2000: return 'High', '#ffa500'
        if score > 1000: return 'Low', '#32cd32'
        return 'Very Low', '#808080'

    def get_advanced_patterns(self):
        """获取高级模式分析结果"""
        if not self.patterns_cache:
            self.patterns_cache['A'] = find_patterns_updated(self.data[:len(self.data)//2])
            self.patterns_cache['B'] = find_patterns_updated(self.data[len(self.data)//2:])
            
            self.pure_patterns_cache['A'] = find_pure_patterns(
                self.patterns_cache['A'], 
                self.data[len(self.data)//2:]
            )
            self.pure_patterns_cache['B'] = find_pure_patterns(
                self.patterns_cache['B'],
                self.data[:len(self.data)//2]
            )
        return self.patterns_cache, self.pure_patterns_cache

def render_sankey(analysis_data):
    """基础Sankey图生成"""
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

def render_advanced_visualization(analysis_data, selected_index):
    """高级Sankey图生成"""
    c, score_A, score_B, pure_score_A, pure_score_B = analysis_data[selected_index]
    
    # 主Sankey图
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=[f'PATIENT:{selected_index+1}', 'Positive P', 'Negative N'] + 
                  [f'P{i}' for i in range(len(score_A[1]))] +
                  [f'N{i}' for i in range(len(score_B[1]))],
            color=['#ECEFF1', '#F8BBD0', '#DCEDC8'] + 
                  ['#FFEBEE']*len(score_A[1]) + 
                  ['#F1F8E9']*len(score_B[1])
        ),
        link=dict(
            source=[0,0] + [1]*len(score_A[1]) + [2]*len(score_B[1]),
            target=[1,2] + list(range(3, 3+len(score_A[1]))) + 
                   list(range(3+len(score_A[1]), 3+len(score_A[1])+len(score_B[1]))),
            value=[score_A[0], score_B[0]] + [v[-1] for v in score_A[1]] + [v[-1] for v in score_B[1]]
        )
    ))
    
    # Pure Sankey图
    pure_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=[f'PATIENT:{selected_index+1}', 'Pure P', 'Pure N'] + 
                  [f'PP{i}' for i in range(len(pure_score_A[1]))] +
                  [f'PN{i}' for i in range(len(pure_score_B[1]))],
            color=['#ECEFF1', '#F8BBD0', '#DCEDC8'] + 
                  ['#FFEBEE']*len(pure_score_A[1]) + 
                  ['#F1F8E9']*len(pure_score_B[1])
        ),
        link=dict(
            source=[0,0] + [1]*len(pure_score_A[1]) + [2]*len(pure_score_B[1]),
            target=[1,2] + list(range(3, 3+len(pure_score_A[1]))) + 
                   list(range(3+len(pure_score_A[1]), 3+len(pure_score_A[1])+len(pure_score_B[1]))),
            value=[pure_score_A[0], pure_score_B[0]] + [v[-1] for v in pure_score_A[1]] + [v[-1] for v in pure_score_B[1]]
        )
    ))
    
    return fig, pure_fig

def highlight_risk(row):
    """风险高亮逻辑"""
    risk_level = row['综合风险']
    color_map = {
        'Very High': '#ff4444',
        'High': '#ffa500',
        'Low': '#32cd32',
        'Very Low': '#808080'
    }
    return [f'background-color: {color_map[risk_level]}' for _ in row]

def main_interface():
    st.title('Misdiagnosis Detection Tool')
    st.markdown("---")

    with st.expander("📁 Upload Files", expanded=True):
        uploaded_file = st.file_uploader("Upload Files（CSV）", type="csv")

    if uploaded_file:
        data = load_and_preprocess(uploaded_file)
        analyzer = DiagnosisAnalyzer(data.values)
        
        # 存储处理后的数据到session state
        st.session_state.processed_data = {
            'A': data[:len(data)//2].values,
            'B': data[len(data)//2:].values,
            'C': data.sample(frac=0.2).values
        }

        # 实时分析面板
        st.markdown("实时分析面板")
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container():
                st.markdown("🧪 检测样本数")
                st.markdown(f'<div class="metric-card">{len(data):,}</div>', unsafe_allow_html=True)

        with col2:
            with st.container():
                st.markdown("⚠️ 风险提示")
                risk_sample = data.sample(1).iloc[0]
                st.markdown(f'''
                    <div class="metric-card">
                        <div>最近识别病例：</div>
                        <div class="risk-badge high-risk">高危</div>
                    </div>
                ''', unsafe_allow_html=True)

        # 扩展标签页
        tab_analysis, tab_visual, tab_adv_visual, tab_report = st.tabs([
            "📊 基础分析", "📈 模式流向", "🔍 高级可视化", "📝 风险报告"
        ])

        with tab_analysis:
            with st.spinner('正在分析数据...'):
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

        with tab_adv_visual:
            if 'processed_data' in st.session_state:
                patterns_A, patterns_B = analyzer.get_advanced_patterns()[0]
                pure_A, pure_B = analyzer.get_advanced_patterns()[1]
                
                specific_instances = find_specific_instances(
                    st.session_state.processed_data['C'],
                    patterns_A,
                    patterns_B,
                    pure_A,
                    pure_B
                )
                
                total_instances = len(specific_instances)
                choices = [f"病例 {i+1}" for i in range(total_instances)]
                selected = st.selectbox("选择分析病例", options=choices)
                
                if selected:
                    index = int(selected.split()[-1]) - 1
                    fig, pure_fig = render_advanced_visualization(specific_instances, index)
                    
                    st.subheader("诊断模式流向")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("纯净模式分析")
                    st.plotly_chart(pure_fig, use_container_width=True)

        with tab_report:
            if 'processed_data' in st.session_state:
                risk_data = []
                for idx, (c, _, _, pure_A, pure_B) in enumerate(specific_instances):
                    risk_score = max(pure_A, pure_B)
                    risk_level, _ = analyzer.get_risk_level(risk_score)
                    
                    risk_data.append({
                        "ID": idx+1,
                        "阳性分数": pure_A,
                        "阴性分数": pure_B,
                        "综合风险": risk_level,
                        "紧急程度": "⚠️" if risk_score > 2000 else ""
                    })
                
                df_risk = pd.DataFrame(risk_data)
                st.dataframe(
                    df_risk.style.apply(highlight_risk, subset=['综合风险']),
                    height=600,
                    use_container_width=True
                )

                # 添加风险统计摘要
                st.markdown("### 风险统计摘要")
                risk_summary = pd.DataFrame(df_risk['综合风险'].value_counts()).reset_index()
                risk_summary.columns = ['风险等级', '病例数量']
                
                # 使用plotly创建风险分布图
                fig = go.Figure(data=[
                    go.Bar(
                        x=risk_summary['风险等级'],
                        y=risk_summary['病例数量'],
                        marker_color=['#ff4444', '#ffa500', '#32cd32', '#808080']
                    )
                ])
                
                fig.update_layout(
                    title='风险等级分布',
                    xaxis_title='风险等级',
                    yaxis_title='病例数量',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # 添加紧急处理建议
                st.markdown("### 紧急处理建议")
                urgent_cases = df_risk[df_risk['紧急程度'] == "⚠️"]
                if not urgent_cases.empty:
                    st.warning(f"发现 {len(urgent_cases)} 个需要紧急处理的病例")
                    st.dataframe(urgent_cases, use_container_width=True)
                else:
                    st.success("当前没有需要紧急处理的病例")

def run_analysis():
    """运行主分析流程"""
    try:
        main_interface()
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        st.warning("请检查输入数据格式是否正确，或者刷新页面重试。")

if __name__ == "__main__":
    run_analysis()
