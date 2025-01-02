import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def main():
    # 設定頁面
    st.set_page_config(page_title="Misdiagnosis Detection Tool", layout="wide")

    # 頂部標題與簡介
    st.title("🔍 Misdiagnosis Detection Tool")
    st.write("這是一個檢測誤診風險的工具，提供清晰的數據可視化與風險分析。")

    # 上傳資料
    st.sidebar.header("📂 上傳資料")
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 檔案", type=["csv"])

    if uploaded_file is not None:
        # 載入資料
        data = load_data(uploaded_file)

        # 資料展示
        st.subheader("📊 資料集預覽")
        st.dataframe(data.head())

        # 桑基圖
        st.subheader("📈 誤診流向圖（桑基圖）")
        create_sankey_chart(data)

        # 風險表格
        st.subheader("⚠️ 高風險誤診分析表")
        display_risk_table(data)
    else:
        st.info("請先上傳 CSV 檔案以進行分析。")

@st.cache_data
def load_data(file):
    """載入並緩存上傳的資料"""
    return pd.read_csv(file)

@st.cache_data
def calculate_risk(data):
    """計算風險等級"""
    data['Risk Level'] = data['Misdiagnosis Probability'].apply(
        lambda x: "高" if x > 0.7 else ("中" if x > 0.4 else "低")
    )
    return data

def create_sankey_chart(data):
    """建立桑基圖"""
    source = data['Initial Diagnosis']
    target = data['Final Diagnosis']
    value = data['Misdiagnosis Probability']

    labels = list(pd.concat([source, target]).unique())
    source_indices = [labels.index(src) for src in source]
    target_indices = [labels.index(tgt) for tgt in target]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=value
                )
            )
        ]
    )

    fig.update_layout(title_text="誤診流向圖", font_size=10)
    st.plotly_chart(fig, use_container_width=True)

def display_risk_table(data):
    """顯示誤診風險表格並高亮風險等級"""
    data = calculate_risk(data)

    def highlight_risk(row):
        if row['Risk Level'] == '高':
            return ['background-color: red'] * len(row)
        elif row['Risk Level'] == '中':
            return ['background-color: orange'] * len(row)
        elif row['Risk Level'] == '低':
            return ['background-color: green'] * len(row)
        else:
            return [''] * len(row)

    st.dataframe(data.style.apply(highlight_risk, axis=1))

if __name__ == "__main__":
    main()
