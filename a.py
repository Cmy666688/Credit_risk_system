import streamlit as st
st.title("我的第一个 Streamlit 应用")
st.write("Hello, Streamlit!")

st.title("Streamlit 文本演示")
st.header("标题示例")
st.write("这是普通文本")
st.markdown("### Markdown 标题")

import pandas as pd
df = pd.DataFrame({ "姓名": ["张三", "李四", "王五"], 
"年龄": [25, 30, 35], "成绩": [85, 92, 78] })
st.dataframe(df)

st.sidebar.header("图表控制")
chart_type = st.sidebar.selectbox("选择图表类型", ["柱状图", "折线图", "饼图"])
data_range = st.sidebar.slider("数据范围", 1, 100, (10, 50))


tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
with tab1: 
    st.write("选项卡 1 的内容") 
with tab2:
    st.write("选项卡 2 的内容") 
with tab3: 
    st.write("选项卡 3的内容")