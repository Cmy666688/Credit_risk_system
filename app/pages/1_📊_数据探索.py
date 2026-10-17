"""
📊 数据探索页面
================
EDA：数据概览、缺失值分析、目标分布、特征分布、相关性。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import distribution_plot, correlation_heatmap

st.set_page_config(page_title="数据探索", page_icon="📊", layout="wide")
render_sidebar()

st.title("📊 数据探索")
st.caption("理解你的数据是建模的第一步")


# ===== 数据加载 =====
from app.components.data_utils import find_data_file


@st.cache_data
def load_data(path_str: str, nrows: int | None = None) -> pd.DataFrame:
    """加载数据。path_str 必须是字符串以便 cache_data 哈希。"""
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


# 探测可用的数据源
real_path = ROOT / "data" / "raw" / "train.csv"
sample_path = ROOT / "data" / "sample" / "sample.csv"
real_available = real_path.exists()
sample_available = sample_path.exists()

if not real_available and not sample_available:
    st.error("❌ 未找到任何数据")
    st.info(
        "**方式 1**：从天池下载真实数据，放到 `data/raw/train.csv`\n\n"
        "**方式 2**：运行 `python scripts/generate_sample_data.py --n_samples 50000` 生成合成数据"
    )
    st.stop()

# ===== 数据源选择 =====
options = []
if real_available:
    options.append(f"📦 真实数据 (train.csv)")
if sample_available:
    options.append(f"🧪 合成数据 (sample.csv)")

col1, col2 = st.columns([3, 1])
with col1:
    data_source = st.radio(
        "选择数据源",
        options,
        index=0,  # 默认第一个：真实数据优先
        horizontal=True,
    )
with col2:
    if st.button("🔄 重新加载", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# 真实数据 80 万行太大，可视化时只读前 N 行
if "真实" in data_source:
    target_path = real_path
    nrows = st.slider(
        "读取行数（真实数据较大，建议先用部分数据探索）",
        min_value=10_000, max_value=800_000, value=100_000, step=10_000,
    )
else:
    target_path = sample_path
    nrows = None

df = load_data(str(target_path), nrows=nrows)

if df.empty:
    st.error("⚠️ 数据加载失败")
    st.stop()

st.success(f"✅ 已加载 {len(df):,} 条数据，{df.shape[1]} 个字段")

# ===== Tab 1: 数据概览 =====
tab1, tab2, tab3, tab4 = st.tabs(["📋 数据概览", "🕳️ 缺失值分析", "🎯 目标分布", "📈 特征分析"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总样本数", f"{len(df):,}")
    col2.metric("特征数", df.shape[1])
    col3.metric("内存占用", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    if "isDefault" in df.columns:
        col4.metric("违约率", f"{df['isDefault'].mean():.2%}")

    st.markdown("### 数据样本（前 10 行）")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 字段类型分布")
    dtype_counts = df.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["类型", "数量"]
    st.dataframe(dtype_counts, use_container_width=True, hide_index=True)

    with st.expander("📊 数值特征统计"):
        st.dataframe(df.describe().T, use_container_width=True)

with tab2:
    st.markdown("### 缺失值统计")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        st.success("🎉 数据无缺失！")
    else:
        missing_df = pd.DataFrame({
            "字段": missing.index,
            "缺失数": missing.values,
            "缺失率": (missing.values / len(df) * 100).round(2),
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # 可视化
        import plotly.express as px
        fig = px.bar(
            missing_df,
            x="字段",
            y="缺失率",
            title="各字段缺失率（%）",
            color="缺失率",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "isDefault" not in df.columns:
        st.warning("数据中没有 isDefault 字段，无法分析目标分布")
    else:
        col1, col2 = st.columns(2)
        with col1:
            import plotly.express as px
            target_dist = df["isDefault"].value_counts().reset_index()
            target_dist.columns = ["isDefault", "数量"]
            target_dist["isDefault"] = target_dist["isDefault"].map({0: "正常", 1: "违约"})

            fig = px.pie(
                target_dist,
                values="数量",
                names="isDefault",
                title="目标变量分布",
                color_discrete_sequence=["#28a745", "#dc3545"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 类别不平衡分析")
            n_pos = (df["isDefault"] == 1).sum()
            n_neg = (df["isDefault"] == 0).sum()

            st.metric("正常样本数", f"{n_neg:,}")
            st.metric("违约样本数", f"{n_pos:,}")
            st.metric("不平衡比例", f"1 : {n_neg/n_pos:.2f}")

            if n_neg / n_pos > 3:
                st.warning(
                    "⚠️ 数据存在类别不平衡，建议在模型训练时启用：\n"
                    "- XGBoost: `scale_pos_weight`\n"
                    "- MLP: 加权损失函数 / Focal Loss"
                )

with tab4:
    st.markdown("### 单特征分布")

    col1, col2 = st.columns([1, 3])
    with col1:
        # 仅展示数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "isDefault" in numeric_cols:
            numeric_cols.remove("isDefault")
        if "id" in numeric_cols:
            numeric_cols.remove("id")

        selected_col = st.selectbox("选择特征", numeric_cols)

    with col2:
        if selected_col:
            fig = distribution_plot(df, selected_col, target_col="isDefault" if "isDefault" in df.columns else None)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 特征相关性")
    if st.button("📊 生成相关性矩阵", help="对前 15 个数值特征计算相关性"):
        fig = correlation_heatmap(df, top_n=15)
        st.plotly_chart(fig, use_container_width=True)
