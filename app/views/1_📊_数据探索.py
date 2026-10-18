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

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin
require_admin()

st.title("📊 数据探索")
st.caption("理解你的数据是建模的第一步")


# ===== 数据加载 =====
from app.components.data_utils import (
    find_lending_club_file,
    load_lending_club_data,
)


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
lc_path = find_lending_club_file(ROOT)
real_available = real_path.exists()
sample_available = sample_path.exists()
lc_available = lc_path is not None

if not real_available and not sample_available and not lc_available:
    st.error("❌ 未找到任何数据")
    st.info(
        "**方式 1**：从天池下载真实数据，放到 `data/raw/train.csv`\n\n"
        "**方式 2**：运行 `python scripts/generate_sample_data.py --n_samples 50000` 生成合成数据\n\n"
        "**方式 3**：运行 `python scripts/generate_lending_club_sample.py` 生成 Lending Club 示例数据"
    )
    st.stop()

# ===== 数据源选择 =====
options = []
if real_available:
    options.append("📦 真实数据 (train.csv)")
if sample_available:
    options.append("🧪 合成数据 (sample.csv)")
if lc_available:
    options.append("🌐 Lending Club (Adapter)")

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

raw_lc_df = None
if "Lending Club" in data_source:
    # 读取 LC 原始数据，并套用 Adapter 归一化成天池 schema
    raw_lc_df, df = load_lending_club_data(str(ROOT))
    if df.empty:
        st.error("⚠️ Lending Club 数据加载失败")
        st.info("请运行 `python scripts/generate_lending_club_sample.py` 生成示例数据")
        st.stop()
else:
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

# ===== Lending Club：Adapter 归一化对照（扩展点 3）=====
if raw_lc_df is not None:
    from src.data.adapters import LendingClubAdapter

    adapter = LendingClubAdapter()
    st.success(
        f"✅ 已加载 Lending Club 数据并套用 **LendingClubAdapter**："
        f"{len(df):,} 条 × {df.shape[1]} 个字段"
    )

    st.markdown("#### 🔁 数据源适配：Lending Club schema → 天池 schema")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**① 归一化前（Lending Club 原始字段）**")
        st.caption(f"共 {raw_lc_df.shape[1]} 列")
        st.dataframe(raw_lc_df.head(3), use_container_width=True)
    with c2:
        st.markdown("**② 归一化后（天池字段）**")
        st.caption(f"共 {df.shape[1]} 列")
        st.dataframe(df.head(3), use_container_width=True)

    with st.expander("📋 字段映射表（Lending Club → 天池）", expanded=True):
        mapping = adapter.FIELD_MAPPING
        map_df = pd.DataFrame({
            "Lending Club 字段": list(mapping.keys()),
            "→ 天池字段": list(mapping.values()),
        })
        st.dataframe(map_df, use_container_width=True, hide_index=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("重命名字段", len(adapter.FIELD_MAPPING))
    col_b.metric("派生标签列", "isDefault")
    col_c.metric("补齐列", f"id + n0~n14（{len(adapter.ANONYMOUS_COLUMNS) + 1} 个）")

    with st.expander("🧹 单位清洗 & 标签派生"):
        st.markdown(
            "**单位清洗（兼容带 % 的字符串 / 已是数值两种格式）**\n"
            "- `int_rate`：`\"13.56%\"` → `13.56`\n"
            "- `revol_util`：`\"30.5%\"` → `30.5`\n"
            "- `term`：`\" 36 months\"` → `36`"
        )
        st.markdown(
            "**标签派生（`loan_status` → `isDefault`）**："
            "只有**结局明确**的样本参与建模，在贷/宽限期等未定样本整行剔除"
        )
        status_map = pd.DataFrame({
            "loan_status 取值": [
                "Fully Paid 及同类",
                "Charged Off / Default 及同类",
                "Current",
                "In Grace Period",
                "Late (16-30 days) / Late (31-120 days)",
            ],
            "处理": ["保留", "保留", "剔除", "剔除", "剔除"],
            "isDefault": ["0（正常）", "1（违约）", "—", "—", "—"],
            "说明": [
                "已结清，明确为正常",
                "已坏账，明确为违约",
                "仍在还款、结局未定",
                "宽限期内，结局未定",
                "逾期但未坏账，结局未定",
            ],
        })
        st.dataframe(status_map, use_container_width=True, hide_index=True)

    st.divider()

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
                    f"⚠️ 数据存在类别不平衡（正常 : 违约 ≈ 1 : {n_neg / n_pos:.2f}）。\n\n"
                    "训练时会**自动**按该比例处理，无需手动配置：\n"
                    "- XGBoost / LightGBM：自动计算 `scale_pos_weight`\n"
                    "- MLP：加权损失函数（`class_weight: balanced`）/ Focal Loss\n\n"
                    "如需固定权重，可在 `configs/model_config.yaml` 中显式填写数值。"
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
