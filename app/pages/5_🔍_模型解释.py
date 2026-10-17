"""
🔍 模型解释页面
================
SHAP 全局/局部解释。
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import feature_importance_bar

st.set_page_config(page_title="模型解释", page_icon="🔍", layout="wide")
render_sidebar()

st.title("🔍 模型解释 (SHAP)")
st.caption("理解模型为什么这么判 - 全局重要性 + 局部解释")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))


# ===== 加载模型与数据 =====
@st.cache_resource
def load_resources():
    from src.models.xgb_model import XGBModel
    from src.explain.shap_analyzer import SHAPAnalyzer
    from src.features.builder import FeatureBuilder
    from app.components.data_utils import find_data_file

    model_path = ROOT / "data" / "models" / "latest" / "xgb_model.json"
    prep_path = ROOT / "data" / "models" / "latest" / "preprocessor.pkl"
    selector_path = ROOT / "data" / "models" / "latest" / "feature_selector.pkl"

    if not model_path.exists() or not prep_path.exists():
        return None

    # 优先用真实数据，回退到合成数据
    data_path, _ = find_data_file(ROOT, prefer="auto")
    if data_path is None:
        return None

    model = XGBModel(configs.get("model").xgboost)
    model.load(str(model_path))

    prep = joblib.load(prep_path)
    selector = joblib.load(selector_path) if selector_path.exists() else None

    # 全量太慢，1500 行做解释足够代表性
    df = pd.read_csv(data_path, nrows=1500)
    target = configs.get("data").target_column
    id_col = configs.get("data").id_column

    X = df.drop(columns=[target, id_col], errors="ignore")
    X_processed = prep.transform(X)

    builder = FeatureBuilder(configs.get("feature"))
    X_processed = builder.build(X_processed)

    if selector is not None:
        X_processed = selector.transform(X_processed)

    # 构建 SHAP（用前 1000 条作为背景）
    analyzer = SHAPAnalyzer(max_background_size=500)
    analyzer.fit(model.model, X_processed.head(500))

    return {
        "model": model,
        "X": X_processed,
        "y": df[target] if target in df.columns else None,
        "analyzer": analyzer,
    }


resources = load_resources()
if resources is None:
    st.warning("⚠️ 模型或数据未就绪，请先完成训练")
    st.stop()

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["🌍 全局解释", "🎯 单样本解释", "📈 模型自带重要性"])

with tab1:
    st.markdown("## 全局特征重要性")
    st.markdown("""
    SHAP 全局重要性 = 所有样本 SHAP 值绝对值的均值，
    比 XGBoost 自带的 `feature_importance_` 更准确。
    """)

    n_samples = st.slider("分析样本数", 50, 1000, 200, step=50, help="样本越多越准，但耗时增加")

    if st.button("🔬 计算全局 SHAP 值", type="primary"):
        with st.spinner(f"计算 {n_samples} 个样本的 SHAP 值..."):
            X_sample = resources["X"].sample(min(n_samples, len(resources["X"])), random_state=42)
            importance = resources["analyzer"].global_importance(X_sample, top_n=20)

        st.success("✅ 计算完成")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = feature_importance_bar(importance, top_n=20, title="SHAP 全局特征重要性")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                pd.DataFrame({
                    "特征": importance.index,
                    "SHAP 重要性": importance.values.round(4),
                }),
                use_container_width=True,
                hide_index=True,
            )

with tab2:
    st.markdown("## 单样本 SHAP 解释")

    sample_idx = st.number_input(
        "选择样本索引",
        0,
        len(resources["X"]) - 1,
        0,
        help="从测试集中选择一个样本进行解释"
    )

    if st.button("🔍 解释此样本", type="primary"):
        sample = resources["X"].iloc[sample_idx]

        with st.spinner("计算 SHAP 值..."):
            explanation = resources["analyzer"].explain_single(sample)

        # 预测信息
        proba = resources["model"].predict_proba(sample.to_frame().T)[0]
        true_label = resources["y"].iloc[sample_idx] if resources["y"] is not None else None

        col1, col2, col3 = st.columns(3)
        col1.metric("预测违约概率", f"{proba:.2%}")
        col2.metric("真实标签", f"{int(true_label)}" if true_label is not None else "未知")
        col3.metric("Top1 影响特征", explanation.get_top_features(1)[0]["feature"])

        # 瀑布图
        from app.components.chart_utils import shap_waterfall
        fig = shap_waterfall(
            explanation.feature_names,
            explanation.feature_values,
            explanation.shap_values,
            explanation.base_value,
            top_n=10,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top 特征表
        st.markdown("### Top 10 特征贡献")
        st.dataframe(
            pd.DataFrame(explanation.get_top_features(top_n=10)),
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.markdown("## 模型自带特征重要性")
    st.markdown("XGBoost 自带的 gain 重要性")

    importance_type = st.selectbox(
        "重要性类型",
        ["gain", "weight", "cover", "total_gain", "total_cover"],
        help="gain: 平均增益（推荐）；weight: 使用次数；cover: 覆盖样本数",
    )

    importance = resources["model"].get_feature_importance(importance_type=importance_type)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = feature_importance_bar(importance, top_n=20, title=f"XGBoost {importance_type} 重要性")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(
            pd.DataFrame({
                "特征": importance.index[:20],
                "重要性": importance.values[:20].round(4),
            }),
            use_container_width=True,
            hide_index=True,
        )
