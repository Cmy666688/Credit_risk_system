"""
🔧 特征工程页面
================
展示预处理流程、衍生特征构造、LLM 特征建议。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="特征工程", page_icon="🔧", layout="wide")
render_sidebar()

st.title("🔧 特征工程")
st.caption("特征工程决定模型上限")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))


# ===== 数据加载（统一回退逻辑：真实数据 > 合成数据） =====
from app.components.data_utils import render_data_status_or_stop

df = render_data_status_or_stop(ROOT, prefer="auto")

# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ 预处理流程",
    "🔬 衍生特征",
    "✂️ 特征筛选",
    "🤖 LLM 特征建议",
])

with tab1:
    st.markdown("## 数据预处理流程")
    st.markdown("以下步骤定义在 `configs/feature_config.yaml` 中：")

    feature_cfg = configs.get("feature")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 缺失值处理")
        st.code(f"""
数值列策略: {feature_cfg.missing_value.numerical_strategy}
类别列策略: {feature_cfg.missing_value.categorical_strategy}
丢弃阈值: {feature_cfg.missing_value.drop_threshold}
        """, language="yaml")

        st.markdown("### 异常值处理")
        outlier = feature_cfg.outlier
        st.code(f"""
方法: {outlier.method}
下分位: {outlier.lower_quantile}
上分位: {outlier.upper_quantile}
        """, language="yaml")

    with col2:
        st.markdown("### 类别编码")
        st.code(f"""
默认方法: {feature_cfg.encoding.method}
自定义映射: {len(feature_cfg.encoding.get('custom', {}))} 个字段
        """, language="yaml")

        st.markdown("### 数值标准化")
        st.code(f"""
方法: {feature_cfg.scaling.method}
        """, language="yaml")

    # 演示预处理效果
    st.divider()
    st.markdown("### 🔥 实时演示：对前 1000 条数据应用预处理")

    if st.button("🚀 执行预处理", type="primary"):
        from src.data.preprocessor import Preprocessor

        with st.spinner("处理中..."):
            sample_df = df.head(1000).copy()
            target = configs.get("data").target_column
            id_col = configs.get("data").id_column

            X = sample_df.drop(columns=[target, id_col], errors="ignore")

            prep = Preprocessor(configs.get("data"), feature_cfg)
            X_processed = prep.fit_transform(X)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**原始数据**")
            st.dataframe(X.head(), use_container_width=True)
            st.caption(f"shape: {X.shape}, 缺失值: {X.isnull().sum().sum()}")

        with col2:
            st.markdown("**处理后数据**")
            st.dataframe(X_processed.head(), use_container_width=True)
            st.caption(
                f"shape: {X_processed.shape}, 缺失值: {X_processed.isnull().sum().sum()}"
            )

        st.success("✅ 预处理完成")

with tab2:
    st.markdown("## 衍生特征构造")
    st.markdown("基于业务理解构造新特征，常常能显著提升模型表现。")

    feature_cfg = configs.get("feature")
    rules = feature_cfg.derived_features.rules

    st.markdown(f"### 当前配置了 **{len(rules)}** 条衍生规则：")

    for i, rule in enumerate(rules):
        with st.expander(f"📐 规则 {i+1}：{rule['name']}"):
            st.markdown(f"**业务含义**: {rule['description']}")
            st.code(rule["formula"], language="python")

    st.divider()

    if st.button("🔬 应用所有衍生规则", type="primary"):
        from src.features.builder import FeatureBuilder

        with st.spinner("构造衍生特征..."):
            sample_df = df.head(1000).copy()
            builder = FeatureBuilder(feature_cfg)
            df_new = builder.build(sample_df)

        new_features = builder.get_created_features()
        st.success(f"✅ 成功创建 {len(new_features)} 个衍生特征：{new_features}")

        # 展示新特征的统计信息
        if new_features:
            st.markdown("### 衍生特征统计")
            st.dataframe(df_new[new_features].describe().T, use_container_width=True)

with tab3:
    st.markdown("## 特征筛选")
    st.markdown("通过多种方法移除冗余、低价值特征。")

    feature_cfg = configs.get("feature")
    selection_cfg = feature_cfg.feature_selection

    st.markdown(f"**启用状态**: {'✅ 已启用' if selection_cfg.enabled else '❌ 已禁用'}")

    if selection_cfg.enabled:
        st.markdown("### 筛选方法")
        for method in selection_cfg.methods:
            st.markdown(f"#### 📌 {method['name']}")
            st.json(dict(method))

with tab4:
    st.markdown("## 🤖 LLM 智能特征建议")
    st.markdown("让 DeepSeek 基于数据字段和样本，提出特征工程建议。")

    if st.button("💡 生成特征建议", type="primary", help="约 10-30 秒"):
        try:
            from src.llm.client import DeepSeekClient
            from src.llm.prompts import PromptManager
            from src.llm.agents.feature_agent import FeatureAgent

            llm_cfg = configs.get("llm")

            with st.spinner("LLM 思考中..."):
                client = DeepSeekClient(llm_cfg)
                pm = PromptManager(llm_cfg)
                agent = FeatureAgent(client, pm, llm_cfg.agents.feature_agent)

                result = agent.suggest(
                    df=df.head(1000),
                    current_auc=0.72,
                    current_model="XGBoost",
                )

            st.success("✅ 已生成建议")

            # 展示建议
            suggestions = result.get("suggestions", [])

            if suggestions:
                st.markdown(f"### 💡 共 {len(suggestions)} 条建议")
                for i, sug in enumerate(suggestions):
                    with st.expander(f"建议 {i+1}：{sug.get('name', '未命名')}"):
                        st.markdown(f"**业务含义**: {sug.get('business_meaning', '')}")
                        st.markdown(f"**预期效果**: {sug.get('expected_impact', '')}")
                        st.code(sug.get("code", ""), language="python")
            else:
                st.info("未解析到结构化建议")

            if "general_advice" in result:
                st.markdown("### 整体建议")
                st.info(result["general_advice"])

        except Exception as e:
            st.error(f"调用失败: {e}")
            st.caption("请检查 .env 中的 DEEPSEEK_API_KEY 配置")
