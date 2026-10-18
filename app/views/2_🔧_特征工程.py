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

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin
require_admin()

st.title("🔧 特征工程")
st.caption("特征工程决定模型上限")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))


# ===== 数据加载（统一回退逻辑：真实数据 > 合成数据） =====
from app.components.data_utils import render_data_status_or_stop

df = render_data_status_or_stop(ROOT, prefer="auto")


def build_transform_compare_table(prep, X, X_processed, feature_cfg, data_cfg) -> pd.DataFrame:
    """逐列变换对照：原始列 → 处理后列名 / 变换链 / 同一行的变换前后取值。

    变换链与列名取自预处理器 fit 出来的状态，保证展示的就是真实发生的步骤。
    """
    custom = feature_cfg.encoding.get("custom", {})
    scale_method = feature_cfg.scaling.method
    dt_cols = list(data_cfg.columns.datetime)

    def fmt(value):
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return round(float(value), 4)
        return value

    populated = X.dropna()
    row = populated.index[0] if len(populated) else X.index[0]
    rows = []

    for col in X.columns:
        if col in prep._dropped_cols:
            rows.append({
                "列名": col, "处理后列名": "—（已丢弃）",
                "变换类型": "丢弃（缺失率 > 阈值）",
                "变换前": X.at[row, col], "变换后": "—",
            })
            continue

        steps = []
        if col in prep._col_strategies:
            steps.append(f"填充·{prep._col_strategies[col]}")
        if col in prep._outlier_clips:
            steps.append("异常值截断")
        if col in custom:
            steps.append("自定义映射")
        elif col in prep._label_encoders:
            steps.append("标签编码")
        if col in dt_cols:
            steps.append("时间派生")

        if col in dt_cols:
            derived = [d for d in (f"{col}_year", f"{col}_month") if d in X_processed.columns]
            out_name = ", ".join(derived) if derived else "—"
            after = " / ".join(str(fmt(X_processed.at[row, d])) for d in derived) if derived else "—"
        else:
            out_name = col
            if col in X_processed.columns:
                after = fmt(X_processed.at[row, col])
                if scale_method != "none":
                    steps.append(f"标准化·{scale_method}")
            else:
                after = "—"

        rows.append({
            "列名": col, "处理后列名": out_name,
            "变换类型": " → ".join(steps) if steps else "保持原样",
            "变换前": X.at[row, col], "变换后": after,
        })

    return pd.DataFrame(rows)


# ===== Tabs =====
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ 预处理流程",
    "🔬 衍生特征",
    "✂️ 特征筛选",
    "🤖 LLM 特征建议",
])

with tab1:
    st.markdown("## 数据预处理流程")
    st.caption("4 个步骤按顺序执行，全部参数来自 `configs/feature_config.yaml`")

    feature_cfg = configs.get("feature")
    mv = feature_cfg.missing_value
    data_cols = configs.get("data").columns
    numeric_cols = set(data_cols.numerical) | set(data_cols.anonymous)
    per_col = dict(mv.get("strategy", {}) or {})

    # ---- 流程总览：4 张步骤卡片 ----
    step1, step2, step3, step4 = st.columns(4, gap="medium")
    with step1:
        with st.container(border=True):
            st.markdown("#### 1️⃣ 缺失值处理")
            st.markdown(
                f"- 数值列：`{mv.numerical_strategy}`\n"
                f"- 类别列：`{mv.categorical_strategy}`\n"
                f"- 丢弃阈值：`{mv.drop_threshold}`\n"
                f"- 按列覆盖：`{len(per_col)}` 列"
            )
    with step2:
        with st.container(border=True):
            st.markdown("#### 2️⃣ 异常值处理")
            st.markdown(
                f"- 方法：`{feature_cfg.outlier.method}`\n"
                f"- 下分位：`{feature_cfg.outlier.lower_quantile}`\n"
                f"- 上分位：`{feature_cfg.outlier.upper_quantile}`"
            )
    with step3:
        with st.container(border=True):
            st.markdown("#### 3️⃣ 类别编码")
            st.markdown(
                f"- 默认方法：`{feature_cfg.encoding.method}`\n"
                f"- 自定义映射：`{len(feature_cfg.encoding.get('custom', {}))}` 个字段"
            )
    with step4:
        with st.container(border=True):
            st.markdown("#### 4️⃣ 数值标准化")
            st.markdown(f"- 方法：`{feature_cfg.scaling.method}`")

    # ---- 扩展点 6：按列填充策略 ----
    st.markdown("#### 🧩 按列填充策略（扩展点 6）")
    st.markdown(
        "`missing_value.strategy` 可按列覆盖全局策略，未列出的列沿用 "
        "`numerical_strategy` / `categorical_strategy`；新增的 `knn` 由 "
        "`Preprocessor` 在训练集上拟合、推理时复用（避免数据泄漏，单笔推理也可用）。"
    )

    if per_col:
        strat_rows = [
            {
                "列名": col,
                "类型": "数值" if col in numeric_cols else "类别",
                "按列策略": strategy,
                "类型默认": (mv.numerical_strategy if col in numeric_cols
                             else mv.categorical_strategy),
            }
            for col, strategy in per_col.items()
        ]
        st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    knn_cols = [c for c, s in per_col.items() if s == "knn"]
    if knn_cols:
        st.markdown(
            f"**KNN 参数**：邻居数 `{mv.get('knn_neighbors', 5)}`、"
            f"参考样本上限 `{mv.get('knn_reference_sample', 2000)}`、"
            f"分批大小 `{mv.get('knn_batch_size', 10000)}`；"
            f"启用 knn 的列：`{knn_cols}`"
        )

    # ---- 实时演示 ①：常规预处理 ----
    st.divider()
    st.markdown("### 🔥 实时演示")

    with st.container(border=True):
        st.markdown("#### ① 常规预处理：对前 1000 条数据执行完整流程")
        if st.button("🚀 执行预处理", type="primary"):
            from src.data.preprocessor import Preprocessor

            with st.spinner("处理中..."):
                sample_df = df.head(1000).copy()
                target = configs.get("data").target_column
                id_col = configs.get("data").id_column
                X = sample_df.drop(columns=[target, id_col], errors="ignore").copy()

                prep = Preprocessor(configs.get("data"), feature_cfg)
                X_processed = prep.fit_transform(X)

            k1, k2, k3 = st.columns(3)
            k1.metric("样本数", f"{X.shape[0]:,}")
            k2.metric("处理前缺失值", f"{int(X.isnull().sum().sum()):,}")
            k3.metric("处理后缺失值", f"{int(X_processed.isnull().sum().sum()):,}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始数据**")
                st.dataframe(X.head(), use_container_width=True)
            with col2:
                st.markdown("**处理后数据**")
                st.dataframe(X_processed.head(), use_container_width=True)

            st.caption(f"处理后 shape：{X_processed.shape}（列数会因时间派生而增加）")

            st.markdown("**逐列变换对照**")
            st.caption(
                "同一行数据的「变换前 → 变换后」，以及这一列实际走过的步骤。"
                f"因 `scaling.method = {feature_cfg.scaling.method}`，数值列最终会变成 z-score，"
                "所以「变换后」出现负值、且量级远小于原始值是正常的。"
            )
            st.dataframe(
                build_transform_compare_table(prep, X, X_processed, feature_cfg, configs.get("data")),
                use_container_width=True, hide_index=True,
            )
            st.success("✅ 预处理完成")

    # ---- 实时演示 ②：按列填充策略（扩展点 6）----
    with st.container(border=True):
        st.markdown("#### ② 按列填充策略：注入 10% 缺失后按策略填充")
        st.caption(
            "人为把 annualIncome / dti / employmentLength 各挖空 10%，观察各列策略的实际效果"
        )

        if st.button("🧪 演示：注入缺失值后按列策略填充", type="primary"):
            from src.data.preprocessor import Preprocessor

            with st.spinner("拟合并填充..."):
                sample_df = df.head(1000).copy()
                target = configs.get("data").target_column
                id_col = configs.get("data").id_column
                X = sample_df.drop(columns=[target, id_col], errors="ignore").copy()

                demo_cols = [c for c in ["annualIncome", "dti", "employmentLength"]
                             if c in X.columns]
                X_true = X.copy()   # 注入缺失前的真实值，用于评估填充质量
                rng = np.random.default_rng(42)
                for c in demo_cols:
                    idx = rng.choice(len(X), size=max(1, int(len(X) * 0.1)), replace=False)
                    X.iloc[idx, X.columns.get_loc(c)] = np.nan

                prep = Preprocessor(configs.get("data"), feature_cfg)
                X_processed = prep.fit_transform(X)

            summary = pd.DataFrame([
                {
                    "列名": c,
                    "生效策略": prep._col_strategies.get(c, "-"),
                    "注入缺失": int(X[c].isnull().sum()),
                    "填充后缺失": int(X_processed[c].isnull().sum()),
                }
                for c in demo_cols
            ])
            st.markdown("**注入缺失 → 填充结果**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            k1, k2 = st.columns(2)
            k1.metric("KNN 是否激活", "✅ 是" if prep._knn_imputer is not None else "❌ 否")
            k2.metric("KNN 参考特征列", f"{len(prep._knn_reference_cols)} 个")

            # ---- 策略 A/B 对比：改动前（纯全局策略） vs 改动后（按列覆盖 + knn）----
            st.markdown("**策略 A/B 对比：改动前 vs 改动后**")
            st.caption(
                "同一份注入缺失的数据，分别用「改动前的全局策略」和「改动后的按列策略」填充，"
                "再回注入前的真实值算误差。为避免标准化干扰，这里只比较填充阶段（原始尺度）。"
            )

            import copy
            from src.utils.config_loader import AttrDict

            old_feature_cfg = AttrDict(copy.deepcopy(dict(feature_cfg)))
            old_feature_cfg.missing_value.strategy = {}   # 清空按列覆盖 -> 纯全局策略

            prep_old = Preprocessor(configs.get("data"), old_feature_cfg).fit(X)
            filled_old = prep_old._apply_imputers(X.copy())
            filled_new = prep._apply_imputers(X.copy())

            numeric_set = set(data_cols.numerical) | set(data_cols.anonymous)
            comp_rows, errs = [], {}
            for c in demo_cols:
                mask = X[c].isnull()
                if not mask.any():
                    continue
                if c in numeric_set:
                    true = X_true.loc[mask, c].astype(float)
                    err_old = (filled_old.loc[mask, c].astype(float) - true).abs().mean()
                    err_new = (filled_new.loc[mask, c].astype(float) - true).abs().mean()
                    err_name = "MAE"
                else:
                    true = X_true.loc[mask, c].astype(str)
                    err_old = (filled_old.loc[mask, c].astype(str) != true).mean()
                    err_new = (filled_new.loc[mask, c].astype(str) != true).mean()
                    err_name = "错误率"

                errs[c] = (float(err_old), float(err_new), err_name)
                if err_new < err_old:
                    verdict = "✅ 改动后更优"
                elif err_new > err_old:
                    verdict = "⚠️ 改动前更优"
                else:
                    verdict = "— 持平"
                comp_rows.append({
                    "列名": c,
                    "旧策略": prep_old._col_strategies.get(c, "-"),
                    "新策略": prep._col_strategies.get(c, "-"),
                    f"旧{err_name}": round(float(err_old), 4),
                    f"新{err_name}": round(float(err_new), 4),
                    "结论": verdict,
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            if "dti" in errs:
                mae_old, mae_new, _ = errs["dti"]
                k1, k2, k3 = st.columns(3)
                k1.metric("dti · 旧策略 MAE（median）", f"{mae_old:.4f}")
                k2.metric("dti · 新策略 MAE（knn）", f"{mae_new:.4f}",
                          delta=f"{mae_new - mae_old:.4f}", delta_color="inverse")
                k3.metric("误差下降", f"{(mae_old - mae_new) / mae_old * 100:.1f}%"
                          if mae_old else "—")

                st.markdown("**抽样明细：真实值 vs 两种策略的填充值（dti）**")
                mask = X["dti"].isnull()
                detail = pd.DataFrame({
                    "annualIncome": X.loc[mask, "annualIncome"].head(8).round(0),
                    "dti 真实值": X_true.loc[mask, "dti"].head(8).round(2),
                    "旧策略(median)": filled_old.loc[mask, "dti"].head(8).round(2),
                    "新策略(knn)": filled_new.loc[mask, "dti"].head(8).round(2),
                })
                detail["旧误差"] = (detail["旧策略(median)"] - detail["dti 真实值"]).abs().round(2)
                detail["新误差"] = (detail["新策略(knn)"] - detail["dti 真实值"]).abs().round(2)
                st.dataframe(detail, use_container_width=True)

            st.success("✅ 按列策略填充完成")

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
