"""
🤖 模型训练页面
================
模型选择、超参调整、一键训练、结果可视化。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar
from app.components.chart_utils import (
    confusion_matrix_heatmap,
    roc_curve_plot,
    pr_curve_plot,
    feature_importance_bar,
)

st.set_page_config(page_title="模型训练", page_icon="🤖", layout="wide")
render_sidebar()

st.title("🤖 模型训练与评估")
st.caption("交互式训练 XGBoost、MLP，并对比效果")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))

# ===== 训练控制面板 =====
st.markdown("## ⚙️ 训练配置")

col1, col2, col3 = st.columns(3)

with col1:
    use_sample = st.checkbox("使用示例数据", value=True, help="未勾选则使用真实数据")
    skip_mlp = st.checkbox("跳过 MLP", value=False, help="加快调试速度")

with col2:
    enable_tuning = st.checkbox("启用超参优化", value=False, help="耗时较长（数分钟）")
    n_trials = st.number_input("Optuna trials 数", 5, 200, 30, disabled=not enable_tuning)

with col3:
    st.markdown("**当前 XGBoost 关键参数**")
    model_cfg = configs.get("model")
    xgb_params = model_cfg.xgboost.params
    st.caption(f"max_depth = {xgb_params.max_depth}")
    st.caption(f"learning_rate = {xgb_params.learning_rate}")
    st.caption(f"n_estimators = {xgb_params.n_estimators}")

st.divider()

# ===== 训练按钮 =====
if st.button("🚀 开始训练", type="primary", use_container_width=True):
    from src.pipeline.orchestrator import TrainingPipeline

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(step_name, pct):
        progress_bar.progress(pct / 100)
        status_text.text(f"⏳ 当前步骤：{step_name}")

    try:
        # 临时调整配置
        if enable_tuning:
            configs.get("model").xgboost.tuning.n_trials = int(n_trials)

        pipeline = TrainingPipeline(configs)

        with st.spinner("训练中，请稍候..."):
            result = pipeline.run(
                use_sample=use_sample,
                skip_mlp=skip_mlp,
                skip_tuning=not enable_tuning,
                progress_callback=progress_callback,
            )

        progress_bar.progress(1.0)
        status_text.empty()

        # 保存到 session state，供后续展示
        st.session_state["latest_result"] = result
        st.success(f"✅ 训练完成！耗时 {result['duration']:.2f}s")

    except Exception as e:
        st.error(f"训练失败: {e}")
        st.exception(e)

st.divider()

# ===== 评估结果展示 =====
st.markdown("## 📊 评估结果")

# 加载最新结果
latest_dir = ROOT / "data" / "models" / "latest"
eval_path = latest_dir / "eval_results.json"

if eval_path.exists():
    eval_results = json.loads(eval_path.read_text(encoding="utf-8"))

    # 模型对比
    st.markdown("### 模型对比")
    cmp_data = []
    for model_name, metrics in eval_results.items():
        cmp_data.append({
            "模型": model_name.upper(),
            "AUC": f"{metrics['auc']:.4f}",
            "KS": f"{metrics['ks']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
        })
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    # 关键指标
    st.markdown("### 详细指标")

    selected_model = st.selectbox("选择模型查看详情", list(eval_results.keys()))
    if selected_model:
        metrics = eval_results[selected_model]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC", f"{metrics['auc']:.4f}")
        col2.metric("KS", f"{metrics['ks']:.4f}")
        col3.metric("F1", f"{metrics['f1']:.4f}")
        col4.metric("准确率", f"{metrics['accuracy']:.4f}")

        col5, col6 = st.columns(2)
        col5.metric("精确率", f"{metrics['precision']:.4f}")
        col6.metric("召回率", f"{metrics['recall']:.4f}")

    # 提示：可以基于已保存的模型重新计算 ROC/PR/混淆矩阵
    with st.expander("💡 如何获得更详细的可视化（ROC、PR、混淆矩阵）"):
        st.markdown("""
        要展示 ROC 曲线、PR 曲线和混淆矩阵，需要在测试集上重新调用模型。
        可以：
        1. 切换到 "模型解释" 页面查看可视化
        2. 在 Notebook 中加载模型后调用 `Evaluator.get_roc_curve()` 等方法
        3. 启动一次新的训练（结果会缓存）
        """)
else:
    st.info("📭 暂无训练结果，请先点击上方 **开始训练** 按钮。")

# ===== 训练历史 =====
st.divider()
st.markdown("## 📜 历史训练记录")

models_dir = ROOT / "data" / "models"
if models_dir.exists():
    runs = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name != "latest"],
        reverse=True,
    )

    if runs:
        runs_data = []
        for run_dir in runs[:10]:  # 最近 10 次
            meta_path = run_dir / "meta.json"
            eval_path_run = run_dir / "eval_results.json"

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                row = {
                    "运行 ID": run_dir.name,
                    "时间": meta.get("timestamp", "")[:19],
                    "训练样本": meta.get("train_samples", 0),
                    "特征数": len(meta.get("feature_names", [])),
                }

                if eval_path_run.exists():
                    eval_data = json.loads(eval_path_run.read_text(encoding="utf-8"))
                    if "ensemble" in eval_data:
                        row["AUC"] = round(eval_data["ensemble"]["auc"], 4)
                    elif "xgboost" in eval_data:
                        row["AUC"] = round(eval_data["xgboost"]["auc"], 4)

                runs_data.append(row)

        if runs_data:
            st.dataframe(pd.DataFrame(runs_data), use_container_width=True, hide_index=True)
    else:
        st.caption("暂无历史记录")
