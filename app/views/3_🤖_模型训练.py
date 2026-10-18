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

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin
require_admin()

st.title("🤖 模型训练与评估")
st.caption("交互式训练 XGBoost、LightGBM、MLP，并对比效果")

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
    skip_lgb = st.checkbox("跳过 LightGBM", value=False, help="加快调试速度")

with col2:
    enable_tuning = st.checkbox("启用超参优化", value=False, help="耗时较长（数分钟）")
    n_trials = st.number_input("Optuna trials 数", 5, 200, 30, disabled=not enable_tuning)

with col3:
    model_cfg = configs.get("model")

    st.markdown("**当前关键参数**")
    xgb_params = model_cfg.xgboost.params
    lgb_params = model_cfg.lightgbm.params
    st.caption(f"XGB  max_depth = {xgb_params.max_depth}")
    st.caption(f"XGB  learning_rate = {xgb_params.learning_rate}")
    st.caption(f"LGB  num_leaves = {lgb_params.num_leaves}")
    st.caption(f"LGB  learning_rate = {lgb_params.learning_rate}")
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
                skip_lgb=skip_lgb,
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

# 指标展示名称，与 configs/model_config.yaml 的 evaluation.metrics 一一对应
METRIC_LABELS = {
    "auc": "AUC",
    "ks": "KS",
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "log_loss": "对数损失",
}
DEFAULT_METRICS = ["auc", "ks", "accuracy", "precision", "recall", "f1", "log_loss"]

# 展示哪些指标由配置驱动（evaluation.metrics），避免漏展示已计算的预设指标
try:
    display_metrics = list(configs.get("model").evaluation.metrics)
except Exception:
    display_metrics = DEFAULT_METRICS
if not display_metrics:
    display_metrics = DEFAULT_METRICS

latest_dir = ROOT / "data" / "models" / "latest"
eval_path = latest_dir / "eval_results.json"

# 优先使用本次会话的训练结果（训练刚结束即可展示），否则读取磁盘上的 latest 产物
eval_results = None
eval_source = ""
session_result = st.session_state.get("latest_result")
if isinstance(session_result, dict) and session_result.get("eval_results"):
    eval_results = session_result["eval_results"]
    eval_source = "本次训练（内存缓存）"
elif eval_path.exists():
    try:
        eval_results = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_source = str(eval_path)
    except (json.JSONDecodeError, OSError) as e:
        st.error(
            f"❌ 评估结果文件解析失败：{eval_path}\n\n"
            f"错误信息：{e}\n\n文件可能已损坏，请重新训练生成。"
        )
        eval_results = None

if not eval_results:
    if latest_dir.exists() and any(latest_dir.iterdir()):
        st.error(
            "❌ 检测到模型产物，但缺少评估结果文件 `eval_results.json`。\n\n"
            f"预期路径：{eval_path}\n\n"
            "可能原因：评估步骤未执行或结果未落盘。"
            "请重新训练，并查看控制台 / 日志中的『评估』步骤输出。"
        )
    else:
        st.info("📭 暂无训练结果，请先点击上方 **开始训练** 按钮。")
else:
    st.caption(f"数据来源：{eval_source}")

    def _fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

    # 模型对比
    st.markdown("### 模型对比")
    cmp_data = []
    for model_name, metrics in eval_results.items():
        row = {"模型": model_name.upper()}
        for mk in display_metrics:
            row[METRIC_LABELS.get(mk, mk)] = _fmt(metrics.get(mk))
        cmp_data.append(row)
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    # 详细指标
    st.markdown("### 详细指标")
    selected_model = st.selectbox("选择模型查看详情", list(eval_results.keys()))
    if selected_model:
        metrics = eval_results.get(selected_model, {})
        n_cols = min(4, len(display_metrics))
        cols = st.columns(n_cols)
        for i, mk in enumerate(display_metrics):
            cols[i % n_cols].metric(METRIC_LABELS.get(mk, mk), _fmt(metrics.get(mk)))

    # 提示：可以基于已保存的模型重新计算 ROC/PR/混淆矩阵
    with st.expander("💡 如何获得更详细的可视化（ROC、PR、混淆矩阵）"):
        st.markdown("""
        要展示 ROC 曲线、PR 曲线和混淆矩阵，需要在测试集上重新调用模型。
        可以：
        1. 切换到 "模型解释" 页面查看可视化
        2. 在 Notebook 中加载模型后调用 `Evaluator.get_roc_curve()` 等方法
        3. 启动一次新的训练（结果会缓存）
        """)

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
