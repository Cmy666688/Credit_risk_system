"""A/B 测试仪表盘工具（扩展点 5）
==================================
把「A/B 测试」页面的纯逻辑（离线实验执行、结果图表构建）从 Streamlit 页面中抽离，
便于单元测试与复用。设计对齐扩展点 4 的 app/components/monitor_utils.py：

    1. run_experiment 复用 src/ab_test/router.py 的 ABTestRouter 与 analyze_experiment，
       不重复实现分流/统计逻辑
    2. 纯逻辑不依赖 streamlit，可直接用 pytest 测试（图表函数返回 plotly Figure）
    3. 默认值写在 DEFAULT_AB_CONFIG，可被 configs/ab_test_config.yaml 覆盖
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from src.ab_test.router import ABTestRouter, analyze_experiment

# ===== 默认配置（可被 configs/ab_test_config.yaml 覆盖）=====
DEFAULT_AB_CONFIG: Dict[str, Any] = {
    "ratio": 0.5,           # 分流到 A 组的比例
    "bucket_count": 10000,  # 哈希分桶数量，决定 ratio 精度
    "alpha": 0.05,          # 显著性水平
    "sample_size": 5000,    # 离线实验样本量
    "group_a_model": "xgboost",
    "group_b_model": "lightgbm",
    "data_source": "auto",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 覆盖 base 的同名键。"""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_ab_config(root: Any, config_path: Optional[Any] = None) -> Dict[str, Any]:
    """加载 A/B 配置：默认值 + configs/ab_test_config.yaml 覆盖。"""
    path = Path(config_path) if config_path else Path(root) / "configs" / "ab_test_config.yaml"
    config = _deep_merge(DEFAULT_AB_CONFIG, {})
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = _deep_merge(config, data)
    return config


def run_experiment(
    model_a,
    model_b,
    X: pd.DataFrame,
    keys: Sequence[Any],
    outcomes: Sequence[Any],
    ratio: float = 0.5,
    bucket_count: int = 10000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """用 ABTestRouter 对样本分流并调用对应模型，产出曝光记录与离线分析结果。

    参数:
        model_a / model_b: 两个模型，需实现 ``predict_proba``
        X: 预处理后的特征矩阵（行与 keys/outcomes 对齐）
        keys: 分流键（本项目用申请编号 id）
        outcomes: 真实结局（违约标签），未知的填 None
        ratio / bucket_count: 透传给 ABTestRouter

    返回:
        (records, result)：records 为曝光记录列表，result 为 analyze_experiment 的输出
    """
    router = ABTestRouter(model_a, model_b, ratio=ratio, bucket_count=bucket_count)
    groups = np.array([router.assign(k) for k in keys])
    mask_a = groups == "A"

    # 按分组批量预测（避免逐行调用模型）
    proba = np.full(len(X), np.nan)
    if mask_a.any():
        proba[mask_a] = np.ravel(model_a.predict_proba(X[mask_a]))
    if (~mask_a).any():
        proba[~mask_a] = np.ravel(model_b.predict_proba(X[~mask_a]))

    records = [
        {
            "key": k,
            "group": g,
            "model_version": "A" if g == "A" else "B",
            "prediction": float(p),
            "outcome": None if o is None else int(o),
        }
        for k, g, p, o in zip(keys, groups, proba, outcomes)
    ]
    return records, analyze_experiment(records)


def bad_rate_figure(result: Dict[str, Any], alpha: float = 0.05) -> go.Figure:
    """A/B 两组违约率对比柱状图（违约率越低越好）。"""
    rate_a = result["A"]["bad_rate"]
    rate_b = result["B"]["bad_rate"]
    n_a, n_b = result["A"]["n"], result["B"]["n"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=["A 组（对照）", "B 组（实验）"],
                y=[rate_a, rate_b],
                text=[f"{rate_a:.2%}（n={n_a}）", f"{rate_b:.2%}（n={n_b}）"],
                textposition="outside",
                marker_color=["#4C78A8", "#F58518"],
            )
        ]
    )

    p_value = result.get("p_value")
    if p_value is None:
        note = "样本不足，无法做显著性检验"
    else:
        sig = "差异显著" if result["significant"] else "差异不显著"
        better = result.get("better_group")
        tail = f"，{better} 组更优" if better else ""
        note = f"两比例 z 检验 p={p_value:.4f}（α={alpha}）：{sig}{tail}"

    fig.update_layout(
        title=f"违约率对比　{note}",
        yaxis_title="违约率",
        yaxis_tickformat=".1%",
        margin=dict(t=60, b=40),
        showlegend=False,
    )
    return fig
