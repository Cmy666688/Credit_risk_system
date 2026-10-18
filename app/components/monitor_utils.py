"""
监控仪表盘工具
==============
把「模型监控仪表盘」的纯逻辑（数据筛选、KPI 计算、漂移检测、告警判定、图表构建）
从 Streamlit 页面中抽离出来，便于单元测试与在其他页面复用。

设计要点:
    1. 纯函数、不依赖 st —— 可直接用 pytest 测试（图表函数返回 plotly Figure）
    2. 配置可扩展：默认值写在 DEFAULT_MONITOR_CONFIG，可被
       configs/monitoring_config.yaml 覆盖
    3. 漂移检测复用扩展点 4 的 DriftDetector（KS + PSI）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from src.monitoring.drift_detector import DriftDetector

# ===== 默认配置（可被 configs/monitoring_config.yaml 覆盖）=====
DEFAULT_MONITOR_CONFIG: Dict[str, Any] = {
    "drift": {
        "threshold": 0.1,          # KS/PSI 漂移阈值
        "missing_threshold": 0.1,  # 缺失率变化阈值
        "top_n": 15,               # 漂移榜单展示的 Top N
    },
    "alert": {
        "bad_rate": 0.25,           # 违约率告警阈值
        "avg_interest_rate": 20.0,  # 平均利率告警阈值（%）
        "drifted_features": 5,      # 漂移特征数告警阈值
    },
    # 展示维度：列名 -> 中文标签（可自由扩展）
    "dimensions": {
        "grade": "贷款等级",
        "purpose": "贷款用途",
        "homeOwnership": "房屋所有权",
        "verificationStatus": "验证状态",
    },
    "live_window": 60,       # 实时模拟流的窗口长度（分钟）
    "live_seed": 42,         # 实时模拟流随机种子（可复现）
    "date_column": "issueDate",
    "target_column": "isDefault",
    "id_column": "id",
}


# ===== 配置加载 =====

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并字典，override 覆盖 base 的同名键。"""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_monitor_config(root: Any, config_path: Optional[Any] = None) -> Dict[str, Any]:
    """加载监控配置：默认值 + configs/monitoring_config.yaml 覆盖。"""
    path = Path(config_path) if config_path else Path(root) / "configs" / "monitoring_config.yaml"
    config = _deep_merge(DEFAULT_MONITOR_CONFIG, {})
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = _deep_merge(config, data)
    return config


# ===== 数据源 =====

def available_sources(root: Any) -> List[Dict[str, Any]]:
    """返回可选数据源（含是否可用标记）。"""
    root = Path(root)
    return [
        {"key": "sample", "label": "合成示例数据",
         "path": root / "data" / "sample" / "sample.csv",
         "available": (root / "data" / "sample" / "sample.csv").exists()},
        {"key": "real", "label": "真实天池数据",
         "path": root / "data" / "raw" / "train.csv",
         "available": (root / "data" / "raw" / "train.csv").exists()},
        {"key": "lending_club", "label": "Lending Club",
         "path": root / "data" / "external" / "lending_club.csv",
         "available": (root / "data" / "external" / "lending_club.csv").exists()},
        # 实时模拟流始终可用，用于演示手动刷新后的数据演进
        {"key": "live", "label": "实时模拟流", "path": None, "available": True},
    ]


# ===== 数据筛选 / 聚合 =====

def filter_by_time_range(
    df: pd.DataFrame,
    date_col: str,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """按时间范围筛选（基于 issueDate 之类的日期列）。"""
    if df.empty or date_col not in df.columns or (start is None and end is None):
        return df
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.notna().sum() == 0:
        # 整列都无法解析为日期（如某些外部数据源），不做筛选
        return df
    mask = dt.notna()
    if start is not None:
        mask &= dt >= pd.Timestamp(start)
    if end is not None:
        mask &= dt <= pd.Timestamp(end)
    return df[mask].copy()


def aggregate_time_series(
    df: pd.DataFrame,
    date_col: str = "issueDate",
    target_col: str = "isDefault",
    interest_col: str = "interestRate",
) -> pd.DataFrame:
    """按「月」聚合出趋势序列：申请量 / 违约率 / 平均利率。"""
    empty = pd.DataFrame(columns=["时间", "申请量", "违约率", "平均利率"])
    if df.empty or date_col not in df.columns:
        return empty
    dt = pd.to_datetime(df[date_col], errors="coerce")
    tmp = df.assign(时间=dt.dt.to_period("M").dt.to_timestamp())
    tmp = tmp[tmp["时间"].notna()]
    if tmp.empty:
        return empty
    agg = {"申请量": ("时间", "size")}
    if target_col in tmp.columns:
        agg["违约率"] = (target_col, "mean")
    if interest_col in tmp.columns:
        agg["平均利率"] = (interest_col, "mean")
    out = tmp.groupby("时间").agg(**agg).reset_index()
    if "违约率" not in out.columns:
        out["违约率"] = np.nan
    if "平均利率" not in out.columns:
        out["平均利率"] = np.nan
    return out.sort_values("时间").reset_index(drop=True)


# ===== KPI =====

def compute_kpis(
    df: pd.DataFrame,
    target_col: str = "isDefault",
    interest_col: str = "interestRate",
    loan_col: str = "loanAmnt",
) -> Dict[str, float]:
    """计算监控 KPI：样本量 / 违约率 / 平均利率 / 平均贷款额。"""
    n = len(df)
    nan = float("nan")
    return {
        "samples": n,
        "bad_rate": float(df[target_col].mean()) if n and target_col in df.columns else nan,
        "avg_interest_rate": float(df[interest_col].mean()) if n and interest_col in df.columns else nan,
        "avg_loan": float(df[loan_col].mean()) if n and loan_col in df.columns else nan,
    }


def kpis_from_live(live_df: pd.DataFrame) -> Dict[str, float]:
    """实时模拟流的 KPI（字段为 申请量/违约率/平均利率）。"""
    if live_df.empty:
        return {"samples": 0, "bad_rate": float("nan"),
                "avg_interest_rate": float("nan"), "avg_loan": float("nan")}
    return {
        "samples": int(live_df["申请量"].sum()),
        "bad_rate": float(live_df["违约率"].mean()),
        "avg_interest_rate": float(live_df["平均利率"].mean()),
        "avg_loan": float("nan"),
    }


# ===== 漂移检测（复用扩展点 4）=====

def compute_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: Dict[str, Any],
    exclude_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """检测 current 相对 baseline 的特征漂移，返回便于展示的明细表。"""
    columns = ["特征", "方法", "统计量", "缺失率变化", "缺失漂移"]
    if baseline_df is None or current_df is None or baseline_df.empty or current_df.empty:
        return pd.DataFrame(columns=columns)

    # 日期列在漂移检测前先排除（字符串日期无法做 KS/PSI，且本身不做监控）
    drop_date = {date_col} if date_col else set()
    exclude = set(exclude_cols or []) | drop_date

    drift_cfg = config.get("drift", {})
    detector = DriftDetector(
        baseline_df.drop(columns=list(drop_date), errors="ignore"),
        exclude_cols=list(exclude),
        categorical_cols=categorical_cols or [],
    )
    # 当前数据只保留与基线同名的列，避免凭空多出的列干扰
    current = current_df.drop(columns=list(drop_date), errors="ignore")
    result = detector.detect(
        current,
        threshold=drift_cfg.get("threshold", 0.1),
        missing_threshold=drift_cfg.get("missing_threshold", 0.1),
    )

    rows = []
    for feature, info in result.items():
        stat = info.get("ks_stat", info.get("psi"))
        rows.append({
            "特征": feature,
            "方法": "KS" if info.get("method") == "ks" else "PSI",
            "统计量": float(stat) if stat is not None else float("nan"),
            "缺失率变化": float(info.get("missing_delta", 0.0)),
            "缺失漂移": bool(info.get("missing_drift", False)),
        })
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("统计量", ascending=False).reset_index(drop=True)
    return df


# ===== 告警判定 =====

def evaluate_alerts(
    kpis: Dict[str, float],
    drift_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    """超出阈值时生成告警列表（error 级别优先于 warning）。"""
    alert_cfg = config.get("alert", {})
    alerts: List[Dict[str, str]] = []

    bad_rate = kpis.get("bad_rate")
    if bad_rate is not None and not np.isnan(bad_rate) and bad_rate > alert_cfg.get("bad_rate", 1.0):
        alerts.append({
            "level": "error",
            "title": "违约率超阈值",
            "message": f"当前违约率 {bad_rate:.2%}，超过告警阈值 {alert_cfg['bad_rate']:.2%}",
            "metric": "bad_rate",
        })

    avg_rate = kpis.get("avg_interest_rate")
    thr_rate = alert_cfg.get("avg_interest_rate", float("inf"))
    if avg_rate is not None and not np.isnan(avg_rate) and avg_rate > thr_rate:
        alerts.append({
            "level": "warning",
            "title": "平均利率偏高",
            "message": f"当前平均利率 {avg_rate:.2f}%，超过告警阈值 {thr_rate:.2f}%",
            "metric": "avg_interest_rate",
        })

    n_drift = 0 if drift_df is None or drift_df.empty else len(drift_df)
    thr_drift = alert_cfg.get("drifted_features", 10 ** 9)
    if n_drift > thr_drift:
        alerts.append({
            "level": "warning",
            "title": "漂移特征过多",
            "message": f"检测到 {n_drift} 个特征发生漂移，超过告警阈值 {thr_drift} 个",
            "metric": "drifted_features",
        })
    return alerts


# ===== 图表构建 =====

def trend_figure(ts_df: pd.DataFrame, title: str = "申请量与违约率趋势") -> Optional[go.Figure]:
    """折线图 + 柱状图组合：申请量（柱）与违约率（折线，右轴）。"""
    if ts_df is None or ts_df.empty or "申请量" not in ts_df.columns:
        return None
    fig = go.Figure()
    fig.add_bar(
        x=ts_df["时间"], y=ts_df["申请量"], name="申请量",
        marker_color="#4C78A8", yaxis="y",
    )
    if "违约率" in ts_df.columns:
        fig.add_scatter(
            x=ts_df["时间"], y=ts_df["违约率"], name="违约率 (%)",
            mode="lines+markers", line=dict(color="#E45756", width=2), yaxis="y2",
        )
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        xaxis=dict(title=""),
        yaxis=dict(title="申请量"),
        yaxis2=dict(
            title="违约率", overlaying="y", side="right",
            tickformat=".1%", showgrid=False,
        ),
    )
    return fig


def dimension_pie_figure(
    df: pd.DataFrame,
    dim_col: str,
    dim_label: Optional[str] = None,
    top_n: int = 8,
) -> Optional[go.Figure]:
    """饼图：按展示维度统计占比。"""
    if df is None or df.empty or dim_col not in df.columns:
        return None
    counts = df[dim_col].astype(str).value_counts().head(top_n)
    chart_df = counts.rename_axis(dim_col).reset_index(name="数量")
    fig = px.pie(
        chart_df, names=dim_col, values="数量", hole=0.45,
        title=f"{dim_label or dim_col} 分布",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def drift_bar_figure(
    drift_df: pd.DataFrame,
    threshold: float = 0.1,
    top_n: int = 15,
    title: str = "特征漂移 Top N",
) -> Optional[go.Figure]:
    """水平柱状图：各特征漂移统计量（含阈值参考线）。

    可读性优化：
        - 柱上直接标注「统计量 (方法)」，无需对照坐标轴读数
        - 悬停显示缺失率变化与是否缺失率漂移
        - 按漂移类型着色：红=分布漂移（超阈值）、橙=缺失率漂移、蓝=其它
        - 高度随条目数量自适应，避免少量特征时柱子被拉成一块实心矩形
    """
    if drift_df is None or drift_df.empty:
        return None

    data = drift_df.head(top_n).sort_values("统计量", ascending=True)
    raw = data["统计量"].astype(float).to_numpy()
    stats = np.nan_to_num(raw, nan=0.0)          # 画图用，NaN 归零
    colors = [
        "#E45756" if v > threshold else ("#F2B134" if md else "#4C78A8")
        for v, md in zip(raw, data["缺失漂移"])
    ]
    text = [
        ("— " if np.isnan(v) else f"{v:.3f} ") + f"({m})"
        for v, m in zip(raw, data["方法"])
    ]
    hover = [
        f"<b>{f}</b><br>方法：{m}<br>统计量：{'—' if np.isnan(v) else f'{v:.4f}'}"
        f"<br>缺失率变化：{md:.2%}" + ("<br>⚠ 存在缺失率漂移" if nd else "")
        for f, m, v, md, nd in zip(
            data["特征"], data["方法"], raw, data["缺失率变化"], data["缺失漂移"]
        )
    ]

    fig = go.Figure(go.Bar(
        x=stats,
        y=data["特征"],
        orientation="h",
        marker_color=colors,
        text=text,
        textposition="outside",
        cliponaxis=False,
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.add_vline(
        x=threshold,
        line=dict(color="#666", dash="dash", width=1.5),
        annotation_text=f"阈值 {threshold}",
        annotation_position="top right",
    )

    # 为柱外的数值标签留出横向空间
    x_max = max(float(np.max(stats)) if len(stats) else 0.0, float(threshold))
    fig.update_layout(
        title=title,
        height=int(max(220, min(700, 46 * len(data) + 150))),
        margin=dict(l=10, r=90, t=50, b=10),
        bargap=0.35,
        xaxis=dict(
            title="漂移统计量（KS/PSI）",
            range=[0, x_max * 1.32],
            rangemode="tozero",
        ),
        yaxis=dict(title="", automargin=True),
    )
    return fig


# ===== 实时模拟流 =====

def make_live_stream(window: int = 60, tick: int = 0, seed: int = 42) -> pd.DataFrame:
    """生成一段可复现的「模拟实时流」（分钟级），tick 越大数据越往后演进。

    用于演示手动刷新：每次点击「刷新数据」tick+1，图表随之变化。
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(window) + int(tick)
    applications = rng.poisson(np.clip(120 + 30 * np.sin(idx / 6.0), 1, None))
    bad_rate = np.clip(0.15 + 0.05 * np.sin(idx / 9.0) + rng.normal(0, 0.008, window), 0.0, 1.0)
    avg_interest = 12.0 + 2.0 * np.sin(idx / 7.0) + rng.normal(0, 0.25, window)
    now = pd.Timestamp.now().floor("min")
    times = pd.date_range(end=now, periods=window, freq="min")
    return pd.DataFrame({
        "时间": times,
        "申请量": applications.astype(int),
        "违约率": bad_rate,
        "平均利率": avg_interest,
    })
