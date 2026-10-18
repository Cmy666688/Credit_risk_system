"""
📈 模型监控仪表盘
==================
生产监控视图：数据量 / 违约率 / 利率等指标卡片、趋势折线图、
维度分布饼图、特征漂移柱状图，并在指标超阈值时高亮告警。
支持按数据源 / 时间范围 / 展示维度筛选，并通过「刷新数据」按钮手动刷新。
监控配置区位于主内容区顶部（原侧边栏内容已迁移至此）；全局侧边栏仅保留系统状态。

数据来源：
    - 合成示例数据 / 真实天池数据 / Lending Club（静态，可做时间范围筛选）
    - 实时模拟流（每次点击「刷新数据」，数据向前演进）

运行: streamlit run app/streamlit_app.py  →  左侧导航选择「模型监控」
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components import monitor_utils as mu
from app.components.sidebar import render_sidebar
from src.utils.config_loader import ConfigLoader

# ===== 页面配置 =====
st.set_page_config(page_title="模型监控", page_icon="📈", layout="wide")
render_sidebar()

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin
require_admin()

CONFIGS = ConfigLoader(str(ROOT / "configs"))
DATA_CFG = CONFIGS.get("data")
MONITOR_CFG = mu.load_monitor_config(ROOT)

# 漂移检测用的列配置：排除 id 与标签列；类别列强制走 PSI
EXCLUDE_COLS = [DATA_CFG.id_column, DATA_CFG.target_column]
CATEGORICAL_COLS = list(DATA_CFG.columns.categorical)


# ===== 数据加载（带缓存）=====
@st.cache_data(show_spinner=False)
def _load_static(source_key: str, nrows=None) -> pd.DataFrame:
    """加载静态数据源：合成 / 真实 / Lending Club（归一化后）。"""
    if source_key == "lending_club":
        from app.components.data_utils import load_lending_club_data
        _, normalized = load_lending_club_data(str(ROOT))
        return normalized

    path = (
        ROOT / "data" / "sample" / "sample.csv"
        if source_key == "sample"
        else ROOT / "data" / "raw" / "train.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, nrows=nrows)


# ===== 页面标题 =====
st.title("📈 模型监控仪表盘")
st.caption("监控生产数据的规模、质量与分布稳定性；指标越限自动告警，可手动刷新。")

# ===== 监控配置区（原侧边栏内容，已迁移至主内容区）=====
with st.container(border=True):
    st.markdown("#### ⚙️ 监控配置")

    # ---- 数据源选择（只列出实际可用的数据源）----
    sources = [s for s in mu.available_sources(ROOT) if s["available"]]
    src_labels = [s["label"] for s in sources]
    default_src = "合成示例数据" if "合成示例数据" in src_labels else src_labels[0]
    source_label = st.segmented_control(
        "数据源", src_labels, default=default_src, key="mon_source",
    ) or default_src
    source = next(s for s in sources if s["label"] == source_label)

    # ---- 手动刷新（与「数据探索」页面保持一致：清缓存 + 重跑）----
    if st.button("🔄 刷新数据", use_container_width=True):
        st.session_state["mon_live_tick"] = int(st.session_state.get("mon_live_tick", 0)) + 1
        st.cache_data.clear()
        st.rerun()

    # ---- 阈值配置（两列并排，窄屏自动换行）----
    thr_left, thr_right = st.columns(2)
    with thr_left:
        threshold = st.number_input(
            "漂移阈值 (KS/PSI)", min_value=0.01, max_value=0.5,
            value=float(MONITOR_CFG["drift"]["threshold"]), step=0.01, key="mon_threshold",
        )
    with thr_right:
        bad_rate_thr = st.number_input(
            "违约率告警阈值", min_value=0.0, max_value=1.0,
            value=float(MONITOR_CFG["alert"]["bad_rate"]), step=0.01,
            format="%.2f", key="mon_bad_rate_thr",
        )

    # ---- 加载数据（供时间范围 / 展示维度使用）----
    is_live = source["key"] == "live"
    df_full = pd.DataFrame()

    if not is_live:
        nrows = None
        if source["key"] == "real":
            nrows = st.slider("读取行数（真实数据较大）", 10_000, 800_000, 100_000, 10_000, key="mon_nrows")
        df_full = _load_static(source["key"], nrows)
        if df_full.empty:
            st.error(f"❌ {source['label']} 加载失败或文件不存在")
            st.info("请先准备数据（见「数据探索」页面提示），或切换到其它数据源。")
            st.stop()

    # ---- 时间范围 / 展示维度（依赖已加载的数据）----
    date_range = None
    dim_col = None
    dim_label = None

    if not is_live:
        date_col = MONITOR_CFG["date_column"]
        dt = pd.to_datetime(df_full[date_col], errors="coerce") if date_col in df_full.columns else None

        filter_left, filter_right = st.columns(2)
        with filter_left:
            if dt is not None and dt.notna().sum() > 0:
                d_min, d_max = dt.min().date(), dt.max().date()
                date_range = st.date_input(
                    "时间范围", value=(d_min, d_max),
                    min_value=d_min, max_value=d_max, key="mon_date",
                )
        with filter_right:
            dim_options = [c for c in MONITOR_CFG["dimensions"] if c in df_full.columns]
            if dim_options:
                dim_col = st.selectbox(
                    "展示维度", dim_options,
                    format_func=lambda c: MONITOR_CFG["dimensions"].get(c, c),
                    key="mon_dim",
                )
                dim_label = MONITOR_CFG["dimensions"].get(dim_col, dim_col)


# ===== 运行时配置（把配置区阈值合并进来）=====
runtime_cfg = {
    "drift": {
        "threshold": threshold,
        "missing_threshold": MONITOR_CFG["drift"]["missing_threshold"],
    },
    "alert": {**MONITOR_CFG["alert"], "bad_rate": bad_rate_thr},
}


def _metric(label: str, value: str, alert: bool = False, alert_text: str = "超阈值") -> None:
    """指标卡片；告警时高亮（红色 delta + ⚠️ 标记）。"""
    st.metric(
        f"{label}{' ⚠️' if alert else ''}",
        value,
        delta=alert_text if alert else None,
        delta_color="inverse" if alert else "normal",
        border=True,
    )


# ===== 监控面板 =====
def monitor_panel() -> None:
    now = datetime.now().strftime("%H:%M:%S")

    # ---- 取数 & 计算 ----
    if is_live:
        # 实时模拟流：每次点击「刷新数据」tick+1，数据向前演进
        tick = int(st.session_state.get("mon_live_tick", 0))
        current = mu.make_live_stream(MONITOR_CFG["live_window"], tick, MONITOR_CFG["live_seed"])
        baseline = mu.make_live_stream(MONITOR_CFG["live_window"], 0, MONITOR_CFG["live_seed"])
        kpis = mu.kpis_from_live(current)
        ts_df, pie_fig = current, None
        drift_df = mu.compute_drift(baseline, current, runtime_cfg)
    else:
        start = end = None
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = date_range
        elif date_range is not None:
            start = end = date_range

        current = mu.filter_by_time_range(df_full, MONITOR_CFG["date_column"], start, end)
        kpis = mu.compute_kpis(current, MONITOR_CFG["target_column"])
        ts_df = mu.aggregate_time_series(
            current, MONITOR_CFG["date_column"], MONITOR_CFG["target_column"],
        )
        drift_df = mu.compute_drift(
            df_full, current, runtime_cfg,
            exclude_cols=EXCLUDE_COLS, categorical_cols=CATEGORICAL_COLS,
            date_col=MONITOR_CFG["date_column"],
        )
        pie_fig = mu.dimension_pie_figure(current, dim_col, dim_label) if dim_col else None

    # ---- 状态行 ----
    left, right = st.columns([3, 2])
    with left:
        st.caption(f"数据源：{source_label}")
    with right:
        st.caption(f"⏱️ 刷新时间：{now}")

    # ---- 告警提示 ----
    alerts = mu.evaluate_alerts(kpis, drift_df, runtime_cfg)
    if alerts:
        for a in alerts:
            if a["level"] == "error":
                st.error(a["message"], title=a["title"], icon="🚨")
            else:
                st.warning(a["message"], title=a["title"], icon="⚠️")
    else:
        st.success("各项监控指标均在阈值内", title="监控正常", icon="✅")

    # ---- KPI 指标卡片 ----
    n_drift = 0 if drift_df is None or drift_df.empty else len(drift_df)
    bad_rate, avg_rate = kpis["bad_rate"], kpis["avg_interest_rate"]

    with st.container(horizontal=True):
        _metric("样本量", f"{kpis['samples']:,}")
        _metric("违约率", "—" if np.isnan(bad_rate) else f"{bad_rate:.2%}",
                alert=(not np.isnan(bad_rate) and bad_rate > runtime_cfg["alert"]["bad_rate"]))
        _metric("平均利率", "—" if np.isnan(avg_rate) else f"{avg_rate:.2f}%",
                alert=(not np.isnan(avg_rate) and avg_rate > runtime_cfg["alert"]["avg_interest_rate"]))
        _metric("漂移特征数", f"{n_drift}",
                alert=n_drift > runtime_cfg["alert"]["drifted_features"])
        _metric("平均贷款额", "—" if np.isnan(kpis["avg_loan"]) else f"{kpis['avg_loan']:,.0f}")

    # ---- 趋势 + 维度分布 ----
    c_left, c_right = st.columns([2, 1])
    with c_left:
        with st.container(border=True):
            trend = mu.trend_figure(ts_df, title="实时流入趋势（模拟）" if is_live else "申请量与违约率趋势")
            if trend is not None:
                st.plotly_chart(trend, width="stretch", alt="申请量与违约率趋势图")
            else:
                st.info("当前筛选范围内没有可展示的时间数据")
    with c_right:
        with st.container(border=True):
            if pie_fig is not None:
                st.plotly_chart(pie_fig, width="stretch", alt=f"{dim_label} 分布饼图")
            else:
                st.info("实时模拟流不支持维度分布" if is_live else "没有可展示的维度数据")

    # ---- 特征漂移 ----
    st.subheader("特征漂移（相对基线）")
    base_desc = (
        "基线 = 模拟流的首个窗口；当前 = 演进到最新的窗口。"
        if is_live
        else "基线 = 全量数据分布；当前 = 时间范围内筛选出的数据。"
    )
    st.caption(
        f"{base_desc}　颜色说明：🔴 分布漂移（超阈值）｜🟠 缺失率漂移｜🔵 其它；"
        f"灰色虚线为阈值 {threshold}；柱上数字为统计量（KS/PSI，方法见括号）。"
    )

    drift_bar = mu.drift_bar_figure(
        drift_df, threshold=threshold, top_n=MONITOR_CFG["drift"]["top_n"],
        title=f"特征漂移 Top {min(n_drift, MONITOR_CFG['drift']['top_n'])}（共 {n_drift} 个）",
    )
    if drift_bar is not None:
        st.plotly_chart(drift_bar, width="stretch", alt="特征漂移 Top N 柱状图")
        st.dataframe(drift_df, width="stretch", hide_index=True, alt="特征漂移明细表")
    else:
        st.success("未检测到明显漂移", icon="✅")


monitor_panel()

st.divider()
st.caption("📈 模型监控 · 监控数据源与阈值在页面顶部配置区设置；配置项见 `configs/monitoring_config.yaml`")
