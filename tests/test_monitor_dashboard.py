"""
模型监控仪表盘测试（扩展点 4 · 模型监控）
==========================================
独立测试文件（不并入 test_basic.py）。

分两部分:
    1. 纯逻辑测试 —— app/components/monitor_utils.py 中的函数
       （数据筛选、KPI、漂移、告警、图表）
    2. 页面测试 —— 用 streamlit.testing.v1.AppTest 真实运行
       app/pages/8_📈_模型监控.py，验证数据加载 / 图表渲染 / 交互操作

运行: pytest tests/test_monitor_dashboard.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.components import monitor_utils as mu

PAGE_PATH = ROOT / "app" / "views" / "8_📈_模型监控.py"


def _baseline(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "loanAmnt": rng.normal(10000, 1000, n),
        "grade": rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),
        "interestRate": rng.normal(12, 1.5, n),
        "isDefault": rng.choice([0, 1], n, p=[0.8, 0.2]),
        "issueDate": pd.date_range("2015-01-01", periods=n, freq="D").astype(str),
    })


@pytest.fixture(scope="module")
def config() -> dict:
    return mu.load_monitor_config(ROOT)


@pytest.fixture(scope="module")
def drift_kwargs() -> dict:
    return dict(exclude_cols=["id", "isDefault"], categorical_cols=["grade"], date_col="issueDate")


# ==================== 配置 ====================

class TestConfig:
    def test_defaults_loaded_from_repo_config(self, config):
        assert config["drift"]["threshold"] > 0
        assert "alert" in config and "dimensions" in config
        assert "grade" in config["dimensions"]

    def test_config_file_override_merges(self, tmp_path):
        cfg_file = tmp_path / "monitoring_config.yaml"
        cfg_file.write_text("alert:\n  bad_rate: 0.5\n", encoding="utf-8")
        cfg = mu.load_monitor_config(ROOT, config_path=cfg_file)
        assert cfg["alert"]["bad_rate"] == 0.5
        # 未覆盖的默认值仍在
        assert cfg["alert"]["avg_interest_rate"] == mu.DEFAULT_MONITOR_CONFIG["alert"]["avg_interest_rate"]

    def test_defaults_when_no_file(self, tmp_path):
        cfg = mu.load_monitor_config(ROOT, config_path=tmp_path / "nope.yaml")
        assert cfg["drift"]["threshold"] == mu.DEFAULT_MONITOR_CONFIG["drift"]["threshold"]


# ==================== 数据源 / 加载 ====================

class TestSources:
    def test_live_source_always_available(self):
        sources = mu.available_sources(ROOT)
        live = next(s for s in sources if s["key"] == "live")
        assert live["available"] is True

    def test_sample_data_loadable(self):
        path = ROOT / "data" / "sample" / "sample.csv"
        assert path.exists(), "合成示例数据应存在"
        df = pd.read_csv(path, nrows=500)
        assert {"loanAmnt", "grade", "isDefault", "issueDate"}.issubset(df.columns)


# ==================== 筛选 / 聚合 / KPI ====================

class TestFilterAggregateKpi:
    def test_filter_by_time_range_reduces_rows(self):
        df = _baseline()
        out = mu.filter_by_time_range(df, "issueDate", "2015-01-01", "2015-03-01")
        assert 0 < len(out) < len(df)

    def test_filter_unparsable_date_returns_unchanged(self):
        df = pd.DataFrame({"issueDate": ["not-a-date"] * 5})
        assert len(mu.filter_by_time_range(df, "issueDate", "2015-01-01", "2016-01-01")) == 5

    def test_filter_without_bounds_returns_unchanged(self):
        df = _baseline()
        assert len(mu.filter_by_time_range(df, "issueDate")) == len(df)

    def test_aggregate_time_series(self):
        ts = mu.aggregate_time_series(_baseline())
        assert list(ts.columns) == ["时间", "申请量", "违约率", "平均利率"]
        assert not ts.empty
        assert ts["时间"].is_monotonic_increasing

    def test_compute_kpis_values(self):
        df = pd.DataFrame({"isDefault": [0, 1, 1, 0], "interestRate": [10.0, 20.0, 30.0, 40.0],
                           "loanAmnt": [100.0, 200.0, 300.0, 400.0]})
        kpis = mu.compute_kpis(df)
        assert kpis["samples"] == 4
        assert kpis["bad_rate"] == 0.5
        assert kpis["avg_interest_rate"] == 25.0
        assert kpis["avg_loan"] == 250.0

    def test_kpis_from_live(self):
        live = mu.make_live_stream(window=10, tick=0)
        kpis = mu.kpis_from_live(live)
        assert kpis["samples"] > 0
        assert 0 <= kpis["bad_rate"] <= 1


# ==================== 漂移 ====================

class TestDrift:
    def test_detects_numeric_and_categorical(self, config, drift_kwargs):
        base = _baseline()
        cur = base.copy()
        cur["loanAmnt"] = cur["loanAmnt"] + 600
        rng = np.random.default_rng(9)
        cur["grade"] = rng.choice(["A", "B", "C"], len(cur), p=[0.1, 0.2, 0.7])

        drift = mu.compute_drift(base, cur, config, **drift_kwargs)
        assert not drift.empty
        assert {"特征", "方法", "统计量", "缺失率变化", "缺失漂移"}.issubset(drift.columns)
        assert drift["统计量"].is_monotonic_decreasing
        assert drift.loc[drift["特征"] == "loanAmnt", "方法"].iloc[0] == "KS"
        assert drift.loc[drift["特征"] == "grade", "方法"].iloc[0] == "PSI"

    def test_no_drift_when_identical(self, config, drift_kwargs):
        base = _baseline()
        assert mu.compute_drift(base, base.copy(), config, **drift_kwargs).empty

    def test_excludes_id_target_and_date(self, config, drift_kwargs):
        base = _baseline()
        cur = base.copy()
        cur["loanAmnt"] = cur["loanAmnt"] + 600
        cur["id"] = cur["id"] + 10 ** 6
        cur["isDefault"] = 1
        drift = mu.compute_drift(base, cur, config, **drift_kwargs)
        assert "id" not in set(drift["特征"])
        assert "isDefault" not in set(drift["特征"])
        assert "issueDate" not in set(drift["特征"])

    def test_missing_rate_drift_captured(self, config, drift_kwargs):
        base = _baseline()
        cur = base.copy()
        cur.loc[cur.index[:600], "loanAmnt"] = np.nan
        drift = mu.compute_drift(base, cur, config, **drift_kwargs)
        assert "loanAmnt" in set(drift["特征"])
        row = drift[drift["特征"] == "loanAmnt"].iloc[0]
        assert bool(row["缺失漂移"]) and row["缺失率变化"] > 0.1

    def test_empty_inputs_return_empty_frame(self, config):
        drift = mu.compute_drift(pd.DataFrame(), pd.DataFrame(), config)
        assert drift.empty and list(drift.columns) == ["特征", "方法", "统计量", "缺失率变化", "缺失漂移"]


# ==================== 告警 ====================

class TestAlerts:
    def test_high_bad_rate_triggers_error(self, config):
        alerts = mu.evaluate_alerts({"bad_rate": 0.9, "avg_interest_rate": 10.0}, None, config)
        assert any(a["level"] == "error" and a["metric"] == "bad_rate" for a in alerts)

    def test_high_interest_triggers_warning(self, config):
        alerts = mu.evaluate_alerts({"bad_rate": 0.1, "avg_interest_rate": 99.0}, None, config)
        assert any(a["metric"] == "avg_interest_rate" for a in alerts)

    def test_many_drifted_features_triggers_warning(self, config):
        drift = pd.DataFrame({"特征": [f"f{i}" for i in range(20)], "方法": ["KS"] * 20,
                              "统计量": [0.2] * 20, "缺失率变化": [0.0] * 20, "缺失漂移": [False] * 20})
        alerts = mu.evaluate_alerts({"bad_rate": 0.1, "avg_interest_rate": 10.0}, drift, config)
        assert any(a["metric"] == "drifted_features" for a in alerts)

    def test_no_alerts_when_normal(self, config):
        assert mu.evaluate_alerts({"bad_rate": 0.1, "avg_interest_rate": 10.0}, None, config) == []


# ==================== 图表 ====================

class TestFigures:
    def test_trend_figure_has_bar_and_scatter(self):
        fig = mu.trend_figure(mu.aggregate_time_series(_baseline()))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert {t.type for t in fig.data} == {"bar", "scatter"}

    def test_trend_figure_none_on_empty(self):
        assert mu.trend_figure(pd.DataFrame()) is None

    def test_dimension_pie_figure(self):
        fig = mu.dimension_pie_figure(_baseline(), "grade", "贷款等级")
        assert isinstance(fig, go.Figure) and fig.data[0].type == "pie"

    def test_dimension_pie_none_when_col_absent(self):
        assert mu.dimension_pie_figure(_baseline(), "not_exist") is None

    def test_drift_bar_figure_with_threshold_line(self, config, drift_kwargs):
        cur = _baseline()
        cur["loanAmnt"] = cur["loanAmnt"] + 600
        drift = mu.compute_drift(_baseline(), cur, config, **drift_kwargs)
        fig = mu.drift_bar_figure(drift, threshold=0.1)
        assert isinstance(fig, go.Figure)
        assert fig.data[0].orientation == "h"
        assert fig.layout.shapes, "应包含阈值参考线"
        # 柱上标注「统计量 (方法)」，且每根柱子都有
        texts = list(fig.data[0].text)
        assert len(texts) == len(drift)
        assert all("(" in t and ")" in t for t in texts)

    def test_drift_bar_colors_and_labels_by_type(self):
        """红=分布漂移；橙=缺失率漂移；柱上标注统计量与方法；高度随条目自适应"""
        drift = pd.DataFrame({
            "特征": ["缺失率漂移列", "分布漂移列"],
            "方法": ["PSI", "KS"],
            "统计量": [0.02, 0.30],            # 后者超阈值
            "缺失率变化": [0.40, 0.0],
            "缺失漂移": [True, False],
        })
        fig = mu.drift_bar_figure(drift, threshold=0.1)
        # 排序为升序（小→大），故橙色在前、红色在后
        assert list(fig.data[0].marker.color) == ["#F2B134", "#E45756"]
        assert list(fig.data[0].text) == ["0.020 (PSI)", "0.300 (KS)"]
        # 高度自适应：2 个条目应明显矮于满屏，且不小于最小值
        assert 220 <= fig.layout.height <= 700

    def test_drift_bar_height_adapts_to_count(self):
        def _mk(n):
            return pd.DataFrame({
                "特征": [f"f{i}" for i in range(n)], "方法": ["KS"] * n,
                "统计量": [0.2] * n, "缺失率变化": [0.0] * n, "缺失漂移": [False] * n,
            })
        h1 = mu.drift_bar_figure(_mk(1)).layout.height
        h15 = mu.drift_bar_figure(_mk(15)).layout.height
        assert h1 < h15, "条目越多图表应越高"
        assert h1 <= 300, "单条目时不应拉成全屏实心矩形"

    def test_drift_bar_none_on_empty(self):
        assert mu.drift_bar_figure(pd.DataFrame()) is None


# ==================== 实时模拟流 ====================

class TestLiveStream:
    def test_shape_and_columns(self):
        live = mu.make_live_stream(window=30, tick=0)
        assert list(live.columns) == ["时间", "申请量", "违约率", "平均利率"]
        assert len(live) == 30

    def test_reproducible_for_same_tick(self):
        a = mu.make_live_stream(window=30, tick=5, seed=1)
        b = mu.make_live_stream(window=30, tick=5, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_changes_across_ticks(self):
        a = mu.make_live_stream(window=30, tick=0, seed=1)
        b = mu.make_live_stream(window=30, tick=50, seed=1)
        assert not a["申请量"].equals(b["申请量"])


# ==================== 页面（AppTest）====================

@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PAGE_PATH))
    # 监控页有管理员门禁（require_admin），注入管理员会话后再运行
    at.session_state["user"] = {"id": 1, "username": "admin", "role": "admin"}
    return at


def test_page_renders_without_exception(app):
    """数据加载 + 图表渲染：页面能正常跑完且无异常"""
    app.run(timeout=90)
    assert not app.exception
    # 至少渲染了 KPI 指标卡片
    assert len(app.metric) >= 3
    labels = [m.label for m in app.metric]
    assert any("违约率" in l for l in labels)


def test_page_config_controls_moved_to_main_area(app):
    """配置控件已从侧边栏迁移到主内容区，侧边栏不再残留相关控件"""
    app.run(timeout=90)
    assert not app.exception
    # 主内容区存在配置控件：数据源 / 展示维度 / 刷新按钮 / 两个阈值
    assert len(app.segmented_control) >= 1      # 数据源
    assert len(app.selectbox) >= 1              # 展示维度
    assert any("刷新" in b.label for b in app.button), "应有「刷新数据」按钮"
    assert len(app.number_input) >= 2           # 漂移阈值 + 违约率告警阈值
    assert app.selectbox[0].value in ("grade", "purpose", "homeOwnership", "verificationStatus")
    # 侧边栏已清理：不再包含这些控件（用户信息区的「退出登录」按钮除外）
    assert len(app.sidebar.selectbox) == 0
    assert len(app.sidebar.number_input) == 0
    sidebar_btns = [b.label for b in app.sidebar.button]
    assert all("刷新" not in l for l in sidebar_btns), "侧边栏不应残留「刷新」按钮"


def test_page_interaction_refresh_button(app):
    """交互操作：点击「🔄 刷新数据」后页面正常重跑（手动刷新，无自动刷新）"""
    app.run(timeout=90)
    refresh_btn = next(b for b in app.button if "刷新" in b.label)
    refresh_btn.click().run(timeout=90)
    assert not app.exception
    assert len(app.metric) >= 3


def test_page_interaction_switch_dimension(app):
    """交互操作：切换展示维度后页面仍正常"""
    app.run(timeout=90)
    app.selectbox[0].set_value("purpose").run(timeout=90)
    assert not app.exception
    assert app.selectbox[0].value == "purpose"


def test_page_interaction_narrow_time_range(app):
    """交互操作：缩小时间范围（应触发漂移检测）后页面仍正常"""
    app.run(timeout=90)
    if app.date_input:
        app.date_input[0].set_value(("2015-01-01", "2015-06-01")).run(timeout=90)
    assert not app.exception


# ==================== 真实数据端到端 ====================

class TestRealDataIntegration:
    """在真实合成数据（sample.csv）上跑通「加载 → 筛选 → 漂移 → 图表」全链路。"""

    def test_full_pipeline_on_sample_csv(self, config):
        from src.utils.config_loader import ConfigLoader
        data_cfg = ConfigLoader(str(ROOT / "configs")).get("data")

        df = pd.read_csv(ROOT / "data" / "sample" / "sample.csv")
        dt = pd.to_datetime(df["issueDate"], errors="coerce")
        window_start = dt.min()
        window_end = dt.min() + pd.Timedelta(days=120)
        cur = mu.filter_by_time_range(df, "issueDate", window_start, window_end)
        assert 0 < len(cur) < len(df)

        kpis = mu.compute_kpis(cur, data_cfg.target_column)
        assert kpis["samples"] == len(cur)
        assert 0 <= kpis["bad_rate"] <= 1

        drift = mu.compute_drift(
            df, cur, config,
            exclude_cols=[data_cfg.id_column, data_cfg.target_column],
            categorical_cols=list(data_cfg.columns.categorical),
            date_col="issueDate",
        )
        assert not drift.empty, "缩小时间窗口后应检测出漂移特征"
        assert "id" not in set(drift["特征"])

        # 图表可正常渲染
        assert mu.trend_figure(mu.aggregate_time_series(cur)) is not None
        assert mu.dimension_pie_figure(cur, "grade", "贷款等级") is not None
        assert mu.drift_bar_figure(drift, threshold=config["drift"]["threshold"]) is not None
        # 告警判定可执行
        assert isinstance(mu.evaluate_alerts(kpis, drift, config), list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
