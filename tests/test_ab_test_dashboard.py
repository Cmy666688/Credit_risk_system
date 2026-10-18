"""A/B 测试仪表盘工具单元测试（扩展点 5 的 Web 支撑层）

独立测试文件（不并入 test_basic.py）。

覆盖:
    1. load_ab_config：默认值兜底 + configs/ab_test_config.yaml 覆盖
    2. run_experiment：曝光记录字段、分组预测、ratio 控制组规模、无结局样本、显著性识别
    3. bad_rate_figure：返回 plotly Figure 且标注 p 值

运行: pytest tests/test_ab_test_dashboard.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.components import ab_test_utils as abu
from src.ab_test.router import ABTestRouter


class _StubModel:
    """替身模型：predict_proba 返回固定概率。"""

    def __init__(self, value: float):
        self.value = value

    def predict_proba(self, X):
        return np.full(len(X), self.value)


def make_inputs(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    keys = np.arange(n)
    outcomes = rng.integers(0, 2, n)
    return X, keys, outcomes


class TestLoadConfig:
    def test_defaults_when_file_missing(self, tmp_path):
        # tmp_path 下没有 configs/ab_test_config.yaml -> 回退到代码默认值
        cfg = abu.load_ab_config(tmp_path)
        assert cfg["ratio"] == 0.5
        assert cfg["bucket_count"] == 10000

    def test_yaml_overrides_defaults(self):
        # 项目自带的 configs/ab_test_config.yaml
        cfg = abu.load_ab_config(ROOT)
        assert cfg["alpha"] == 0.05
        assert cfg["group_a_model"] == "xgboost"
        assert cfg["group_b_model"] == "lightgbm"
        assert cfg["sample_size"] == 5000


class TestRunExperiment:
    def test_records_shape_and_fields(self):
        X, keys, outcomes = make_inputs(1000)
        records, _ = abu.run_experiment(_StubModel(0.2), _StubModel(0.7), X, keys, outcomes)
        assert len(records) == 1000
        assert all(r["group"] in {"A", "B"} for r in records)
        # 预测值来自对应分组的模型
        assert all(
            r["prediction"] == (0.2 if r["group"] == "A" else 0.7) for r in records
        )

    def test_result_consistent_with_router_analyze(self):
        from src.ab_test.router import analyze_experiment

        X, keys, outcomes = make_inputs(1000)
        records, result = abu.run_experiment(_StubModel(0.3), _StubModel(0.3), X, keys, outcomes)
        assert result == analyze_experiment(records)

    def test_ratio_controls_group_size(self):
        X, keys, outcomes = make_inputs(5000)
        records, _ = abu.run_experiment(
            _StubModel(0.3), _StubModel(0.3), X, keys, outcomes, ratio=0.3
        )
        share_a = sum(r["group"] == "A" for r in records) / len(records)
        assert share_a == pytest.approx(0.3, abs=0.03)

    def test_no_outcome_records_keep_none(self):
        X, keys, _ = make_inputs(500)
        records, result = abu.run_experiment(
            _StubModel(0.3), _StubModel(0.3), X, keys, [None] * 500
        )
        assert all(r["outcome"] is None for r in records)
        assert result["significant"] is False
        assert result["A"]["n"] == 0 and result["p_value"] is None

    def test_significant_difference_and_better_group(self):
        X, keys, _ = make_inputs(3000, seed=1)
        router = ABTestRouter(None, None, ratio=0.5)
        groups = np.array([router.assign(k) for k in keys])
        # 按分组构造结局：A 组违约率 ~10%，B 组 ~30%
        outcomes = np.zeros(len(keys), dtype=int)
        for group, rate in (("A", 0.10), ("B", 0.30)):
            idx = np.where(groups == group)[0]
            outcomes[idx[: int(len(idx) * rate)]] = 1

        _, result = abu.run_experiment(_StubModel(0.3), _StubModel(0.3), X, keys, outcomes)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["better_group"] == "A"   # A 组违约率更低


class TestBadRateFigure:
    def test_returns_plotly_figure_with_pvalue(self):
        X, keys, outcomes = make_inputs(1000)
        _, result = abu.run_experiment(_StubModel(0.3), _StubModel(0.3), X, keys, outcomes)
        fig = abu.bad_rate_figure(result, alpha=0.05)
        assert isinstance(fig, go.Figure)
        assert "p=" in fig.layout.title.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
