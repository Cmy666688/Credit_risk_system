"""A/B 测试框架单元测试（扩展点 5）

独立测试文件（不并入 test_basic.py）。

覆盖:
    1. 回归：指导书示例的 100 桶把分流比例精度锁死在 1%（本实现改为可配置 bucket_count）
    2. ABTestRouter：分组确定性与稳定性、分流比例与 ratio 一致、边界(0/1)、非法入参校验
    3. route：按分组调用正确模型并返回其预测概率
    4. ExperimentLogger：JSONL 曝光日志的写入/读取往返、缺失文件容错、ndarray 预测值归一化
    5. analyze_experiment：两比例 z 检验识别显著差异 / 无差异、结局缺失样本被剔除、样本不足兜底

运行: pytest tests/test_ab_test.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ab_test.router import ABTestRouter, ExperimentLogger, analyze_experiment


# ===== 测试替身 =====
class _StubModel:
    """替身模型：predict_proba 返回固定概率，并记录被调用次数。"""

    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        return np.full(len(X), self.value)


KEYS = [f"loan_{i}" for i in range(20000)]


def share_a(router: ABTestRouter, keys=KEYS) -> float:
    return sum(router.assign(k) == "A" for k in keys) / len(keys)


# ===== 回归：指导书示例的 100 桶精度问题 =====
class TestGuideRegression:
    def test_100_buckets_loses_ratio_precision(self):
        """指导书 `% 100` 的 100 桶只能表达 1% 整数倍比例：
        ratio=0.995 会被向上取整成 100%（全部走 A），默认 10000 桶则如实得到 99.5%。"""
        coarse = ABTestRouter(None, None, ratio=0.995, bucket_count=100)
        fine = ABTestRouter(None, None, ratio=0.995, bucket_count=10000)

        assert share_a(coarse) == pytest.approx(1.0)          # 精度损失：全部走 A
        assert share_a(fine) == pytest.approx(0.995, abs=0.005)


# ===== 分组 =====
class TestAssign:
    def test_same_key_is_stable(self):
        router = ABTestRouter(None, None, ratio=0.5)
        assert len({router.assign("loan_1") for _ in range(100)}) == 1

    def test_stable_across_instances(self):
        r1 = ABTestRouter(None, None, ratio=0.5)
        r2 = ABTestRouter(None, None, ratio=0.5)
        assert [r1.assign(k) for k in KEYS[:500]] == [r2.assign(k) for k in KEYS[:500]]

    def test_distribution_matches_ratio(self):
        assert share_a(ABTestRouter(None, None, ratio=0.5)) == pytest.approx(0.5, abs=0.02)
        assert share_a(ABTestRouter(None, None, ratio=0.3)) == pytest.approx(0.3, abs=0.02)

    def test_ratio_boundaries(self):
        assert share_a(ABTestRouter(None, None, ratio=1.0)) == 1.0   # 全部 A
        assert share_a(ABTestRouter(None, None, ratio=0.0)) == 0.0   # 全部 B

    def test_invalid_ratio_raises(self):
        # 指导书示例不校验：ratio 越界会静默地把流量全导向一侧，这里改为显式报错
        with pytest.raises(ValueError):
            ABTestRouter(None, None, ratio=1.5)
        with pytest.raises(ValueError):
            ABTestRouter(None, None, ratio=-0.1)

    def test_invalid_bucket_count_raises(self):
        with pytest.raises(ValueError):
            ABTestRouter(None, None, bucket_count=0)


# ===== 路由 =====
class TestRoute:
    def test_route_calls_group_model_and_returns_proba(self):
        model_a, model_b = _StubModel(0.1), _StubModel(0.9)
        router = ABTestRouter(model_a, model_b, ratio=0.5)
        features = np.zeros((1, 3))

        group, proba = router.route("loan_1", features)
        assert group == router.assign("loan_1")
        expected = 0.1 if group == "A" else 0.9
        assert proba[0] == pytest.approx(expected)
        assert (model_a.calls, model_b.calls) == (1 if group == "A" else 0,
                                                   1 if group == "B" else 0)


# ===== 曝光日志 =====
class TestExperimentLogger:
    def test_log_and_load_roundtrip(self, tmp_path):
        logger = ExperimentLogger(str(tmp_path / "events.jsonl"))
        logger.log("loan_1", "A", "v1", 0.2, outcome=0)
        logger.log("loan_2", "B", "v2", 0.7)          # outcome 先留空

        records = logger.load()
        assert len(records) == 2
        assert records[0]["key"] == "loan_1"
        assert records[0]["group"] == "A"
        assert records[0]["model_version"] == "v1"
        assert records[0]["outcome"] == 0
        assert records[1]["outcome"] is None           # 结局待补

    def test_ndarray_prediction_is_flattened(self, tmp_path):
        logger = ExperimentLogger(str(tmp_path / "events.jsonl"))
        logger.log("loan_1", "A", "v1", np.array([0.42]))
        assert logger.load()[0]["prediction"] == pytest.approx(0.42)

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert ExperimentLogger(str(tmp_path / "nope.jsonl")).load() == []

    def test_log_appends_valid_jsonl(self, tmp_path):
        path = tmp_path / "events.jsonl"
        logger = ExperimentLogger(str(path))
        logger.log("loan_1", "A", "v1", 0.2, outcome=1)
        # 每行都应是可解析的独立 JSON
        with path.open(encoding="utf-8") as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        assert len(lines) == 1


# ===== 离线分析 =====
def make_records(group: str, n: int, n_bad: int, outcome: bool = True):
    return [
        {
            "key": f"{group}_{i}",
            "group": group,
            "model_version": f"v_{group}",
            "prediction": 0.3,
            "outcome": (1 if i < n_bad else 0) if outcome else None,
        }
        for i in range(n)
    ]


class TestAnalyze:
    def test_significant_difference_detected(self):
        records = make_records("A", 1000, 100) + make_records("B", 1000, 200)
        result = analyze_experiment(records)
        assert result["A"]["n"] == 1000 and result["B"]["n"] == 1000
        assert result["A"]["bad_rate"] == pytest.approx(0.1)
        assert result["B"]["bad_rate"] == pytest.approx(0.2)
        assert result["rate_diff"] == pytest.approx(0.1)      # B - A > 0
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["better_group"] == "A"                  # A 违约率更低

    def test_no_difference_not_significant(self):
        records = make_records("A", 1000, 150) + make_records("B", 1000, 150)
        result = analyze_experiment(records)
        assert result["significant"] is False
        assert result["better_group"] is None
        assert result["p_value"] == pytest.approx(1.0, abs=1e-6)

    def test_records_without_outcome_excluded(self):
        # 尚未成熟（outcome=None）的曝光不应计入分母
        records = make_records("A", 1000, 100) + make_records("B", 1000, 200)
        records += make_records("A", 50, 50, outcome=False)   # 50 条无结局
        result = analyze_experiment(records)
        assert result["A"]["n"] == 1000

    def test_insufficient_data_returns_none_stats(self):
        records = make_records("A", 10, 1, outcome=False)      # 全部无结局
        result = analyze_experiment(records)
        assert result["A"]["n"] == 0
        assert result["z"] is None and result["p_value"] is None
        assert result["significant"] is False

    def test_logger_analyze_reads_file(self, tmp_path):
        logger = ExperimentLogger(str(tmp_path / "events.jsonl"))
        for r in make_records("A", 1000, 100) + make_records("B", 1000, 200):
            logger.log(r["key"], r["group"], r["model_version"], r["prediction"],
                       outcome=r["outcome"])
        result = logger.analyze()
        assert result["significant"] is True
        assert result["better_group"] == "A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
