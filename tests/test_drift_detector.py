"""特征漂移检测单元测试（扩展点 4：模型监控）

独立测试文件（不并入 test_basic.py）。

覆盖:
    1. 回归：指导书示例直接用 ks_2samp 处理字符串类别列会抛错（本实现改用 PSI）
    2. 数值特征：KS 检验识别漂移 / 无漂移，threshold 阈值生效
    3. 类别特征：字符串列不再报错，分布漂移能被 PSI 检出
    4. 缺失率漂移：单独统计 missing_delta（原示例 dropna 会漏掉）
    5. 列结构漂移：特征列消失 / 新增列
    6. exclude_cols 排除 id / isDefault；categorical_cols 对整数编码类别列强制 PSI
    7. 边界：全 NaN 列 / 常量列不崩溃

运行: pytest tests/test_drift_detector.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_detector import DriftDetector


def make_baseline(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "loanAmnt": rng.normal(10000, 1000, n),          # 数值
        "grade": rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]),  # 字符串类别
        "homeOwnership": rng.choice([0, 1, 2], n),        # 编码成整数的类别
        "annualIncome": rng.normal(50000, 5000, n),       # 数值
        "isDefault": rng.choice([0, 1], n, p=[0.8, 0.2]),  # 标签列
    })


@pytest.fixture(scope="module")
def baseline() -> pd.DataFrame:
    return make_baseline()


def detector(baseline: pd.DataFrame, **kwargs) -> DriftDetector:
    return DriftDetector(baseline, exclude_cols=["id", "isDefault"], **kwargs)


# ===== 回归：指导书示例的字符串列问题 =====
class TestGuideRegression:
    def test_ks_2samp_raises_on_string_columns(self, baseline):
        """指导书示例对字符串类别列调用 ks_2samp 会抛 TypeError —— 本实现用 PSI 规避"""
        from scipy.stats import ks_2samp
        with pytest.raises(Exception):
            ks_2samp(baseline["grade"].dropna(), baseline["grade"].dropna())

    def test_detector_does_not_crash_on_string_columns(self, baseline):
        result = detector(baseline).detect(baseline.copy())
        assert isinstance(result, dict)


# ===== 数值特征：KS =====
class TestNumericDrift:
    def test_no_drift_when_identical(self, baseline):
        assert detector(baseline).detect(baseline.copy()) == {}

    def test_ks_drift_detected(self, baseline):
        new = baseline.copy()
        new["loanAmnt"] = new["loanAmnt"] + 300          # 均值整体偏移
        result = detector(baseline).detect(new)
        assert "loanAmnt" in result
        assert result["loanAmnt"]["method"] == "ks"
        assert result["loanAmnt"]["ks_stat"] > 0.1
        assert 0.0 <= result["loanAmnt"]["p_value"] <= 1.0  # 保留指导书的 p_value 字段

    def test_threshold_is_respected(self, baseline):
        new = baseline.copy()
        new["loanAmnt"] = new["loanAmnt"] + 300
        det = detector(baseline)
        assert "loanAmnt" not in det.detect(new, threshold=0.99)   # 阈值过高 -> 不报
        assert "loanAmnt" in det.detect(new, threshold=0.005)      # 阈值放宽 -> 报


# ===== 类别特征：PSI =====
class TestCategoricalDrift:
    def test_psi_detects_distribution_shift(self, baseline):
        new = baseline.copy()
        rng = np.random.default_rng(1)
        new["grade"] = rng.choice(["A", "B", "C"], len(new), p=[0.1, 0.3, 0.6])  # 与基线对调
        result = detector(baseline).detect(new)
        assert result["grade"]["method"] == "psi"
        assert result["grade"]["psi"] > 0.1

    def test_no_psi_drift_when_same_distribution(self, baseline):
        rng = np.random.default_rng(2)
        new = baseline.copy()
        # 用同样的分布重新抽样，PSI 应很小
        new["grade"] = rng.choice(["A", "B", "C"], len(new), p=[0.6, 0.3, 0.1])
        assert "grade" not in detector(baseline).detect(new)

    def test_categorical_cols_forces_psi_for_int_codes(self, baseline):
        """整数编码的类别列若显式声明，应走 PSI 而非 KS"""
        rng = np.random.default_rng(3)
        new = baseline.copy()
        new["homeOwnership"] = rng.choice([0, 1, 2], len(new), p=[0.1, 0.3, 0.6])
        det = detector(baseline, categorical_cols=["homeOwnership"])
        result = det.detect(new)
        assert result["homeOwnership"]["method"] == "psi"


# ===== 缺失率漂移 =====
class TestMissingRateDrift:
    def test_missing_rate_drift_captured(self, baseline):
        new = baseline.copy()
        new.loc[new.index[:600], "annualIncome"] = np.nan   # 30% 变为缺失
        result = detector(baseline).detect(new)
        assert "annualIncome" in result
        assert result["annualIncome"]["missing_drift"] is True
        assert result["annualIncome"]["missing_delta"] > 0.1

    def test_same_missing_rate_is_not_drift(self, baseline):
        new = baseline.copy()
        result = detector(baseline).detect(new)
        assert result == {}


# ===== 列结构漂移 =====
class TestSchemaDrift:
    def test_missing_column_detected(self, baseline):
        new = baseline.drop(columns=["loanAmnt"])
        schema = detector(baseline).detect_schema(new)
        assert "loanAmnt" in schema["missing_columns"]

    def test_new_column_detected(self, baseline):
        new = baseline.copy()
        new["brand_new_feature"] = 1
        schema = detector(baseline).detect_schema(new)
        assert "brand_new_feature" in schema["new_columns"]

    def test_excluded_columns_not_reported_as_missing(self, baseline):
        """id / isDefault 被排除，不应出现在结构漂移里"""
        new = baseline.drop(columns=["id", "isDefault"])
        schema = detector(baseline).detect_schema(new)
        assert "id" not in schema["missing_columns"]
        assert "isDefault" not in schema["missing_columns"]


# ===== 排除列 =====
class TestExcludeColumns:
    def test_id_and_target_excluded(self, baseline):
        new = baseline.copy()
        new["id"] = new["id"] + 999999          # id 剧变
        new["isDefault"] = 1                    # 标签全变
        result = detector(baseline).detect(new)
        assert "id" not in result
        assert "isDefault" not in result

    def test_id_drift_would_be_flagged_if_not_excluded(self, baseline):
        new = baseline.copy()
        new["id"] = new["id"] + 999999
        result = DriftDetector(baseline).detect(new)   # 不排除
        assert "id" in result


# ===== 边界情况 =====
class TestEdgeCases:
    def test_all_nan_column_does_not_crash(self, baseline):
        # 基线和新数据该列都全为空 -> KS 返回 nan，不得崩溃，也不得误报为分布漂移
        base = baseline.copy()
        base["annualIncome"] = np.nan
        new = base.copy()
        result = detector(base).detect(new)
        assert result.get("annualIncome") is None

    def test_column_fully_missing_is_flagged(self, baseline):
        # 反过来：新数据该列 100% 缺失，是严重漂移，应当被抓到
        new = baseline.copy()
        new["annualIncome"] = np.nan
        assert detector(baseline).detect(new)["annualIncome"]["missing_drift"] is True

    def test_constant_column_not_flagged(self, baseline):
        new = baseline.copy()
        new["const_feat"] = 1.0
        base = baseline.copy()
        base["const_feat"] = 1.0
        assert "const_feat" not in detector(base).detect(new)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
