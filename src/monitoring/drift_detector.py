"""
模型监控：特征漂移检测器
=========================
检测新（生产）数据相对于训练基线的分布漂移，用于模型衰减预警。

说明:
    本模块由《开发指导书》7.4「扩展点 4：模型监控」的示例代码演化而来。
    指导书示例保持为基本骨架（DriftDetector + detect + ks_2samp + threshold），
    并在其基础上修复了会导致"照抄跑不通"的问题，具体差异见文件末注释。

要点:
    1. 按列类型路由：数值列用 KS 检验，类别列用 PSI（指导书示例对
       所有列一律用 ks_2samp，遇到字符串类别列会直接抛错）
    2. 除分布漂移外，同时监控缺失率漂移与列结构漂移
    3. 数值列返回 ks_stat / p_value，便于在 Web 端展示

使用示例:
    >>> det = DriftDetector(X_train, exclude_cols=["id", "isDefault"])
    >>> drifts = det.detect(X_new, threshold=0.1)
    >>> schema = det.detect_schema(X_new)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.utils.logger import get_logger

logger = get_logger("monitoring.drift_detector")

_EPS = 1e-6  # 防止 PSI 计算出现 ln(0) / 除零


class DriftDetector:
    """基于 KS（数值特征）+ PSI（类别特征）的特征漂移检测器。

    参数:
        baseline_data: 训练基线数据（通常是训练集的特征矩阵）
        exclude_cols: 不参与监控的列，如 id、标签列
        categorical_cols: 显式声明为类别的列（含被编码成整数的类别列），
            声明后将强制走 PSI，而不会因其是数值类型被误用 KS
    """

    def __init__(
        self,
        baseline_data: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        # 指导书示例为 self.baseline = baseline_data（直接引用），
        # 这里改为 copy：避免外部 DataFrame 后续被修改时污染基线快照
        self.baseline = baseline_data.copy()
        self.exclude_cols = set(exclude_cols or [])
        self.categorical_cols = set(categorical_cols or [])

    # -------------- 公共 API --------------

    def detect(
        self,
        new_data: pd.DataFrame,
        threshold: float = 0.1,
        missing_threshold: Optional[float] = None,
    ) -> Dict[str, dict]:
        """对每一列做漂移检测，返回判定为漂移的列。

        参数:
            new_data: 新（生产）数据
            threshold: KS 统计量 / PSI 的漂移阈值（默认 0.1，与指导书一致）
            missing_threshold: 缺失率变化阈值；为 None 时复用 threshold

        返回:
            {列名: {"method", ..., "drifted": True}}，仅包含发生漂移的列
        """
        missing_threshold = threshold if missing_threshold is None else missing_threshold
        drifts: Dict[str, dict] = {}

        for col in self._monitored_columns():
            # 列缺失属于"结构漂移"，由 detect_schema 负责，
            # 此处不再像指导书那样静默 continue 掉（否则上游断供无人知晓）
            if col not in new_data.columns:
                continue
            entry = self._detect_column(col, new_data[col], threshold, missing_threshold)
            if entry is not None:
                drifts[col] = entry

        if drifts:
            logger.warning(f"检测到 {len(drifts)} 列发生漂移: {list(drifts.keys())}")
        else:
            logger.info("未检测到特征漂移")
        return drifts

    def detect_schema(self, new_data: pd.DataFrame) -> Dict[str, List[str]]:
        """检测列结构漂移。

        返回:
            missing_columns: 基线有、新数据没有（上游断供/改名，最严重的漂移）
            new_columns: 新数据新增、基线没有的列
        """
        missing = [c for c in self._monitored_columns() if c not in new_data.columns]
        added = [c for c in new_data.columns if c not in self.baseline.columns]
        if missing:
            logger.warning(f"特征列缺失（结构漂移）: {missing}")
        if added:
            logger.info(f"出现新增列: {added}")
        return {"missing_columns": missing, "new_columns": added}

    # -------------- 内部实现 --------------

    def _monitored_columns(self) -> List[str]:
        return [c for c in self.baseline.columns if c not in self.exclude_cols]

    def _detect_column(
        self,
        col: str,
        new_series: pd.Series,
        threshold: float,
        missing_threshold: float,
    ) -> Optional[dict]:
        base = self.baseline[col]

        # 指导书示例先 dropna 再比分布，会把"缺失率变化"这一关键漂移信号抹掉；
        # 这里单独计算缺失率差值并作为独立的漂移判据
        missing_delta = abs(_missing_ratio(new_series) - _missing_ratio(base))
        missing_drift = missing_delta > missing_threshold

        if self._use_ks(col, base, new_series):
            b = base.dropna().to_numpy(dtype=float)
            n = new_series.dropna().to_numpy(dtype=float)
            if len(b) == 0 or len(n) == 0:
                # 边界保护：空样本 / 全 NaN 时 ks_2samp 会返回 nan 或报警，
                # 指导书示例未处理，这里显式兜底
                stat, pvalue = float("nan"), float("nan")
            else:
                stat, pvalue = ks_2samp(b, n)
            value_drift = (not np.isnan(stat)) and float(stat) > threshold
            if value_drift or missing_drift:
                return {
                    "method": "ks",
                    "ks_stat": float(stat),
                    "p_value": float(pvalue),
                    "missing_delta": float(missing_delta),
                    "missing_drift": bool(missing_drift),
                    "drifted": True,
                }
        else:
            # 指导书示例对字符串类别列也调用 ks_2samp，会直接抛 TypeError/ValueError；
            # 类别列改为 PSI（信贷风控衡量分布稳定性的行业标准），字符串同样适用
            psi = _psi(base, new_series)
            value_drift = psi > threshold
            if value_drift or missing_drift:
                return {
                    "method": "psi",
                    "psi": float(psi),
                    "missing_delta": float(missing_delta),
                    "missing_drift": bool(missing_drift),
                    "drifted": True,
                }
        return None

    def _use_ks(self, col: str, base: pd.Series, new_series: pd.Series) -> bool:
        """是否使用 KS（数值列）；类别列（含被编码成整数的）使用 PSI。"""
        if col in self.categorical_cols:
            return False
        return _is_numeric(base) and _is_numeric(new_series)


# -------------- 模块级工具函数 --------------

def _is_numeric(series: pd.Series) -> bool:
    """数值且非布尔（布尔视为类别）。"""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _missing_ratio(series: pd.Series) -> float:
    return float(series.isna().mean())


def _psi(base: pd.Series, new: pd.Series) -> float:
    """Population Stability Index：衡量类别分布的稳定性（越大越不稳定）。"""
    base_dist = base.dropna().astype(str).value_counts(normalize=True)
    new_dist = new.dropna().astype(str).value_counts(normalize=True)
    psi = 0.0
    for category in base_dist.index.union(new_dist.index):
        pb = float(base_dist.get(category, 0.0)) or _EPS
        pn = float(new_dist.get(category, 0.0)) or _EPS
        psi += (pn - pb) * np.log(pn / pb)
    return float(psi)


# ----------------------------------------------------------------------
# 与指导书 7.4 示例代码的差异说明（每处都对应一个"照抄会出问题"的点）：
#   1. self.baseline = baseline_data.copy()  —— 原示例直接引用，基线会被外部修改污染
#   2. 新增 exclude_cols —— 原示例遍历 baseline.columns，会把 id / isDefault 一并监控
#   3. 按列类型路由（数值 KS / 类别 PSI）—— 原示例对字符串类别列调用 ks_2samp 会报错
#   4. 新增 categorical_cols —— 类别列若已被编码成整数，需显式声明才能走 PSI
#   5. 单独统计 missing_delta —— 原示例 dropna 后无法察觉缺失率漂移
#   6. 新增 detect_schema —— 原示例对缺失列 `continue` 静默忽略，漏掉最严重的结构漂移
#   7. 空列 / 全 NaN 的边界兜底 —— 原示例会返回 nan 或触发 scipy 报警
#   8. 使用项目统一 logger —— 原示例无日志，不符合项目约定
# ----------------------------------------------------------------------
