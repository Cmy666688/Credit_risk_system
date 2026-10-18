"""A/B 测试框架：模型分流与效果对比（扩展点 5）
================================================
在线对照实验的最小可用实现，对应指导书 7.5：

    1. ABTestRouter     —— 用确定性哈希把同一个标识稳定分到 A/B 组，分别调用两个模型
    2. ExperimentLogger —— 落地曝光日志：(标识, 分组, 模型版本, 预测, 结局)
       （即指导书 🚀 里 (user_id, model_version, prediction, eventual_outcome) 的可落地实现）
    3. analyze_experiment —— 离线对比两组的违约率，并用两比例 z 检验判断差异是否显著

相对指导书示例代码的修改（原因见下方各处注释）：
    1. 术语：示例注释把 MD5 取模称为"一致性哈希"，实际是确定性哈希分桶，
       一致性哈希（consistent hashing）是分布式哈希环技术，二者用途完全不同。
    2. 分桶数：示例的 `hash_val % 100` 把分流比例精度锁死在 1%，
       这里改为可配置的 bucket_count（默认 10000，精度 0.01%）。
    3. 入参：示例的 `user_id` 在本项目数据里并不存在（data_config.yaml 的唯一标识列是
       `id`），改为通用的 `key`，由调用方传入实际的申请编号。
    4. 校验：示例未校验 ratio，越界时会静默地把全部流量导向一侧，这里显式报错。
    5. 补齐：示例只给了一个分流器。这里补上 🚀 提到的曝光日志与离线对比，
       否则"框架"名不副实（缺埋点、缺结果分析、无显著性判断）。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from src.utils.logger import get_logger

logger = get_logger("ab_test.router")


class ABTestRouter:
    """确定性哈希分桶的 A/B 分流器。

    参数:
        model_a: A 组（对照组）模型，需实现 ``predict_proba`` 并返回正类概率
        model_b: B 组（实验组）模型
        ratio: 分流到 A 组的比例，取值 [0, 1]
        bucket_count: 哈希分桶数量，决定 ratio 能达到的精度（默认 10000 -> 0.01%）
    """

    def __init__(
        self,
        model_a,
        model_b,
        ratio: float = 0.5,
        bucket_count: int = 10000,
    ):
        # 指导书示例未做校验：ratio 越界（如 1.5 / -0.1）不会报错，
        # 而是让所有流量静默落到同一侧，导致实验组/对照组配置被悄悄架空
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio 必须在 [0, 1] 区间内，当前为 {ratio}")
        if bucket_count <= 0:
            raise ValueError(f"bucket_count 必须为正整数，当前为 {bucket_count}")

        self.model_a = model_a
        self.model_b = model_b
        self.ratio = ratio
        self.bucket_count = bucket_count

    def assign(self, key: Any) -> str:
        """只做分组、不预测：同一个 key 恒定返回 ``"A"`` 或 ``"B"``。

        指导书示例注释称为"一致性哈希"，这里纠正为"确定性哈希分桶"：
        用 MD5（跨进程稳定，不受 PYTHONHASHSEED 影响）把 key 映射到固定桶。
        """
        digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        # 指导书示例 `% 100 / 100.0`（100 桶）只支持 1% 的 ratio 精度；
        # 这里用可配置的 bucket_count，默认 10000 桶，精度提升到 0.01%
        bucket = (int(digest, 16) % self.bucket_count) / self.bucket_count
        return "A" if bucket < self.ratio else "B"

    def route(self, key: Any, features: Any) -> Tuple[str, np.ndarray]:
        """按 key 选组并调用对应模型，返回 ``(组名, 预测概率)``。"""
        group = self.assign(key)
        model = self.model_a if group == "A" else self.model_b
        return group, model.predict_proba(features)


class ExperimentLogger:
    """把每次曝光的 ``(key, group, model_version, prediction, outcome)`` 追加成 JSONL。

    指导书 🚀 只给了思路，这里给出可落地的实现；逐行 JSON 追加，
    离线分析可直接 ``pd.read_json(path, lines=True)`` 读取。
    """

    def __init__(self, path: str = "data/ab_test/events.jsonl"):
        self.path = Path(path)

    def log(
        self,
        key: Any,
        group: str,
        model_version: str,
        prediction: Any,
        outcome: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """记录一次曝光。``outcome`` 可先留空，待违约标签成熟后再补写。

        返回写入的完整记录，便于调用方直接断言 / 复用。
        """
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "group": group,
            "model_version": model_version,
            # 模型可能返回 (1,) 形状的 ndarray，统一取标量再存
            "prediction": float(np.ravel(prediction)[0]),
            "outcome": None if outcome is None else int(outcome),
        }
        record.update(extra)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def load(self) -> List[Dict[str, Any]]:
        """读取全部曝光记录；文件不存在时返回空列表。"""
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def analyze(self, outcome_field: str = "outcome", alpha: float = 0.05) -> Dict[str, Any]:
        """读取本文件并做离线 A/B 对比。"""
        return analyze_experiment(self.load(), outcome_field=outcome_field, alpha=alpha)


def _group_stats(
    records: List[Dict[str, Any]], group: str, outcome_field: str
) -> Dict[str, float]:
    """统计某组的样本量、违约数与违约率、平均预测概率。"""
    outcomes = [
        int(r[outcome_field])
        for r in records
        if r.get("group") == group and r.get(outcome_field) is not None
    ]
    preds = [
        float(r["prediction"])
        for r in records
        if r.get("group") == group and r.get("prediction") is not None
    ]
    n = len(outcomes)
    positives = sum(1 for v in outcomes if v == 1)
    return {
        "n": n,
        "positives": positives,
        "bad_rate": (positives / n) if n else float("nan"),
        "mean_pred": (sum(preds) / len(preds)) if preds else float("nan"),
    }


def analyze_experiment(
    records: List[Dict[str, Any]],
    outcome_field: str = "outcome",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """离线对比 A 组与 B 组的违约率（正类即违约），并做两比例 z 检验。

    信贷场景中"效果"以**违约率越低越好**衡量，故 ``rate_diff = B - A``：
    为负说明实验组违约率更低（更好）。

    参数:
        records: 曝光记录列表（通常来自 :class:`ExperimentLogger`）
        outcome_field: 结局字段名，指导书里叫 ``eventual_outcome``
        alpha: 显著性水平，默认 0.05

    返回:
        {"A": {...}, "B": {...}, "rate_diff", "z", "p_value", "significant",
         "alpha", "better_group"}
    """
    stats_a = _group_stats(records, "A", outcome_field)
    stats_b = _group_stats(records, "B", outcome_field)
    rate_a, rate_b = stats_a["bad_rate"], stats_b["bad_rate"]
    n_a, n_b = stats_a["n"], stats_b["n"]

    # 两比例 z 检验：样本量不足时无法判断显著性
    z: Optional[float] = None
    p_value: Optional[float] = None
    if n_a > 0 and n_b > 0:
        p_pool = (stats_a["positives"] + stats_b["positives"]) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        if se > 0:
            z = float((rate_b - rate_a) / se)
            p_value = float(2 * (1 - norm.cdf(abs(z))))

    significant = bool(p_value is not None and p_value < alpha)
    better_group: Optional[str] = None
    if significant and not np.isnan(rate_a) and not np.isnan(rate_b):
        # 违约率低者为优
        better_group = "A" if rate_a < rate_b else "B"

    result = {
        "A": stats_a,
        "B": stats_b,
        "rate_diff": (rate_b - rate_a) if (n_a > 0 and n_b > 0) else float("nan"),
        "z": z,
        "p_value": p_value,
        "significant": significant,
        "alpha": alpha,
        "better_group": better_group,
    }
    logger.info(
        "A/B 分析: A(n={n_a}, bad={ra:.4f}) vs B(n={n_b}, bad={rb:.4f}), "
        "p={p}, 显著={sig}".format(
            n_a=n_a,
            ra=rate_a,
            n_b=n_b,
            rb=rate_b,
            p="N/A" if p_value is None else f"{p_value:.4f}",
            sig=significant,
        )
    )
    return result
