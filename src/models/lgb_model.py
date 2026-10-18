"""
LightGBM 模型封装
=================
对 LGBMClassifier 进行配置化封装，统一接口。

要点:
    与 XGBModel 遵循同一 BaseModel 接口，可无缝替换、对比或接入 Ensemble。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger("models.lgb")


class LGBModel(BaseModel):
    """
    LightGBM 二分类器封装。

    使用示例:
        >>> model = LGBModel(model_config.lightgbm)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> proba = model.predict_proba(X_test)   # shape=(n,)
    """

    def __init__(self, config, name: str = "lightgbm"):
        super().__init__(config, name)
        self.params = dict(config.params)  # 转为普通 dict
        self.early_stopping_cfg = getattr(config, "early_stopping", None)
        self.model: Optional[lgb.LGBMClassifier] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LGBModel":
        """训练 LightGBM 模型"""
        logger.info(
            f"开始训练 LightGBM - 训练样本: {len(X_train)}, 特征: {X_train.shape[1]}"
        )

        # 自动调整 scale_pos_weight（配置为 null 时按训练集实际负正比计算）
        if self.params.get("scale_pos_weight") is None:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            self.params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            logger.info(
                f"自动设置 scale_pos_weight = {self.params['scale_pos_weight']:.2f}"
            )

        self.model = lgb.LGBMClassifier(**self.params)

        # 验证集 + 早停回调
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = []
        if eval_set and self.early_stopping_cfg and self.early_stopping_cfg.enabled:
            callbacks.append(
                lgb.early_stopping(self.early_stopping_cfg.rounds, verbose=False)
            )

        self.model.fit(
            X_train, y_train, eval_set=eval_set, callbacks=callbacks or None
        )

        # 记录训练历史
        if hasattr(self.model, "evals_result_"):
            self._train_history = self.model.evals_result_

        self._is_fitted = True
        self._feature_names = X_train.columns.tolist()

        if eval_set:
            from sklearn.metrics import roc_auc_score

            val_pred = self.model.predict_proba(X_val)[:, 1]
            logger.info(
                f"LightGBM 训练完成 - 验证集 AUC: {roc_auc_score(y_val, val_pred):.4f}"
            )
        else:
            logger.info("LightGBM 训练完成")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（返回正类概率，shape=(n,)）"""
        self._check_fitted()
        # 对齐特征
        if self._feature_names:
            X = X[self._feature_names]
        if isinstance(self.model, lgb.LGBMClassifier):
            return self.model.predict_proba(X)[:, 1]
        # 从文件加载时为 Booster
        return np.asarray(self.model.predict(X))

    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """
        获取特征重要性。

        参数:
            importance_type: split/gain
        """
        self._check_fitted()
        booster = self._booster()
        importance = booster.feature_importance(importance_type=importance_type)
        return pd.Series(importance, index=self._feature_names).sort_values(
            ascending=False
        )

    def save(self, file_path: str) -> None:
        """保存模型（LightGBM 文本格式）"""
        self._check_fitted()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self._booster().save_model(str(path))

        # 单独保存特征名（LightGBM 文本模型不包含）
        feature_path = path.with_suffix(".features.txt")
        feature_path.write_text("\n".join(self._feature_names), encoding="utf-8")

        logger.info(f"LightGBM 模型已保存至 {path}")

    def load(self, file_path: str) -> "LGBModel":
        """加载模型"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        self.model = lgb.Booster(model_file=str(path))

        # 加载特征名
        feature_path = path.with_suffix(".features.txt")
        if feature_path.exists():
            self._feature_names = feature_path.read_text(
                encoding="utf-8"
            ).splitlines()

        self._is_fitted = True
        logger.info(f"LightGBM 模型已从 {path} 加载")
        return self

    def _booster(self) -> lgb.Booster:
        """获取底层 Booster（兼容分类器包装与直接加载两种状态）"""
        if isinstance(self.model, lgb.LGBMClassifier):
            return self.model.booster_
        return self.model
