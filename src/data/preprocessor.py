"""
数据预处理器
============
负责缺失值填充、类别编码、数值标准化、异常值处理等基础预处理。

要点:
    1. fit/transform 分离设计（参考 sklearn 风格）
    2. 状态可保存：训练集 fit 的统计量保留下来用于推理
    3. 流水线友好：每个步骤独立可测试

扩展点 6（换缺失值填充策略）:
    1. 支持在 feature_config.yaml 的 missing_value.strategy 里按列覆盖填充策略
       （未覆盖的列沿用全局 numerical_strategy / categorical_strategy）
    2. 新增 knn 策略，入口方法保留指导书示例的 `_fill_missing(df, col, strategy)` 签名
    3. 相对指导书示例的改动（原因）：
       a. 指导书把策略写成"按列字典 strategy"，但原代码只读全局 numerical_strategy /
          categorical_strategy，照抄会导致改了配置却不生效——这里同时兼容两者。
       b. 指导书的 _fill_missing 是"现算现填"，在 transform/推理时会拿待填数据自己当邻居
          （数据泄漏），且单笔样本无邻居可用——这里改为 fit 阶段学好、transform 复用。
       c. knn 属于"新策略"，必须在 Preprocessor 里实现，与指导书"否则不要修改"并不矛盾，
          但指导书给的逐列 if/elif 骨架没有 KNN 需要的"邻居特征空间"，此处补齐。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger("data.preprocessor")


class Preprocessor:
    """
    数据预处理器。

    使用示例:
        >>> prep = Preprocessor(data_cfg, feature_cfg)
        >>> X_train_clean = prep.fit_transform(X_train)
        >>> X_test_clean = prep.transform(X_test)
    """

    def __init__(self, data_config, feature_config):
        self.data_cfg = data_config
        self.feature_cfg = feature_config

        # 状态：fit 时学到的参数
        self._numerical_imputers: Dict[str, float] = {}
        self._categorical_imputers: Dict[str, str] = {}
        # 扩展点 6：每列最终解析出的填充策略，驱动 _fill_missing 分发
        self._col_strategies: Dict[str, str] = {}
        # 扩展点 6：knn 状态。关键点是"训练集上拟合一次、推理时复用"：
        # 这样 test/线上数据不会拿自身当邻居（数据泄漏），单笔推理也能找到邻居。
        self._knn_columns: list = []
        self._knn_reference_cols: list = []
        self._knn_fallback: Dict[str, float] = {}
        self._knn_imputer: Optional[KNNImputer] = None
        # 扩展点 6：KNN 的距离必须建立在同一尺度上，故对参考特征单独做标准化
        self._knn_scaler: Optional[StandardScaler] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._scaler = None
        self._dropped_cols: list = []
        self._outlier_clips: Dict[str, tuple] = {}

        self._is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合 + 转换"""
        return self.fit(df).transform(df)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """从训练集学习预处理参数"""
        logger.info(f"开始拟合预处理器 - 输入 shape: {df.shape}")
        df = df.copy()

        # 1. 移除高缺失率列
        self._fit_drop_high_missing(df)
        df = df.drop(columns=self._dropped_cols, errors="ignore")

        # 2. 学习缺失值填充值
        self._fit_imputers(df)

        # 3. 学习类别编码器
        self._fit_encoders(df)

        # 4. 学习异常值截断范围
        self._fit_outlier_clips(df)

        self._is_fitted = True
        logger.info(f"预处理器拟合完成 - 移除 {len(self._dropped_cols)} 列")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用学到的参数对数据进行转换"""
        if not self._is_fitted:
            raise RuntimeError("预处理器未拟合，请先调用 fit() 或 fit_transform()")

        df = df.copy()

        # 1. 移除高缺失列
        df = df.drop(columns=self._dropped_cols, errors="ignore")

        # 2. 缺失值填充
        df = self._apply_imputers(df)

        # 3. 异常值截断
        df = self._apply_outlier_clips(df)

        # 4. 类别编码
        df = self._apply_encoders(df)

        # 5. 时间特征处理（简化版）
        df = self._handle_datetime(df)

        # 6. 数值标准化
        df = self._apply_scaler(df)

        logger.info(f"transform 完成 - 输出 shape: {df.shape}")
        return df

    # -------------- 私有方法（按步骤拆分） --------------

    def _fit_drop_high_missing(self, df: pd.DataFrame) -> None:
        """识别高缺失率列"""
        threshold = self.feature_cfg.missing_value.drop_threshold
        missing_ratio = df.isnull().mean()
        self._dropped_cols = missing_ratio[missing_ratio > threshold].index.tolist()
        if self._dropped_cols:
            logger.info(f"高缺失率列将被移除: {self._dropped_cols}")

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """学习缺失值填充策略（支持按列覆盖，见扩展点 6）"""
        missing_cfg = self.feature_cfg.missing_value
        num_strategy = missing_cfg.numerical_strategy
        cat_strategy = missing_cfg.categorical_strategy
        # 扩展点 6：按列覆盖策略。指导书示例把它写在 missing_value.strategy 下；
        # 未列出的列继续沿用全局 numerical_strategy / categorical_strategy。
        per_col = missing_cfg.get("strategy", {}) or {}

        # 数值列
        num_cols = [c for c in self.data_cfg.columns.numerical if c in df.columns]
        num_cols += [c for c in self.data_cfg.columns.anonymous if c in df.columns]
        for col in num_cols:
            strategy = per_col.get(col, num_strategy)
            self._col_strategies[col] = strategy
            if strategy == "knn":
                # knn 不在 fit 时算单值，而是拟合整表填充器（见 _fit_knn）
                self._knn_columns.append(col)
                continue
            self._numerical_imputers[col] = self._compute_numerical_fill(df[col], strategy)

        # 类别列
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        for col in cat_cols:
            strategy = per_col.get(col, cat_strategy)
            if strategy == "knn":
                # 类别列没有数值距离空间，无法用 KNN，回退到全局类别策略
                logger.warning(f"类别列 {col} 不支持 knn 策略，回退为 {cat_strategy}")
                strategy = cat_strategy
            self._col_strategies[col] = strategy
            self._categorical_imputers[col] = self._compute_categorical_fill(df[col], strategy)

        # 扩展点 6：knn 填充器必须在训练集上 fit，供推理时复用
        if self._knn_columns:
            self._fit_knn(df)

    def _compute_numerical_fill(self, series: pd.Series, strategy: str) -> float:
        """计算数值列的填充值。未列出的/未知策略回退 median，与旧行为一致。"""
        if strategy == "mean":
            value = series.mean()
        elif strategy == "zero":
            value = 0.0
        elif strategy == "mode":
            mode = series.mode()
            value = mode.iloc[0] if len(mode) > 0 else np.nan
        elif strategy == "median":
            value = series.median()
        else:
            logger.warning(f"未知的数值填充策略 '{strategy}'，回退为 median")
            value = series.median()
        # 整列全空时 median/mode 都是 NaN，兜底为 0 以保证下游不再有缺失
        if pd.isna(value):
            value = 0.0
        return value

    def _compute_categorical_fill(self, series: pd.Series, strategy: str) -> str:
        """计算类别列的填充值。"""
        if strategy == "mode":
            mode = series.mode()
            return mode.iloc[0] if len(mode) > 0 else "unknown"
        return "unknown"

    def _fit_knn(self, df: pd.DataFrame) -> None:
        """在训练集上拟合 KNN 填充器（只在 fit 阶段执行一次）。

        指导书示例是在 _fill_missing 里"现算现填"，那样 transform/推理时会把待填数据
        自己当作邻居（数据泄漏），且单笔样本无邻居可用。这里改为 fit 一次、推理复用。
        """
        # 邻居特征空间 = 数值列（含匿名列）。整列全空的列不含信息，且 sklearn
        # KNNImputer 会静默丢列导致列数对不上，必须先剔除。
        ref_cols = [c for c in (list(self.data_cfg.columns.numerical)
                                + list(self.data_cfg.columns.anonymous))
                    if c in df.columns and c not in self._dropped_cols]
        ref_df = df[[c for c in ref_cols if df[c].notna().any()]]

        # 控制"找邻居"的样本量：KNNImputer.transform 是 O(n_query × n_ref)，
        # 直接用全量训练集会爆内存/极慢，故取一个有上限的参考子集。
        sample = int(self.feature_cfg.missing_value.get("knn_reference_sample", 2000))
        if 0 < sample < len(ref_df):
            ref_df = ref_df.sample(n=sample, random_state=42)
        # 采样后仍可能出现整列全空，再清一次
        ref_df = ref_df.loc[:, ref_df.notna().any(axis=0)]
        self._knn_reference_cols = list(ref_df.columns)

        # 落不到参考特征里的 knn 目标列（训练集全空）-> 兜底值
        for col in self._knn_columns:
            if col not in self._knn_reference_cols:
                self._knn_fallback[col] = 0.0
                logger.warning(f"列 {col} 在训练集无有效邻居特征，knn 回退 0.0")

        if self._knn_reference_cols:
            n_neighbors = int(self.feature_cfg.missing_value.get("knn_neighbors", 5))
            # KNNImputer 按欧氏距离找邻居。若各列量纲差异悬殊（annualIncome≈1e5 vs
            # dti≈20），距离会被大数量级列主导，"数值小但强相关"的特征（收入类）等于失效。
            # 所以先把参考特征标准化到同一尺度，再做邻居搜索（scale_ 为 0 时 sklearn
            # 会自动置为 1，不会除零）。
            self._knn_scaler = StandardScaler()
            ref_scaled = self._knn_scaler.fit_transform(ref_df)
            self._knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            self._knn_imputer.fit(ref_scaled)
        else:
            self._knn_imputer = None
            self._knn_scaler = None

    def _apply_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用缺失值填充（按列分发到 _fill_missing；knn 需整表先处理）"""
        # knn 依赖整表邻居特征，先统一填充
        if self._knn_columns:
            df = self._apply_knn(df)

        for col, strategy in self._col_strategies.items():
            if strategy == "knn":
                continue  # 已在 _apply_knn 处理
            df = self._fill_missing(df, col, strategy)

        return df

    def _fill_missing(self, df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
        """按策略填充单列（扩展点 6 的扩展入口）。

        保留指导书示例的同名同签名 `_fill_missing(df, col, strategy)`，但把"学参数"
        和"用参数"分开：填充值在 fit 阶段已算好（knn 则是 fit 好的填充器），本方法
        只负责套用，避免 transform 时重新统计导致数据泄漏。
        """
        if col not in df.columns:
            return df
        if strategy == "knn":
            return df  # 已在 _apply_knn 中处理
        if col in self._numerical_imputers:
            df[col] = df[col].fillna(self._numerical_imputers[col])
        elif col in self._categorical_imputers:
            df[col] = df[col].fillna(self._categorical_imputers[col]).astype(str)
        return df

    def _apply_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """用训练集拟合好的 KNN 填充器填充目标列。"""
        imputed_df = pd.DataFrame(index=df.index)
        if self._knn_imputer is not None and self._knn_reference_cols:
            cols = self._knn_reference_cols
            # reindex 保证列顺序/列数与 fit 时一致（缺列自动补 NaN）
            batch_size = int(self.feature_cfg.missing_value.get("knn_batch_size", 10000))
            values = np.empty((len(df), len(cols)), dtype=float)
            for start in range(0, len(df), max(batch_size, 1)):
                chunk = df.iloc[start:start + batch_size].reindex(columns=cols)
                # 与 fit 时一致：先标准化再找邻居，避免大数量级列主导距离
                scaled = self._knn_scaler.transform(chunk)
                values[start:start + batch_size] = self._knn_imputer.transform(scaled)
            # KNNImputer 给出的是"标准化尺度下的邻居均值"，需还原回原始尺度
            values = self._knn_scaler.inverse_transform(values)
            imputed_df = pd.DataFrame(values, columns=cols, index=df.index)

        for col in self._knn_columns:
            if col in imputed_df.columns:
                values = imputed_df[col]
                if values.isnull().any():
                    # 邻居不足等极端情况 -> 兜底，保证无残留 NaN
                    values = values.fillna(self._knn_fallback.get(col, 0.0))
            else:
                values = pd.Series(self._knn_fallback.get(col, 0.0), index=df.index, dtype=float)
            df[col] = values
        return df

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """学习类别编码器"""
        cat_cols = [c for c in self.data_cfg.columns.categorical if c in df.columns]
        custom_mappings = self.feature_cfg.encoding.get("custom", {})

        for col in cat_cols:
            # 优先使用配置文件中的自定义映射
            if col in custom_mappings:
                # 自定义映射不需要 fit
                continue

            le = LabelEncoder()
            # 兼容缺失值（先填充）
            values = df[col].fillna("unknown").astype(str)
            le.fit(values)
            self._label_encoders[col] = le

    def _apply_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用类别编码"""
        custom_mappings = self.feature_cfg.encoding.get("custom", {})

        # 先应用自定义映射
        for col, cfg in custom_mappings.items():
            if col in df.columns and "mapping" in cfg:
                mapping = cfg["mapping"]
                # 未在映射中的值用 -1 表示
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        # 再应用学到的 LabelEncoder
        for col, le in self._label_encoders.items():
            if col in df.columns:
                values = df[col].fillna("unknown").astype(str)
                # 未见过的类别 -> 标记为 -1
                known = set(le.classes_)
                values = values.map(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(values)

        return df

    def _fit_outlier_clips(self, df: pd.DataFrame) -> None:
        """学习异常值截断阈值"""
        outlier_cfg = self.feature_cfg.outlier
        if outlier_cfg.method != "clip":
            return

        for col in outlier_cfg.apply_to:
            if col in df.columns:
                lower = df[col].quantile(outlier_cfg.lower_quantile)
                upper = df[col].quantile(outlier_cfg.upper_quantile)
                self._outlier_clips[col] = (lower, upper)

    def _apply_outlier_clips(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用异常值截断"""
        for col, (lower, upper) in self._outlier_clips.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _handle_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """简化版时间特征处理"""
        for col in self.data_cfg.columns.datetime:
            if col not in df.columns:
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
                df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
                df = df.drop(columns=[col])
            except Exception as e:
                logger.warning(f"时间列 {col} 处理失败: {e}")
                df = df.drop(columns=[col], errors="ignore")
        return df

    def _apply_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """数值标准化（在所有处理后统一应用）"""
        method = self.feature_cfg.scaling.method
        if method == "none":
            return df

        # 仅对数值列（含衍生数值）做标准化
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        if self._scaler is None:
            if method == "standard":
                self._scaler = StandardScaler()
            elif method == "minmax":
                self._scaler = MinMaxScaler()
            elif method == "robust":
                self._scaler = RobustScaler()
            else:
                return df

            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        else:
            # 对齐列
            scaler_cols = [c for c in numeric_cols if hasattr(self._scaler, "mean_")]
            if scaler_cols:
                df[scaler_cols] = self._scaler.transform(df[scaler_cols])

        return df
