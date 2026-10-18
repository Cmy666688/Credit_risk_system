"""
数据加载器
==========
统一负责数据的读取、划分。

要点:
    1. 配置驱动：路径、参数都来自 YAML
    2. 分块读取：支持大数据集
    3. 分层抽样：保证训练/测试集分布一致
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.data.adapters import ADAPTER_REGISTRY

logger = get_logger("data.loader")


class DataLoader:
    """
    数据加载器，根据配置加载训练/测试集。

    使用示例:
        >>> loader = DataLoader(data_config)
        >>> df = loader.load_train()
    """

    def __init__(self, config):
        self.config = config
        self.paths = config.paths

    # -------- 多数据源支持（扩展点 3：添加新数据源） --------
    def _active_source(self):
        """返回当前数据源 (名称, 配置块)，默认 tianchi。"""
        source = getattr(self.paths, "source", "tianchi")
        sources = getattr(self.config, "sources", None) or {}
        return source, (sources.get(source) or {})

    def _source_path(self, key: str, fallback: str) -> str:
        """优先取当前数据源里的路径/文件名，否则回退到全局 paths。"""
        _, cfg = self._active_source()
        return cfg.get(key, fallback)

    def load_train(self, use_sample: bool = False) -> pd.DataFrame:
        """
        加载训练集。

        参数:
            use_sample: 是否使用示例数据（快速调试用）
        """
        if use_sample:
            file_path = Path(self.paths.sample_dir) / self.paths.sample_file
        else:
            raw_dir = self._source_path("raw_dir", self.paths.raw_dir)
            train_file = self._source_path("train_file", self.paths.train_file)
            file_path = Path(raw_dir) / train_file

        return self._load_csv(file_path)

    def load_test(self) -> pd.DataFrame:
        """加载测试集"""
        raw_dir = self._source_path("raw_dir", self.paths.raw_dir)
        test_file = self._source_path("test_file", self.paths.test_file)
        file_path = Path(raw_dir) / test_file
        return self._load_csv(file_path)

    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """加载 CSV 文件，支持大文件分块读取"""
        if not file_path.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {file_path}\n"
                f"请运行 `python scripts/generate_sample_data.py` 生成示例数据，"
                f"或从天池下载真实数据放入 data/raw/"
            )

        loading_cfg = self.config.loading
        file_size_mb = file_path.stat().st_size / (1024 ** 2)

        logger.info(f"加载数据: {file_path} (大小: {file_size_mb:.2f} MB)")

        # 大文件分块读取
        if file_size_mb > 500:
            logger.info(f"文件较大，启用分块读取（chunk_size={loading_cfg.chunk_size}）")
            chunks = []
            for chunk in pd.read_csv(
                file_path,
                chunksize=loading_cfg.chunk_size,
                encoding=loading_cfg.encoding,
                low_memory=loading_cfg.low_memory,
            ):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                file_path,
                encoding=loading_cfg.encoding,
                low_memory=loading_cfg.low_memory,
            )

        # 多数据源：按配置套用 Adapter，把外部 schema 归一化成天池 schema
        df = self._apply_adapter(df)

        logger.info(f"数据加载完成 - shape: {df.shape}")
        return df

    def _apply_adapter(self, df: pd.DataFrame) -> pd.DataFrame:
        """依据当前数据源配置，把外部数据归一化成天池 schema。

        配置来源：configs/data_config.yaml 的 paths.source + sources[source].adapter。
        adapter 为 null 时表示无需适配（如天池原生数据）。
        """
        source, source_cfg = self._active_source()
        adapter_name = source_cfg.get("adapter")
        if not adapter_name:
            return df

        if adapter_name not in ADAPTER_REGISTRY:
            raise KeyError(
                f"未知数据源 Adapter: {adapter_name}，"
                f"可选: {list(ADAPTER_REGISTRY)}"
            )

        adapter = ADAPTER_REGISTRY[adapter_name]()
        df = adapter.transform(df)

        # 归一化后必须满足 data_config 声明的字段契约，缺列尽早报错
        required = (
            list(self.config.columns.numerical)
            + list(self.config.columns.categorical)
            + [self.config.target_column, self.config.id_column]
        )
        adapter.validate(df, required)
        logger.info(f"已套用 Adapter: {adapter_name} - shape: {df.shape}")
        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """获取数据摘要信息"""
        target_col = self.config.target_column

        summary = {
            "n_samples": len(df),
            "n_features": df.shape[1],
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_ratio": float(df.isnull().sum().sum() / df.size),
        }

        if target_col in df.columns:
            target_dist = df[target_col].value_counts(normalize=True).to_dict()
            summary["default_rate"] = float(target_dist.get(1, 0))
            summary["target_distribution"] = target_dist

        return summary


class DataSplitter:
    """
    数据集划分器，支持分层抽样。

    使用示例:
        >>> splitter = DataSplitter(data_config)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df)
    """

    def __init__(self, config):
        self.config = config
        self.split_cfg = config.split
        self.target = config.target_column
        self.id_col = config.id_column

    def split(
        self,
        df: pd.DataFrame,
        with_validation: bool = True,
    ) -> Tuple:
        """
        将数据集划分为训练/验证/测试集。

        参数:
            df: 完整数据集
            with_validation: 是否生成验证集

        返回:
            (X_train, X_val, X_test, y_train, y_val, y_test) 或
            (X_train, X_test, y_train, y_test)
        """
        # 分离特征和标签
        drop_cols = [self.target]
        if self.id_col in df.columns:
            drop_cols.append(self.id_col)

        X = df.drop(columns=drop_cols, errors="ignore")
        y = df[self.target]

        stratify = y if self.split_cfg.stratify else None

        # 第一次划分：训练 vs 测试
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=self.split_cfg.test_size,
            random_state=self.split_cfg.random_state,
            stratify=stratify,
        )

        if not with_validation:
            logger.info(
                f"数据划分完成 - 训练: {X_trainval.shape[0]}, 测试: {X_test.shape[0]}"
            )
            return X_trainval, X_test, y_trainval, y_test

        # 第二次划分：训练 vs 验证
        # 注意 validation_size 是相对于原始数据的比例
        val_ratio = self.split_cfg.validation_size / (1 - self.split_cfg.test_size)
        stratify2 = y_trainval if self.split_cfg.stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_ratio,
            random_state=self.split_cfg.random_state,
            stratify=stratify2,
        )

        logger.info(
            f"数据划分完成 - 训练: {X_train.shape[0]}, "
            f"验证: {X_val.shape[0]}, 测试: {X_test.shape[0]}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
