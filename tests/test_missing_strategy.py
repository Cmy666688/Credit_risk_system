"""缺失值填充策略单元测试（扩展点 6：换缺失值填充策略）

独立测试文件（不并入 test_basic.py）。

覆盖:
    1. 回归：指导书示例只写了按列 strategy，但原代码只读全局 numerical_strategy /
       categorical_strategy，照抄会"改了配置却不生效"（本实现两者兼容）
    2. 按列覆盖：数值列 median 覆盖全局 mean、类别列 mode 覆盖全局 unknown；
       未覆盖的列沿用全局策略
    3. KNN：在训练集上 fit、推理时复用，能借助邻居填出缺失值
    4. KNN 无数据泄漏：测试集自身 dti 全空也能填（填充值来自训练集分布）
    5. 单笔推理可用（"现算现填"写法会在这一步失败）
    6. 边界：整列全空 / 全同列 / 未知策略 / 类别列误用 knn，均能安全回退
    7. 真实配置：feature_config.yaml 的 missing_value.strategy 被读取并激活 knn
    8. 保留指导书示例的 _fill_missing(df, col, strategy) 签名

运行: pytest tests/test_missing_strategy.py -v
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import Preprocessor
from src.utils.config_loader import AttrDict, ConfigLoader

CONFIGS = ConfigLoader(str(ROOT / "configs"))
DATA_CFG = CONFIGS.get("data")


# ===== 测试夹具 =====

def make_train(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """合成训练数据：dti 与 annualIncome 强相关，便于验证 KNN 邻居填充。"""
    rng = np.random.default_rng(seed)
    income = rng.normal(60000, 8000, n)
    return pd.DataFrame({
        "annualIncome": income,
        "dti": income / 1000.0,
        "employmentLength": rng.choice(
            ["< 1 year", "1 year", "10+ years"], n, p=[0.2, 0.6, 0.2]
        ),
    })


def feature_cfg(strategy=None, numerical="median", categorical="unknown",
                drop_threshold=0.95, knn_neighbors=5,
                knn_reference_sample=2000, knn_batch_size=10000) -> AttrDict:
    """构造只含 Preprocessor 所需字段的最小配置。"""
    return AttrDict({
        "missing_value": {
            "numerical_strategy": numerical,
            "categorical_strategy": categorical,
            "drop_threshold": drop_threshold,
            "strategy": dict(strategy or {}),
            "knn_neighbors": knn_neighbors,
            "knn_reference_sample": knn_reference_sample,
            "knn_batch_size": knn_batch_size,
        },
        "encoding": {"custom": {}},
        "outlier": {"method": "none", "lower_quantile": 0.01,
                    "upper_quantile": 0.99, "apply_to": []},
        "scaling": {"method": "none"},
    })


# ===== 2. 按列覆盖 vs 全局策略 =====

class TestPerColumnStrategy:
    def test_per_column_median_overrides_global_mean(self):
        """dti 按列指定 median，annualIncome 沿用全局 mean。"""
        df = make_train(200)
        df.loc[df.index[:40], "dti"] = np.nan            # dti: 按列 median
        df.loc[df.index[:10], "annualIncome"] = np.nan  # annualIncome: 全局 mean

        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "median"}, numerical="mean"))
        out = prep.fit_transform(df)

        assert out["dti"].notna().all() and out["annualIncome"].notna().all()
        # dti 用的是中位数，而不是全局 mean
        assert np.allclose(out.loc[df.index[:40], "dti"], df["dti"].median())
        # annualIncome 未覆盖 -> 用全局 mean
        assert np.allclose(out.loc[df.index[:10], "annualIncome"], df["annualIncome"].mean())
        # 确认 mean 与 median 确实不同，否则本用例没有区分度
        assert not np.isclose(df["annualIncome"].mean(), df["annualIncome"].median())

    def test_categorical_mode_overrides_global_unknown(self):
        """employmentLength 按列指定 mode，覆盖全局 unknown。"""
        df = make_train(300, seed=3)
        df.loc[df.index[:20], "employmentLength"] = np.nan

        prep = Preprocessor(DATA_CFG, feature_cfg(
            {"employmentLength": "mode"}, categorical="unknown"))
        out = prep.fit_transform(df)

        expected = df["employmentLength"].mode().iloc[0]
        assert prep._categorical_imputers["employmentLength"] == expected
        assert out["employmentLength"].notna().all()

    def test_unknown_strategy_falls_back_to_median(self):
        """未知策略不应崩溃，回退到 median（与旧行为一致）。"""
        df = make_train(60)
        df.loc[df.index[:5], "dti"] = np.nan

        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "bogus"}))
        out = prep.fit_transform(df)

        assert out["dti"].notna().all()
        # 仅原先缺失的行应被填成中位数（其余行保持原值）
        assert np.allclose(out.loc[df.index[:5], "dti"], df["dti"].median())

    def test_categorical_knn_unsupported_falls_back(self):
        """类别列没有数值距离空间，knn 应回退到全局类别策略而不是报错。"""
        df = make_train(50)
        df.loc[df.index[:5], "employmentLength"] = np.nan

        prep = Preprocessor(DATA_CFG, feature_cfg(
            {"employmentLength": "knn"}, categorical="unknown"))
        prep.fit(df)

        assert prep._categorical_imputers["employmentLength"] == "unknown"
        assert prep.transform(df)["employmentLength"].notna().all()


# ===== 3~5. KNN 填充 =====

class TestKnnStrategy:
    def test_knn_fills_from_closest_training_neighbor(self):
        """邻居填充：给一个 annualIncome 与训练样本一致的缺失 dti，应填出该邻居附近的值。"""
        train = make_train(300)
        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "knn"})).fit(train)

        test = pd.DataFrame({
            "annualIncome": [train["annualIncome"].iloc[5]],
            "dti": [np.nan],
            "employmentLength": ["1 year"],
        })
        out = prep.transform(test)

        assert out["dti"].notna().all()
        assert out["dti"].iloc[0] == pytest.approx(train["dti"].iloc[5], rel=0.1)

    def test_knn_fit_on_train_no_leakage(self):
        """测试集 dti 整列全空也能填 -> 说明填充值来自训练集邻居，而非测试集自身。"""
        train = make_train(300, seed=1)
        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "knn"})).fit(train)

        test = make_train(60, seed=2)
        test["dti"] = np.nan
        out = prep.transform(test)

        assert out["dti"].notna().all()
        # 填充值必须落在训练集 dti 的取值范围内
        assert out["dti"].between(train["dti"].min(), train["dti"].max()).all()
        # 多次 transform 结果一致（复用训练好的填充器，无副作用）
        out2 = prep.transform(test.copy())
        assert np.allclose(out["dti"].to_numpy(), out2["dti"].to_numpy())

    def test_single_row_inference(self):
        """单笔推理必须可用（指导书"现算现填"写法单行无邻居会失败）。"""
        train = make_train(200)
        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "knn"})).fit(train)

        single = pd.DataFrame({
            "annualIncome": [60000.0], "dti": [np.nan], "employmentLength": ["1 year"],
        })
        out = prep.transform(single)

        assert len(out) == 1
        assert out["dti"].notna().all()

    def test_knn_neighbors_configurable(self):
        """knn_neighbors 参数被真正读取（1 个邻居时应等于最近邻的 dti）。"""
        train = make_train(300, seed=5)
        prep = Preprocessor(DATA_CFG, feature_cfg(
            {"dti": "knn"}, knn_neighbors=1)).fit(train)

        test = pd.DataFrame({
            "annualIncome": [train["annualIncome"].iloc[7]],
            "dti": [np.nan], "employmentLength": ["1 year"],
        })
        out = prep.transform(test)
        assert out["dti"].iloc[0] == pytest.approx(train["dti"].iloc[7], rel=0.05)


# ===== 6. 边界情况 =====

class TestEdgeCases:
    def test_all_nan_knn_column_falls_back(self):
        """整列全空的 knn 目标列无法建立邻居 -> 回退兜底值，不留 NaN。"""
        train = make_train(100)
        train["dti"] = np.nan
        # drop_threshold=1.0 让它不被"高缺失列"规则提前丢弃，才能走到 imputer
        cfg = feature_cfg({"dti": "knn"}, drop_threshold=1.0)
        prep = Preprocessor(DATA_CFG, cfg)

        out = prep.fit_transform(train)
        assert out["dti"].notna().all()
        assert (out["dti"] == 0.0).all()

    def test_constant_column_filled_with_same_value(self):
        """全同列：除缺失外取值恒为 7.0，median 填充后应仍为 7.0。"""
        df = make_train(50)
        df["dti"] = 7.0
        df.loc[df.index[:5], "dti"] = np.nan

        prep = Preprocessor(DATA_CFG, feature_cfg({"dti": "median"}))
        out = prep.fit_transform(df)

        assert out["dti"].notna().all()
        assert (out["dti"] == 7.0).all()

    def test_config_without_strategy_key_is_backward_compatible(self):
        """旧配置没有 strategy 键时，应与改动前行为一致（不激活 knn）。"""
        cfg = AttrDict({
            "missing_value": {"numerical_strategy": "median",
                              "categorical_strategy": "unknown",
                              "drop_threshold": 0.95},
            "encoding": {"custom": {}},
            "outlier": {"method": "none", "lower_quantile": 0.01,
                        "upper_quantile": 0.99, "apply_to": []},
            "scaling": {"method": "none"},
        })
        df = make_train(60)
        df.loc[df.index[:5], "dti"] = np.nan

        prep = Preprocessor(DATA_CFG, cfg)
        out = prep.fit_transform(df)

        assert out["dti"].notna().all()
        assert prep._knn_imputer is None       # 未配置 knn，不应拟合


# ===== 7~8. 真实配置 / 接口约定 =====

class TestRealConfigAndContract:
    def test_real_config_strategy_block_is_read(self):
        """指导书给出的配置块必须真的被 Preprocessor 读到并生效。"""
        feat = CONFIGS.get("feature")
        assert feat.missing_value.strategy["annualIncome"] == "median"
        assert feat.missing_value.strategy["employmentLength"] == "mode"
        assert feat.missing_value.strategy["dti"] == "knn"
        # 全局键仍在（app/pages/2_🔧_特征工程.py 与旧逻辑依赖）
        assert feat.missing_value.numerical_strategy == "median"
        assert feat.missing_value.categorical_strategy == "unknown"

        df = make_train(60)
        df.loc[df.index[3:8], "dti"] = np.nan
        df.loc[df.index[:2], "employmentLength"] = np.nan

        prep = Preprocessor(DATA_CFG, feat)
        out = prep.fit_transform(df)

        assert out.isnull().sum().sum() == 0   # 全流程无残留缺失
        assert prep._knn_imputer is not None   # dti: knn 已激活

    def test_fill_missing_keeps_guide_signature(self):
        """保留指导书示例的 _fill_missing(df, col, strategy) 入口签名。"""
        params = list(inspect.signature(Preprocessor._fill_missing).parameters)
        assert params == ["self", "df", "col", "strategy"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
