"""数据源 Adapter 单元测试（扩展点 3：添加新数据源）

覆盖:
    1. Lending Club Adapter 把外部 schema 归一化成天池 schema
    2. 标签列(isDefault)、ID 列(id)、单位列 等关键转换
    3. 缺列时 validate() 尽早报错
    4. DataLoader 按 data_config 的 source 配置自动套用 / 跳过 Adapter

运行: pytest tests/test_data_adapter.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.adapters import LendingClubAdapter
from src.utils.config_loader import AttrDict, ConfigLoader

CONFIGS = ConfigLoader(str(ROOT / "configs"))
DATA_CFG = CONFIGS.get("data")

# 天池 schema 的字段契约：Adapter 归一化后必须全部满足
TIANCHI_SCHEMA = (
    list(DATA_CFG.columns.numerical)
    + list(DATA_CFG.columns.categorical)
    + list(DATA_CFG.columns.datetime)
    + list(DATA_CFG.columns.anonymous)
)
REQUIRED_COLUMNS = TIANCHI_SCHEMA + [DATA_CFG.target_column, DATA_CFG.id_column]


def make_raw_lending_club(n: int = 20, status: str = "Fully Paid") -> pd.DataFrame:
    """构造一份 Lending Club 原始数据（字段名使用 LC 的原始命名）"""
    return pd.DataFrame({
        "loan_amnt": [10000.0] * n,
        "term": ["36 months"] * n,
        "int_rate": ["13.56%"] * n,
        "installment": [300.0] * n,
        "annual_inc": [60000.0] * n,
        "dti": [10.0] * n,
        "delinq_2yrs": [0] * n,
        "fico_range_low": [700] * n,
        "fico_range_high": [704] * n,
        "open_acc": [10] * n,
        "pub_rec": [0] * n,
        "pub_rec_bankruptcies": [0] * n,
        "revol_bal": [5000.0] * n,
        "revol_util": ["30.5%"] * n,
        "total_acc": [20] * n,
        "grade": ["B"] * n,
        "sub_grade": ["B2"] * n,
        "emp_title": ["engineer"] * n,
        "emp_length": ["5 years"] * n,
        "home_ownership": ["RENT"] * n,
        "verification_status": ["Verified"] * n,
        "purpose": ["debt_consolidation"] * n,
        "zip_code": ["123xx"] * n,
        "addr_state": ["CA"] * n,
        "initial_list_status": ["f"] * n,
        "application_type": ["Individual"] * n,
        "title": ["Debt consolidation"] * n,
        "policy_code": [1] * n,
        "issue_d": ["Jan-2019"] * n,
        "earliest_cr_line": ["Jan-2010"] * n,
        "loan_status": [status] * n,
    })


class TestLendingClubAdapter:
    def test_transform_aligns_schema(self):
        out = LendingClubAdapter().transform(make_raw_lending_club())
        # 归一化后应完全满足天池 schema（validate 不报错）
        LendingClubAdapter().validate(out, REQUIRED_COLUMNS)
        assert "isDefault" in out.columns
        assert "id" in out.columns

    def test_unit_columns_cleaned(self):
        out = LendingClubAdapter().transform(make_raw_lending_club())
        assert out["interestRate"].iloc[0] == pytest.approx(13.56)   # "13.56%" -> 13.56
        assert out["revolUtil"].iloc[0] == pytest.approx(30.5)       # "30.5%" -> 30.5
        assert out["term"].iloc[0] == pytest.approx(36)              # "36 months" -> 36

    def test_label_derivation(self):
        good = LendingClubAdapter().transform(make_raw_lending_club(status="Fully Paid"))
        bad = LendingClubAdapter().transform(make_raw_lending_club(status="Charged Off"))
        assert good["isDefault"].iloc[0] == 0
        assert bad["isDefault"].iloc[0] == 1

    def test_ambiguous_status_rows_dropped(self):
        """结局不明确的样本（Current/宽限期/逾期未坏账）应被剔除，不参与建模"""
        raw = make_raw_lending_club(5, status="Fully Paid")
        raw.loc[1, "loan_status"] = "Current"          # 在贷 -> 剔除
        raw.loc[2, "loan_status"] = "Late (16-30 days)"  # 逾期未坏账 -> 剔除
        raw.loc[3, "loan_status"] = "Charged Off"      # 违约 -> 保留
        out = LendingClubAdapter().transform(raw)
        assert len(out) == 3                            # 5 -> 3 行
        assert set(out["isDefault"]) == {0, 1}
        assert out["isDefault"].sum() == 1              # 仅 Charged Off 那行为 1

    def test_extra_columns_dropped_to_tianchi_schema(self):
        """归一化后应只剩天池 schema（丢弃 LC 的 member_id/url/loan_status 等多余列）"""
        raw = make_raw_lending_club(5)
        raw["member_id"] = range(5)
        raw["desc"] = "junk"
        out = LendingClubAdapter().transform(raw)
        assert "member_id" not in out.columns
        assert "desc" not in out.columns
        assert "loan_status" not in out.columns
        assert set(out.columns) == set(REQUIRED_COLUMNS)

    def test_keep_extra_columns_when_disabled(self):
        raw = make_raw_lending_club(5)
        raw["member_id"] = range(5)
        out = LendingClubAdapter().transform(raw, drop_extra_columns=False)
        assert "member_id" in out.columns       # 关闭归一化裁剪后保留原始列

    def test_anonymous_columns_filled(self):
        out = LendingClubAdapter().transform(make_raw_lending_club())
        for i in range(15):
            assert out[f"n{i}"].iloc[0] == 0

    def test_validate_raises_on_missing_columns(self):
        with pytest.raises(ValueError):
            LendingClubAdapter().validate(pd.DataFrame({"foo": [1]}), REQUIRED_COLUMNS)


class TestDataLoaderWithAdapter:
    def _config(self, tmp_path: Path, source: str = "lending_club") -> AttrDict:
        return AttrDict({
            "paths": {
                "raw_dir": "./data/raw",
                "processed_dir": "./data/processed",
                "sample_dir": "./data/sample",
                "model_dir": "./data/models",
                "train_file": "train.csv",
                "test_file": "testA.csv",
                "sample_file": "sample.csv",
                "source": source,
            },
            "loading": {"chunk_size": 100000, "encoding": "utf-8", "low_memory": False},
            "columns": {
                "numerical": list(DATA_CFG.columns.numerical),
                "categorical": list(DATA_CFG.columns.categorical),
            },
            "target_column": DATA_CFG.target_column,
            "id_column": DATA_CFG.id_column,
            "sources": {
                "lending_club": {
                    "adapter": "lending_club",
                    "raw_dir": str(tmp_path),
                    "train_file": "loan_train.csv",
                }
            },
        })

    def test_load_train_applies_adapter(self, tmp_path):
        from src.data.loader import DataLoader

        make_raw_lending_club(30, status="Charged Off").to_csv(
            tmp_path / "loan_train.csv", index=False
        )
        df = DataLoader(self._config(tmp_path)).load_train()

        assert df.shape[0] == 30
        assert "isDefault" in df.columns          # 标签列已派生
        assert df["isDefault"].iloc[0] == 1
        assert "loanAmnt" in df.columns           # 字段已重命名

    def test_loader_skips_adapter_for_tianchi(self, tmp_path):
        from src.data.loader import DataLoader

        loader = DataLoader(self._config(tmp_path, source="tianchi"))
        out = loader._apply_adapter(pd.DataFrame({"loan_amnt": [1]}))
        # tianchi 的 adapter 为 null -> 原样返回，不做任何转换
        assert list(out.columns) == ["loan_amnt"]

    def test_unknown_adapter_raises(self, tmp_path):
        from src.data.loader import DataLoader

        cfg = self._config(tmp_path)
        cfg.sources.lending_club.adapter = "no_such_adapter"
        with pytest.raises(KeyError):
            DataLoader(cfg)._apply_adapter(pd.DataFrame({"loan_amnt": [1]}))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
