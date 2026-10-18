"""Lending Club 数据源适配器（扩展点 3）

把 Lending Club 的原始 schema 归一化成天池 schema，
使后续的特征工程、模型训练无需感知数据来源差异。

要点:
    1. 字段重命名：把外部字段名映射到 data_config.yaml 里的天池字段名
    2. 关键列派生：标签列(isDefault)、ID 列(id)
    3. 单位清洗：int_rate / revol_util 带有 "%"，term 形如 "36 months"
    4. 缺列补齐：天池的匿名特征 n0~n14 在 Lending Club 中不存在
"""
from __future__ import annotations

import pandas as pd


class LendingClubAdapter:
    """把 Lending Club 数据 schema 归一化到天池 schema"""

    # 源字段 -> 天池字段（与 configs/data_config.yaml 的 columns 对齐）
    FIELD_MAPPING = {
        # --- 数值型 ---
        "loan_amnt": "loanAmnt",
        "term": "term",
        "int_rate": "interestRate",
        "installment": "installment",
        "annual_inc": "annualIncome",
        "dti": "dti",
        "delinq_2yrs": "delinquency_2years",
        "fico_range_low": "ficoRangeLow",
        "fico_range_high": "ficoRangeHigh",
        "open_acc": "openAcc",
        "pub_rec": "pubRec",
        "pub_rec_bankruptcies": "pubRecBankruptcies",
        "revol_bal": "revolBal",
        "revol_util": "revolUtil",
        "total_acc": "totalAcc",
        # --- 类别型 ---
        "grade": "grade",
        "sub_grade": "subGrade",
        "emp_title": "employmentTitle",
        "emp_length": "employmentLength",
        "home_ownership": "homeOwnership",
        "verification_status": "verificationStatus",
        "purpose": "purpose",
        "zip_code": "postCode",
        "addr_state": "regionCode",
        "initial_list_status": "initialListStatus",
        "application_type": "applicationType",
        "title": "title",
        "policy_code": "policyCode",
        # --- 时间型 ---
        "issue_d": "issueDate",
        "earliest_cr_line": "earliesCreditLine",
    }

    # 明确"正常"的 loan_status -> isDefault = 0
    GOOD_STATUS = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
    # 明确"违约"的 loan_status -> isDefault = 1
    BAD_STATUS = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
    # 其余取值（Current / In Grace Period / Late (16-30 days) / Late (31-120 days)）
    # 结局不明确，不参与建模，在 transform 中整行剔除

    # 带 "%" 的字符串列，需清洗成数值
    PERCENT_COLUMNS = ("interestRate", "revolUtil")

    # 天池的匿名特征，Lending Club 没有，补 0
    ANONYMOUS_COLUMNS = [f"n{i}" for i in range(15)]

    def transform(self, df: pd.DataFrame, drop_extra_columns: bool = True) -> pd.DataFrame:
        df = df.rename(columns=self.FIELD_MAPPING)

        # 1) 标签列：只保留有明确结局的样本，剔除在贷/宽限期等未定样本
        if "loan_status" in df.columns:
            df = df[df["loan_status"].isin(self.GOOD_STATUS | self.BAD_STATUS)].copy()
            df["isDefault"] = (~df["loan_status"].isin(self.GOOD_STATUS)).astype(int)
        else:
            df = df.copy()

        # 2) 清洗带单位的字符串列（用 regex 替换，比 rstrip 更稳，且兼容非字符串/缺失值）
        for col in self.PERCENT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("%", "", regex=False),
                    errors="coerce",
                )

        # term 形如 "36 months"，抽出数字
        if "term" in df.columns:
            df["term"] = pd.to_numeric(
                df["term"].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            )

        # 3) 补齐天池有、但 Lending Club 没有的列
        for col in self.ANONYMOUS_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # 4) ID 列
        if "id" not in df.columns:
            df["id"] = range(len(df))

        # 5) 只保留天池 schema 字段，丢弃 LC 的多余列（member_id/url/desc/hardship_* 等）
        if drop_extra_columns:
            keep = (
                set(self.FIELD_MAPPING.values())
                | set(self.ANONYMOUS_COLUMNS)
                | {"isDefault", "id"}
            )
            df = df[[c for c in df.columns if c in keep]]

        return df.reset_index(drop=True)

    def validate(self, df: pd.DataFrame, required_columns) -> None:
        """校验归一化后是否满足 data_config 要求，缺列尽早报错"""
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"数据源字段缺失，未对齐天池 schema: {missing}")
