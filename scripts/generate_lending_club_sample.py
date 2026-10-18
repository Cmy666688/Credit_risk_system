#!/usr/bin/env python
"""
Lending Club 示例数据生成器（扩展点 3：添加新数据源）
=====================================================
生成与 Lending Club 原始字段一致的合成数据，用于演示"多数据源 + Adapter"。

字段名刻意使用 Lending Club 的原始命名（loan_amnt / int_rate / loan_status ...），
与天池 schema 不同，从而可以通过 src/data/adapters/lending_club_adapter.py
演示 schema 归一化前后的差异。

使用:
    python scripts/generate_lending_club_sample.py --n_samples 20000
    python scripts/generate_lending_club_sample.py --n_samples 5000 --output data/external/lending_club.csv

注意: 若已下载真实数据集 data/external/lending_club.csv，则无需运行本脚本。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让脚本能 import src
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_lending_club(n_samples: int = 20000, random_seed: int = 42) -> pd.DataFrame:
    """生成与 Lending Club 原始 schema 一致的合成数据"""
    rng = np.random.default_rng(random_seed)
    n = n_samples

    grade = rng.choice(
        list("ABCDEFG"), n, p=[0.15, 0.25, 0.25, 0.18, 0.10, 0.05, 0.02]
    )
    fico_low = rng.normal(690, 30, n).clip(580, 845).round().astype(int)

    df = pd.DataFrame({
        "id": np.arange(n),

        # ===== 数值型（LC 原始命名）=====
        "loan_amnt": np.round(rng.lognormal(9.5, 0.6, n)).clip(500, 40000),
        "term": rng.choice([" 36 months", " 60 months"], n, p=[0.7, 0.3]),
        "int_rate": [f"{x:.2f}%" for x in rng.normal(13, 4, n).clip(5, 30)],
        "installment": np.round(rng.uniform(100, 1500, n), 2),
        "annual_inc": np.round(rng.lognormal(11, 0.6, n), 2),
        "dti": np.round(rng.normal(18, 8, n).clip(0, 60), 2),
        "delinq_2yrs": rng.poisson(0.3, n),
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "open_acc": rng.poisson(11, n),
        "pub_rec": rng.poisson(0.2, n),
        "pub_rec_bankruptcies": rng.poisson(0.1, n),
        "revol_bal": np.round(rng.lognormal(8.5, 1.2, n), 2),
        "revol_util": [f"{x:.2f}%" for x in (rng.beta(2, 3, n) * 100)],
        "total_acc": rng.poisson(25, n) + 1,

        # ===== 类别型（LC 原始命名）=====
        "grade": grade,
        "sub_grade": [f"{g}{i}" for g, i in zip(grade, rng.integers(1, 6, n))],
        "emp_title": rng.choice(
            ["Engineer", "Teacher", "Nurse", "Manager", "Driver", "Sales"],
            n,
        ),
        "emp_length": rng.choice(
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
             "6 years", "7 years", "8 years", "9 years", "10+ years"],
            n,
            p=[0.10, 0.07, 0.10, 0.09, 0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.30],
        ),
        "home_ownership": rng.choice(
            ["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.40, 0.50, 0.08, 0.02]
        ),
        "verification_status": rng.choice(
            ["Verified", "Source Verified", "Not Verified"], n, p=[0.30, 0.40, 0.30]
        ),
        "purpose": rng.choice(
            ["debt_consolidation", "credit_card", "home_improvement",
             "major_purchase", "medical", "small_business"],
            n,
        ),
        "zip_code": [f"{v:03d}xx" for v in rng.integers(0, 1000, n)],
        "addr_state": rng.choice(
            ["CA", "NY", "TX", "FL", "IL", "WA", "MA"], n
        ),
        "initial_list_status": rng.choice(["f", "w"], n, p=[0.6, 0.4]),
        "application_type": rng.choice(
            ["Individual", "Joint App"], n, p=[0.95, 0.05]
        ),
        "title": rng.choice(
            ["Debt consolidation", "Credit card refinancing", "Home improvement"],
            n,
        ),
        "policy_code": np.ones(n, dtype=int),

        # ===== 时间型（LC 用 "Mon-YYYY" 格式）=====
        "issue_d": [
            pd.Timestamp(x).strftime("%b-%Y")
            for x in rng.choice(pd.date_range("2014-01-01", "2019-12-31"), n)
        ],
        "earliest_cr_line": [
            pd.Timestamp(x).strftime("%b-%Y")
            for x in rng.choice(pd.date_range("1980-01-01", "2012-12-31"), n)
        ],

        # ===== 标签（LC 用字符串，需 Adapter 派生 isDefault）=====
        "loan_status": rng.choice(
            ["Fully Paid", "Charged Off", "Current", "Late (31-120 days)"],
            n,
            p=[0.60, 0.20, 0.15, 0.05],
        ),
    })

    # 模拟真实数据的少量缺失
    for col in ["emp_title", "dti", "revol_util", "pub_rec_bankruptcies"]:
        df.loc[rng.uniform(0, 1, n) < 0.02, col] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser(description="生成 Lending Club 示例数据集")
    parser.add_argument("--n_samples", type=int, default=20000, help="样本数")
    parser.add_argument(
        "--output", type=str, default="./data/external/lending_club.csv",
        help="输出文件路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⏳ 正在生成 {args.n_samples} 条 Lending Club 示例数据...")
    df = generate_lending_club(args.n_samples, args.seed)

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"💾 已保存: {out_path}")
    print("\n📊 数据概览（LC 原始 schema）:")
    print(f"  样本数:     {len(df):,}")
    print(f"  字段数:     {df.shape[1]}")
    print(f"  loan_status 分布:")
    for status, ratio in df["loan_status"].value_counts(normalize=True).items():
        print(f"     {status:<24} {ratio:.2%}")

    print("\n✅ 完成！可在 Web「数据探索」页选择「🌐 Lending Club」查看 Adapter 归一化效果。")


if __name__ == "__main__":
    main()
