"""
数据加载工具
============
所有 Streamlit 页面共用的数据加载函数，统一回退逻辑：
    1. 优先用真实数据 data/raw/train.csv（天池下载）
    2. 其次用合成数据 data/sample/sample.csv（脚本生成）
    3. 都没有时给出明确的提示
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
import streamlit as st


def find_data_file(
    root: Path,
    prefer: Literal["auto", "real", "sample"] = "auto",
) -> Tuple[Optional[Path], Optional[str]]:
    """
    查找可用的数据文件。

    参数
    ----
    root: 项目根目录
    prefer: "real"/"sample"/"auto"
        - real:   只用真实数据，没有就返回 None
        - sample: 只用合成数据，没有就返回 None
        - auto:   优先真实数据，回退到合成数据

    返回
    ----
    (path, source_label)：找到的文件路径和数据来源标签（"真实数据"/"合成数据"）
    没找到则返回 (None, None)
    """
    real_path = root / "data" / "raw" / "train.csv"
    sample_path = root / "data" / "sample" / "sample.csv"

    if prefer == "real":
        return (real_path, "真实数据") if real_path.exists() else (None, None)
    if prefer == "sample":
        return (sample_path, "合成数据") if sample_path.exists() else (None, None)

    # auto
    if real_path.exists():
        return real_path, "真实数据"
    if sample_path.exists():
        return sample_path, "合成数据"
    return None, None


@st.cache_data(show_spinner=False)
def load_data(
    root_str: str,
    prefer: str = "auto",
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    加载数据。被 @st.cache_data 装饰，同样的参数只读盘一次。

    注意 root 用 str 传入是因为 cache_data 不能哈希 Path 对象。

    参数
    ----
    root_str: 项目根目录字符串
    prefer: "auto"/"real"/"sample"
    nrows:  可选限制读取行数（大文件预览时用）

    返回
    ----
    (DataFrame, source_label)。找不到时返回 (空DF, None)。
    """
    root = Path(root_str)
    path, source = find_data_file(root, prefer)  # type: ignore[arg-type]
    if path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(path, nrows=nrows)
    return df, source


def render_data_status_or_stop(
    root: Path,
    prefer: Literal["auto", "real", "sample"] = "auto",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    页面级辅助：尝试加载数据，加载成功显示状态条；失败给出明确提示并 st.stop()。
    返回成功加载的 DataFrame。
    """
    df, source = load_data(str(root), prefer=prefer, nrows=nrows)

    if df.empty:
        st.error("❌ 未找到任何可用数据")
        st.markdown(
            "请通过以下任一方式准备数据：\n\n"
            "**方式 1：使用真实天池数据**（推荐）\n"
            "```bash\n"
            "# 1. 从 https://tianchi.aliyun.com/competition/entrance/531830 报名下载\n"
            "# 2. 把 train.csv 放到下面这个位置\n"
            f"#    {root / 'data' / 'raw' / 'train.csv'}\n"
            "```\n\n"
            "**方式 2：生成合成数据**（快速测试）\n"
            "```bash\n"
            "python scripts/generate_sample_data.py --n_samples 50000\n"
            "```"
        )
        st.stop()

    # 顶栏显示数据来源
    n_rows = len(df)
    if source == "真实数据":
        st.success(f"📦 已加载 **真实天池数据**：{n_rows:,} 行 × {df.shape[1]} 列")
    else:
        st.info(f"🧪 已加载 **合成示例数据**：{n_rows:,} 行 × {df.shape[1]} 列（来自 generate_sample_data.py）")

    return df
