"""数据源适配器（扩展点 3：添加新数据源）

把外部数据集的 schema 归一化成天池 schema，使下游特征工程/模型训练
无需感知数据来源差异。新增数据源时，在这里注册即可被 DataLoader 自动套用。
"""
from src.data.adapters.lending_club_adapter import LendingClubAdapter

ADAPTER_REGISTRY = {
    "lending_club": LendingClubAdapter,
}

__all__ = ["LendingClubAdapter", "ADAPTER_REGISTRY"]
