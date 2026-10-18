"""
导航权限单元测试
================
验证按角色过滤侧边栏可见页面的逻辑（app/components/nav_registry.py）。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import AttrDict
from app.components.nav_registry import (
    PAGE_REGISTRY,
    page_def_by_name,
    role_allowed_pages,
    visible_page_defs,
)


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def auth_cfg() -> AttrDict:
    """与 configs/auth_config.yaml 权限矩阵等价的配置"""
    return AttrDict({
        "roles": {
            "admin": {
                "description": "管理员",
                "pages": [
                    "数据探索", "特征工程", "模型训练", "单笔预测", "模型解释",
                    "LLM智能助手", "系统配置", "模型监控", "A_B测试", "用户管理",
                ],
            },
            "user": {
                "description": "普通用户",
                "pages": ["单笔预测", "模型解释", "LLM智能助手"],
            },
        }
    })


def _user(role):
    return {"id": 1, "username": "tester", "role": role}


# ======================================================================
# 注册表完整性
# ======================================================================

class TestRegistry:
    def test_registry_files_exist(self):
        """注册表中每个页面文件都真实存在"""
        for p in PAGE_REGISTRY:
            assert (PROJECT_ROOT / p.file).exists(), f"页面文件缺失: {p.file}"

    def test_registry_urls_unique(self):
        urls = [p.url_path for p in PAGE_REGISTRY]
        assert len(urls) == len(set(urls))

    def test_registry_names_unique(self):
        names = [p.name for p in PAGE_REGISTRY]
        assert len(names) == len(set(names))

    def test_page_def_by_name(self):
        assert page_def_by_name("单笔预测").url_path == "prediction"
        assert page_def_by_name("不存在的页面") is None


# ======================================================================
# 角色过滤
# ======================================================================

class TestVisiblePages:
    def test_admin_sees_all(self, auth_cfg):
        pages = visible_page_defs(_user("admin"), auth_cfg)
        assert [p.name for p in pages] == [p.name for p in PAGE_REGISTRY]

    def test_user_sees_only_business_pages(self, auth_cfg):
        pages = visible_page_defs(_user("user"), auth_cfg)
        assert [p.name for p in pages] == ["单笔预测", "模型解释", "LLM智能助手"]

    def test_user_cannot_see_admin_pages(self, auth_cfg):
        names = {p.name for p in visible_page_defs(_user("user"), auth_cfg)}
        for admin_page in ("数据探索", "特征工程", "模型训练",
                           "系统配置", "模型监控", "A_B测试", "用户管理"):
            assert admin_page not in names

    def test_not_logged_in_sees_nothing(self, auth_cfg):
        assert visible_page_defs(None, auth_cfg) == []

    def test_user_dict_missing_role(self, auth_cfg):
        assert visible_page_defs({"username": "x"}, auth_cfg) == []

    def test_unknown_role_sees_nothing(self, auth_cfg):
        assert visible_page_defs(_user("superadmin"), auth_cfg) == []

    def test_order_follows_registry(self, auth_cfg):
        """过滤结果保持注册表声明顺序，而非配置文件顺序"""
        pages = visible_page_defs(_user("user"), auth_cfg)
        registry_index = {p.name: i for i, p in enumerate(PAGE_REGISTRY)}
        indexes = [registry_index[p.name] for p in pages]
        assert indexes == sorted(indexes)


# ======================================================================
# 权限矩阵读取
# ======================================================================

class TestRoleAllowedPages:
    def test_admin_pages(self, auth_cfg):
        allowed = role_allowed_pages("admin", auth_cfg)
        assert "用户管理" in allowed and "模型训练" in allowed

    def test_user_pages(self, auth_cfg):
        assert role_allowed_pages("user", auth_cfg) == [
            "单笔预测", "模型解释", "LLM智能助手"
        ]

    def test_none_role_returns_empty(self, auth_cfg):
        assert role_allowed_pages(None, auth_cfg) == []

    def test_matrix_entries_not_in_registry_are_dropped(self):
        """权限矩阵里声明了但注册表没有的页面名会被忽略（防拼写错误悄悄失效）"""
        cfg = AttrDict({"roles": {"user": {"pages": ["单笔预测", "幽灵页面"]}}})
        assert role_allowed_pages("user", cfg) == ["单笔预测"]


# ======================================================================
# 与真实 YAML 配置一致性（端到端配置校验）
# ======================================================================

class TestRealConfig:
    def test_real_auth_config_matrix(self):
        """真实 configs/auth_config.yaml 的角色矩阵可正常驱动过滤"""
        from src.utils.config_loader import ConfigLoader
        configs = ConfigLoader(str(PROJECT_ROOT / "configs"))
        auth_cfg = configs.get("auth")

        admin_pages = {p.name for p in visible_page_defs(_user("admin"), auth_cfg)}
        user_pages = {p.name for p in visible_page_defs(_user("user"), auth_cfg)}

        # 管理员能看到全部注册页面
        assert admin_pages == {p.name for p in PAGE_REGISTRY}
        # 普通用户页面是管理员的子集
        assert user_pages < admin_pages
        # 普通用户必含核心业务页
        assert {"单笔预测", "模型解释", "LLM智能助手"} <= user_pages
