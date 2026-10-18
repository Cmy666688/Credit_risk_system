"""
页面注册与角色过滤
==================
集中定义所有页面及其路由信息，并按登录角色过滤可见页面。

要点:
    1. 配置驱动: 页面可见性来自 configs/auth_config.yaml 的 roles 权限矩阵
    2. 纯 Python: 本模块不依赖 Streamlit，可独立单元测试
    3. 顺序固定: 导航顺序在此处统一声明，与文件名前缀解耦

主入口（streamlit_app.py）用法:
    from app.components.nav_registry import visible_page_defs
    pages = [st.Page(p.file, title=p.title, icon=p.icon, url_path=p.url_path)
             for p in visible_page_defs(user, auth_cfg)]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PageDef:
    """页面定义"""
    name: str        # 权限矩阵中的页面名（与 auth_config.yaml roles.*.pages 一致）
    title: str       # 侧边栏显示标题
    icon: str        # 侧边栏图标
    file: str        # 页面脚本路径（相对项目根）
    url_path: str    # 浏览器 URL 路径（ASCII，避免中文路径问题）


# ===== 页面注册表（顺序即侧边栏顺序）=====
PAGE_REGISTRY: List[PageDef] = [
    PageDef("数据探索", "📊 数据探索", "📊", "app/views/1_📊_数据探索.py", "data-exploration"),
    PageDef("特征工程", "🔧 特征工程", "🔧", "app/views/2_🔧_特征工程.py", "feature-engineering"),
    PageDef("模型训练", "🤖 模型训练", "🤖", "app/views/3_🤖_模型训练.py", "model-training"),
    PageDef("单笔预测", "🎯 单笔预测", "🎯", "app/views/4_🎯_单笔预测.py", "prediction"),
    PageDef("模型解释", "🔍 模型解释", "🔍", "app/views/5_🔍_模型解释.py", "model-explanation"),
    PageDef("LLM智能助手", "💬 LLM智能助手", "💬", "app/views/6_💬_LLM智能助手.py", "llm-assistant"),
    PageDef("模型监控", "📈 模型监控", "📈", "app/views/8_📈_模型监控.py", "monitoring"),
    PageDef("A_B测试", "🧪 A/B测试", "🧪", "app/views/9_🧪_A_B测试.py", "ab-testing"),
    PageDef("用户管理", "👤 用户管理", "👤", "app/views/10_👤_用户管理.py", "user-management"),
    PageDef("系统配置", "⚙️ 系统配置", "⚙️", "app/views/7_⚙️_系统配置.py", "system-config"),
]

# 注册表按 name 建索引，供权限矩阵查找
_PAGES_BY_NAME = {p.name: p for p in PAGE_REGISTRY}


def role_allowed_pages(role: Optional[str], auth_cfg) -> List[str]:
    """
    读取权限矩阵中该角色可见的页面名列表。

    参数:
        role: 用户角色（admin / user），None 表示未登录
        auth_cfg: configs.get("auth") 配置对象

    返回:
        页面名列表；未登录或角色未定义时返回空列表
    """
    if role is None:
        return []
    try:
        pages = list(auth_cfg.roles[role].pages)
    except (KeyError, AttributeError):
        return []
    return [p for p in pages if p in _PAGES_BY_NAME]


def visible_page_defs(user: Optional[dict], auth_cfg) -> List[PageDef]:
    """
    当前用户可见的页面定义列表（保持注册表顺序）。

    参数:
        user: st.session_state["user"]，形如 {"id": 1, "username": "admin", "role": "admin"}
        auth_cfg: configs.get("auth") 配置对象
    """
    role = (user or {}).get("role")
    allowed = set(role_allowed_pages(role, auth_cfg))
    return [p for p in PAGE_REGISTRY if p.name in allowed]


def page_def_by_name(name: str) -> Optional[PageDef]:
    """按页面名查页面定义（测试与调试用）"""
    return _PAGES_BY_NAME.get(name)
