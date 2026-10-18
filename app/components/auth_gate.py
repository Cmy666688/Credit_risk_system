"""
登录门禁组件
============
所有页面共享的登录/注册表单与角色权限校验。

用法（各页面在 render_sidebar() 之后调用）:
    from app.components.auth_gate import require_login, require_admin
    require_login()    # 未登录 -> 显示登录页并中断渲染
    require_admin()    # 非管理员 -> 显示无权限提示并中断渲染
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


@st.cache_resource
def _get_auth_manager():
    """全局唯一的 AuthManager（配置变更后需重启应用生效）"""
    from src.auth.authenticator import AuthManager

    configs = st.session_state.get("configs")
    if configs is None:
        from src.utils.config_loader import ConfigLoader
        configs = ConfigLoader(str(ROOT / "configs"))
    return AuthManager(configs.get("auth"))


def get_current_user() -> dict | None:
    """当前登录用户信息: {id, username, role, ...}，未登录返回 None"""
    return st.session_state.get("user")


def is_admin() -> bool:
    user = get_current_user()
    return bool(user) and user.get("role") == "admin"


# ======================================================================
# 登录 / 注册表单
# ======================================================================

def _render_auth_form() -> None:
    """居中渲染登录/注册卡片，成功后写入 session_state 并刷新页面"""
    mgr = _get_auth_manager()

    # 居中卡片布局
    col_left, col_card, col_right = st.columns([1, 1.4, 1])
    with col_card:
        st.markdown("## 🔐 系统登录")
        st.caption("智能信贷风控分析系统 · 请登录后继续")

        tab_login, tab_register = st.tabs(["登录", "注册新用户"])

        # ---------- 登录 ----------
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input(
                    "密码", type="password", key="login_password"
                )
                submitted = st.form_submit_button("登录", type="primary",
                                                  use_container_width=True)

            if submitted:
                if not username or not password:
                    st.warning("请输入用户名和密码")
                else:
                    user = mgr.authenticate(username.strip(), password)
                    if user is None:
                        st.error("❌ 用户名或密码错误")
                    else:
                        st.session_state["user"] = user
                        st.rerun()

            st.divider()
            cfg = st.session_state.get("configs")
            try:
                default_admin_hint = cfg.get("auth").default_admin
                st.caption(
                    "💡 首次使用：默认管理员账号 "
                    f"`{default_admin_hint.username}` / `{default_admin_hint.password}`"
                    "（登录后请尽快修改密码）"
                )
            except Exception:
                pass

        # ---------- 注册 ----------
        with tab_register:
            st.info("注册后将获得**普通用户**权限，可使用最优模型进行贷款风险评估；"
                    "管理员账号由系统管理员在用户管理页创建。")
            with st.form("register_form", clear_on_submit=True):
                reg_username = st.text_input("用户名", key="reg_username",
                                             help="3-32 位字母/数字/下划线")
                reg_password = st.text_input("密码", type="password",
                                             key="reg_password")
                reg_password2 = st.text_input("确认密码", type="password",
                                              key="reg_password2")
                reg_submitted = st.form_submit_button("注册",
                                                      type="primary",
                                                      use_container_width=True)

            if reg_submitted:
                from src.auth.authenticator import AuthError

                if not reg_username or not reg_password:
                    st.warning("请填写用户名和密码")
                elif reg_password != reg_password2:
                    st.warning("两次输入的密码不一致")
                else:
                    try:
                        mgr.register(reg_username.strip(), reg_password, role="user")
                        st.success("✅ 注册成功，请切换到【登录】页签登录")
                    except AuthError as e:
                        st.error(f"❌ 注册失败：{e}")


# ======================================================================
# 权限门禁
# ======================================================================

def require_login() -> dict:
    """
    页面级登录门禁：未登录时渲染登录表单并中断页面渲染。

    返回:
        当前用户信息 dict
    """
    user = get_current_user()
    if user is None:
        _render_auth_form()
        st.stop()  # 中断后续渲染（st.stop 不会向上抛异常）
    return user


def require_admin() -> dict:
    """
    页面级管理员门禁：未登录先走登录，登录了但非管理员则提示无权限。
    """
    user = require_login()
    if user.get("role") != "admin":
        st.error("🚫 权限不足：该页面仅管理员可见。")
        st.caption("如需访问请联系系统管理员提升权限。")
        st.stop()
    return user
