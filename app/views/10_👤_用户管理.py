"""
👤 用户管理页面
================
管理员专属：用户增删改、角色分配、预测行为审计。

功能:
    1. 用户概览（总数/管理员/普通用户/今日预测）
    2. 新增用户（可指定管理员/普通用户角色）
    3. 用户列表：修改角色 / 重置密码 / 删除
    4. 预测审计日志（所有用户的模型调用记录）
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="用户管理", page_icon="👤", layout="wide")
render_sidebar()

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin, _get_auth_manager
from src.auth.authenticator import AuthError

current_user = require_admin()
mgr = _get_auth_manager()

st.title("👤 用户与权限管理")
st.caption("管理员专属页面：管理账号、分配角色、审计预测行为")

# ======================================================================
# 用户概览
# ======================================================================
users = mgr.list_users()
logs = mgr.recent_prediction_logs(
    limit=int(mgr.config.prediction_log.get("keep_recent", 50))
)

n_admin = sum(1 for u in users if u["role"] == "admin")
n_user = len(users) - n_admin
today = pd.Timestamp.now().strftime("%Y-%m-%d")
n_today_logs = sum(
    1 for log in logs if str(log.get("created_at", "")).startswith(today)
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 用户总数", len(users))
with col2:
    st.metric("🛡️ 管理员", n_admin)
with col3:
    st.metric("👤 普通用户", n_user)
with col4:
    st.metric("🔮 今日预测次数", n_today_logs)

st.divider()

# ======================================================================
# 新增用户
# ======================================================================
st.markdown("## ➕ 新增用户")

with st.form("create_user_form", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        new_username = st.text_input("用户名", help="3-32 位字母/数字/下划线")
    with c2:
        new_password = st.text_input(
            "初始密码", type="password", help=f"至少 {mgr.min_len} 位"
        )
    with c3:
        new_role = st.selectbox(
            "角色",
            options=["user", "admin"],
            format_func=lambda r: "🛡️ 管理员" if r == "admin" else "👤 普通用户",
        )
    create_submitted = st.form_submit_button("创建用户", type="primary")

if create_submitted:
    try:
        mgr.register(new_username.strip(), new_password, role=new_role)
        st.success(f"✅ 用户 `{new_username}` 创建成功（角色: {new_role}）")
        st.rerun()
    except AuthError as e:
        st.error(f"❌ 创建失败：{e}")

st.divider()

# ======================================================================
# 用户列表
# ======================================================================
st.markdown("## 📋 用户列表")

if not users:
    st.info("暂无用户")
else:
    df_users = pd.DataFrame(users).rename(
        columns={
            "id": "ID",
            "username": "用户名",
            "role": "角色",
            "created_at": "创建时间",
            "last_login_at": "最后登录",
        }
    )
    df_users["角色"] = df_users["角色"].map(
        lambda r: "🛡️ 管理员" if r == "admin" else "👤 普通用户"
    )
    st.dataframe(df_users, use_container_width=True, hide_index=True)

    # ---- 单个用户管理操作 ----
    st.markdown("### 🔧 管理操作")
    user_options = {
        f"{u['username']}（{u['role']}）": u for u in users
    }
    selected_label = st.selectbox("选择用户", options=list(user_options.keys()))
    selected = user_options[selected_label]

    tab_role, tab_pwd, tab_del = st.tabs(
        ["🔄 修改角色", "🔑 重置密码", "🗑️ 删除用户"]
    )

    with tab_role:
        new_role_choice = st.radio(
            "目标角色",
            options=["user", "admin"],
            format_func=lambda r: "🛡️ 管理员" if r == "admin" else "👤 普通用户",
            horizontal=True,
            index=0 if selected["role"] == "admin" else 1,
        )
        if st.button("确认修改角色", type="primary"):
            try:
                mgr.update_role(selected["id"], new_role_choice, operator=current_user)
                st.success(f"✅ {selected['username']} 角色已改为 {new_role_choice}")
                st.rerun()
            except AuthError as e:
                st.error(f"❌ {e}")

    with tab_pwd:
        reset_pwd = st.text_input(
            "新密码", type="password", help=f"至少 {mgr.min_len} 位"
        )
        if st.button("确认重置密码", type="primary"):
            try:
                mgr.reset_password(selected["id"], reset_pwd)
                st.success(f"✅ {selected['username']} 的密码已重置")
            except AuthError as e:
                st.error(f"❌ {e}")

    with tab_del:
        is_self = selected["id"] == current_user["id"]
        st.warning(
            f"将删除用户 **{selected['username']}** 及其全部预测日志，该操作不可恢复。"
            + ("（不能删除当前登录账号）" if is_self else "")
        )
        if st.button("确认删除", type="primary", disabled=is_self):
            try:
                mgr.delete_user(selected["id"], operator=current_user)
                st.success(f"✅ 用户 {selected['username']} 已删除")
                st.rerun()
            except AuthError as e:
                st.error(f"❌ {e}")

st.divider()

# ======================================================================
# 预测审计日志
# ======================================================================
st.markdown("## 🔮 预测行为审计")

if not logs:
    st.info("暂无预测记录。用户在【单笔预测】页提交评估后会在此留痕。")
else:
    df_logs = pd.DataFrame(logs).rename(
        columns={
            "id": "日志ID",
            "username": "用户",
            "model_name": "使用模型",
            "probability": "违约概率",
            "risk_level": "风险等级",
            "loan_amount": "贷款金额",
            "recommendation": "审批建议",
            "created_at": "时间",
        }
    )
    df_logs["违约概率"] = df_logs["违约概率"].map(
        lambda p: f"{p * 100:.2f}%" if pd.notna(p) else "-"
    )
    df_logs["贷款金额"] = df_logs["贷款金额"].map(
        lambda v: f"¥{v:,.0f}" if pd.notna(v) else "-"
    )
    st.dataframe(df_logs, use_container_width=True, hide_index=True)
