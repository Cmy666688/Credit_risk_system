"""
认证模块
========
用户注册 / 登录 / 角色权限管理（admin / user 两级角色，SQLite 存储）。
"""
from src.auth.authenticator import AuthError, AuthManager
from src.auth.database import AuthDatabase

__all__ = ["AuthManager", "AuthDatabase", "AuthError"]
