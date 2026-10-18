"""
认证器
======
用户注册 / 登录 / 管理的完整业务逻辑。

要点:
    1. 密码安全: PBKDF2-HMAC-SHA256 + 随机盐（标准库实现，无额外依赖）
    2. 配置驱动: 数据库路径、密码策略、默认管理员均来自 configs/auth_config.yaml
    3. 首次初始化: 用户表为空时自动创建默认管理员
"""
from __future__ import annotations

import hashlib
import re
import secrets
from typing import Any, Dict, List, Optional

from src.auth.database import AuthDatabase
from src.utils.logger import get_logger

logger = get_logger("auth.authenticator")

# 合法用户名: 3-32 位字母/数字/下划线
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")

VALID_ROLES = ("admin", "user")


class AuthError(Exception):
    """认证业务错误（用户可见的提示信息）"""


class AuthManager:
    """
    认证管理器。

    使用示例:
        >>> manager = AuthManager(auth_config)
        >>> manager.register("alice", "pass123", role="user")
        >>> user = manager.authenticate("alice", "pass123")
    """

    def __init__(self, config):
        self.config = config
        self.db = AuthDatabase(config.database.path)
        self.min_len = int(config.password_policy.min_length)
        self.max_len = int(config.password_policy.max_length)
        self.iterations = int(config.hash.iterations)
        self.salt_length = int(config.hash.salt_length)
        self._ensure_default_admin()

    # ================= 密码哈希 =================

    def hash_password(self, password: str) -> str:
        """生成 'salt$hash' 格式的密码摘要"""
        salt = secrets.token_bytes(self.salt_length)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.iterations
        )
        return f"{salt.hex()}${digest.hex()}"

    def verify_password(self, password: str, stored: str) -> bool:
        """校验密码是否与存储摘要匹配（常数时间比较）"""
        try:
            salt_hex, hash_hex = stored.split("$", 1)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(hash_hex)
        except (ValueError, AttributeError):
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, self.iterations
        )
        return secrets.compare_digest(actual, expected)

    # ================= 校验 =================

    def _validate_username(self, username: str) -> None:
        if not isinstance(username, str) or not _USERNAME_RE.match(username):
            raise AuthError(
                "用户名需为 3-32 位字母、数字或下划线（不支持中文及特殊字符）"
            )

    def _validate_password(self, password: str) -> None:
        if not isinstance(password, str):
            raise AuthError("密码不能为空")
        if len(password) < self.min_len:
            raise AuthError(f"密码长度不能少于 {self.min_len} 位")
        if len(password) > self.max_len:
            raise AuthError(f"密码长度不能超过 {self.max_len} 位")

    @staticmethod
    def _validate_role(role: str) -> None:
        if role not in VALID_ROLES:
            raise AuthError(f"非法角色: {role}（可选: {', '.join(VALID_ROLES)}）")

    # ================= 注册 / 登录 =================

    def register(self, username: str, password: str, role: str = "user") -> Dict:
        """
        注册新用户。

        普通用户可自行注册（role 固定为 user）；
        admin 角色仅能由管理员在用户管理页创建。
        """
        self._validate_username(username)
        self._validate_password(password)
        self._validate_role(role)

        user_id = self.db.insert_user(username, self.hash_password(password), role)
        logger.info(f"新用户注册: {username} (role={role}, id={user_id})")
        return {"id": user_id, "username": username, "role": role}

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """
        登录校验。成功返回用户信息（不含密码），失败返回 None。
        """
        stored = self.db.get_password_hash(username)
        if stored is None or not self.verify_password(password, stored):
            logger.warning(f"登录失败: {username}")
            return None

        # 先更新登录时间，再取用户快照，保证返回的 last_login_at 是本次登录时间
        user = self.db.get_user_by_name(username)
        self.db.update_last_login(user["id"])
        user = self.db.get_user_by_name(username)
        logger.info(f"用户登录: {username} (role={user['role']})")
        return user

    # ================= 用户管理（管理员操作） =================

    def list_users(self) -> List[Dict[str, Any]]:
        return self.db.list_users()

    def delete_user(self, user_id: int, operator: Optional[Dict] = None) -> None:
        """删除用户；禁止删除自己（避免系统失去管理员）"""
        users = {u["id"]: u for u in self.db.list_users()}
        if user_id not in users:
            raise AuthError(f"用户不存在: id={user_id}")
        if operator and operator.get("id") == user_id:
            raise AuthError("不能删除当前登录的账号")
        # 至少保留一个管理员
        target = users[user_id]
        if target["role"] == "admin":
            admins = [u for u in users.values() if u["role"] == "admin"]
            if len(admins) <= 1:
                raise AuthError("系统至少需要保留一个管理员账号")
        self.db.delete_user(user_id)
        logger.info(
            f"删除用户: {target['username']} (操作者: "
            f"{operator['username'] if operator else 'system'})"
        )

    def update_role(self, user_id: int, role: str,
                    operator: Optional[Dict] = None) -> None:
        """修改用户角色"""
        self._validate_role(role)
        users = {u["id"]: u for u in self.db.list_users()}
        if user_id not in users:
            raise AuthError(f"用户不存在: id={user_id}")
        target = users[user_id]
        if operator and operator.get("id") == user_id:
            raise AuthError("不能修改自己的角色")
        # 降级保护：最后一个管理员不能被降为 user
        if target["role"] == "admin" and role == "user":
            admins = [u for u in users.values() if u["role"] == "admin"]
            if len(admins) <= 1:
                raise AuthError("系统至少需要保留一个管理员账号")
        self.db.update_role(user_id, role)
        logger.info(
            f"用户角色变更: {target['username']} -> {role} (操作者: "
            f"{operator['username'] if operator else 'system'})"
        )

    def reset_password(self, user_id: int, new_password: str) -> None:
        """管理员重置用户密码"""
        self._validate_password(new_password)
        if not any(u["id"] == user_id for u in self.db.list_users()):
            raise AuthError(f"用户不存在: id={user_id}")
        self.db.update_password(user_id, self.hash_password(new_password))
        logger.info(f"密码已重置: user_id={user_id}")

    # ================= 预测日志 =================

    def log_prediction(
        self,
        user: Dict,
        model_name: str,
        probability: float,
        risk_level: str,
        loan_amount: float,
        recommendation: str,
    ) -> None:
        """记录一次预测（失败不影响预测主流程，由调用方捕获异常）"""
        if not getattr(self.config, "prediction_log", None) or \
                not self.config.prediction_log.get("enabled", True):
            return
        self.db.insert_prediction_log(
            user_id=user["id"],
            username=user["username"],
            model_name=model_name,
            probability=float(probability),
            risk_level=risk_level,
            loan_amount=float(loan_amount),
            recommendation=recommendation,
        )

    def recent_prediction_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.recent_prediction_logs(limit)

    # ================= 初始化 =================

    def _ensure_default_admin(self) -> None:
        """首次启动（用户表为空）时创建默认管理员"""
        if self.db.count_users() > 0:
            return
        cfg = self.config.default_admin
        username = str(cfg.username)
        password = str(cfg.password)
        role = str(cfg.role)
        self._validate_role(role)
        user_id = self.db.insert_user(username, self.hash_password(password), role)
        logger.info(f"已创建默认管理员: {username} (id={user_id})，请尽快修改密码")
