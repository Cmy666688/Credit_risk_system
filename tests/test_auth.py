"""
用户认证模块单元测试
====================
覆盖: 密码哈希、注册、登录、默认管理员、用户管理、预测日志。
数据库使用 tmp_path 临时文件，不污染真实数据。
"""
import sys
from pathlib import Path

import pytest

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import AttrDict
from src.auth.authenticator import AuthError, AuthManager


# ======================================================================
# Fixtures
# ======================================================================

def _make_config(db_path) -> AttrDict:
    """构造与 configs/auth_config.yaml 等价的最小配置"""
    return AttrDict({
        "database": {"path": str(db_path)},
        "password_policy": {"min_length": 6, "max_length": 64},
        "hash": {
            "algorithm": "pbkdf2_sha256",
            "iterations": 1000,  # 测试降低迭代次数加速
            "salt_length": 16,
        },
        "default_admin": {
            "username": "admin",
            "password": "admin123",
            "role": "admin",
        },
        "prediction_log": {"enabled": True, "keep_recent": 50},
    })


@pytest.fixture
def mgr(tmp_path):
    """每个测试独立的 AuthManager（独立临时数据库）"""
    return AuthManager(_make_config(tmp_path / "test_auth.db"))


# ======================================================================
# 密码哈希
# ======================================================================

class TestPasswordHash:
    def test_hash_verify_roundtrip(self, mgr):
        stored = mgr.hash_password("secret123")
        assert mgr.verify_password("secret123", stored)

    def test_wrong_password_fails(self, mgr):
        stored = mgr.hash_password("secret123")
        assert not mgr.verify_password("wrong", stored)

    def test_salt_randomized(self, mgr):
        """同一密码两次哈希结果不同（随机盐）"""
        h1 = mgr.hash_password("secret123")
        h2 = mgr.hash_password("secret123")
        assert h1 != h2

    def test_malformed_hash_fails(self, mgr):
        assert not mgr.verify_password("secret123", "not-a-valid-hash")
        assert not mgr.verify_password("secret123", "")


# ======================================================================
# 注册
# ======================================================================

class TestRegister:
    def test_register_user(self, mgr):
        user = mgr.register("alice", "pass123", role="user")
        assert user["username"] == "alice"
        assert user["role"] == "user"
        assert isinstance(user["id"], int)

    def test_register_admin(self, mgr):
        user = mgr.register("root2", "pass123", role="admin")
        assert user["role"] == "admin"

    def test_duplicate_username_rejected(self, mgr):
        mgr.register("alice", "pass123")
        with pytest.raises(ValueError, match="已存在"):
            mgr.register("alice", "other456")

    def test_invalid_username_rejected(self, mgr):
        with pytest.raises(AuthError):
            mgr.register("ab", "pass123")            # 太短
        with pytest.raises(AuthError):
            mgr.register("非法名字", "pass123")        # 非法字符
        with pytest.raises(AuthError):
            mgr.register("a" * 33, "pass123")         # 太长

    def test_short_password_rejected(self, mgr):
        with pytest.raises(AuthError, match="少于"):
            mgr.register("alice", "12345")            # 5 位 < min_length 6

    def test_invalid_role_rejected(self, mgr):
        with pytest.raises(AuthError, match="非法角色"):
            mgr.register("alice", "pass123", role="superadmin")


# ======================================================================
# 登录
# ======================================================================

class TestAuthenticate:
    def test_default_admin_seeded(self, mgr):
        """空库初始化后自动创建默认管理员"""
        user = mgr.authenticate("admin", "admin123")
        assert user is not None
        assert user["role"] == "admin"

    def test_login_success(self, mgr):
        mgr.register("alice", "pass123")
        user = mgr.authenticate("alice", "pass123")
        assert user is not None
        assert user["username"] == "alice"
        assert "password_hash" not in user  # 不泄露密码摘要

    def test_login_wrong_password(self, mgr):
        mgr.register("alice", "pass123")
        assert mgr.authenticate("alice", "wrong99") is None

    def test_login_unknown_user(self, mgr):
        assert mgr.authenticate("nobody", "pass123") is None

    def test_last_login_updated(self, mgr):
        mgr.register("alice", "pass123")
        assert mgr.authenticate("alice", "pass123")["last_login_at"] is not None


# ======================================================================
# 用户管理
# ======================================================================

class TestUserManagement:
    def test_list_users(self, mgr):
        mgr.register("alice", "pass123")
        mgr.register("bob", "pass456")
        names = [u["username"] for u in mgr.list_users()]
        assert "admin" in names  # 默认管理员
        assert {"alice", "bob"} <= set(names)

    def test_delete_user(self, mgr):
        mgr.register("alice", "pass123")
        alice = mgr.db.get_user_by_name("alice")
        mgr.delete_user(alice["id"])
        assert mgr.db.get_user_by_name("alice") is None

    def test_cannot_delete_self(self, mgr):
        admin = mgr.db.get_user_by_name("admin")
        with pytest.raises(AuthError, match="不能删除当前登录"):
            mgr.delete_user(admin["id"], operator={"id": admin["id"], "username": "admin"})

    def test_cannot_delete_last_admin(self, mgr):
        admin = mgr.db.get_user_by_name("admin")
        with pytest.raises(AuthError, match="至少需要保留一个管理员"):
            mgr.delete_user(admin["id"], operator={"id": 999, "username": "other"})

    def test_delete_admin_allowed_when_another_exists(self, mgr):
        mgr.register("root2", "pass123", role="admin")
        mgr.register("alice", "pass123")
        alice = mgr.db.get_user_by_name("alice")
        # alice 是普通用户，直接删除不涉及管理员保护
        mgr.delete_user(alice["id"])
        root2 = mgr.db.get_user_by_name("root2")
        # 有 2 个管理员时删除其一（此时只剩 admin 默认 + 无，删除 root2 后仍有 admin）
        mgr.delete_user(root2["id"], operator={"id": 999, "username": "other"})
        assert mgr.db.get_user_by_name("root2") is None

    def test_update_role(self, mgr):
        mgr.register("alice", "pass123")
        alice = mgr.db.get_user_by_name("alice")
        mgr.update_role(alice["id"], "admin", operator={"id": 999, "username": "other"})
        assert mgr.db.get_user_by_name("alice")["role"] == "admin"

    def test_cannot_demote_last_admin(self, mgr):
        admin = mgr.db.get_user_by_name("admin")
        with pytest.raises(AuthError, match="至少需要保留一个管理员"):
            mgr.update_role(admin["id"], "user",
                            operator={"id": 999, "username": "other"})

    def test_cannot_change_own_role(self, mgr):
        admin = mgr.db.get_user_by_name("admin")
        with pytest.raises(AuthError, match="不能修改自己的角色"):
            mgr.update_role(admin["id"], "user", operator={"id": admin["id"], "username": "admin"})

    def test_reset_password(self, mgr):
        mgr.register("alice", "pass123")
        alice = mgr.db.get_user_by_name("alice")
        mgr.reset_password(alice["id"], "newpass99")
        assert mgr.authenticate("alice", "newpass99") is not None
        assert mgr.authenticate("alice", "pass123") is None

    def test_delete_user_cascades_logs(self, mgr):
        mgr.register("alice", "pass123")
        alice = mgr.authenticate("alice", "pass123")
        mgr.log_prediction(alice, "Ensemble", 0.42, "中风险", 30000, "⚠️ 人工复核")
        assert len(mgr.recent_prediction_logs()) == 1
        mgr.delete_user(alice["id"])
        assert len(mgr.recent_prediction_logs()) == 0  # 级联删除


# ======================================================================
# 预测日志
# ======================================================================

class TestPredictionLog:
    def test_log_and_query(self, mgr):
        user = mgr.authenticate("admin", "admin123")
        mgr.log_prediction(user, "Ensemble 融合模型", 0.42, "中风险", 30000, "⚠️ 人工复核")
        logs = mgr.recent_prediction_logs()
        assert len(logs) == 1
        assert logs[0]["username"] == "admin"
        assert logs[0]["probability"] == pytest.approx(0.42)

    def test_logs_ordered_desc(self, mgr):
        user = mgr.authenticate("admin", "admin123")
        for i in range(5):
            mgr.log_prediction(user, "XGBoost", 0.1 * i, "低风险", 10000, "✅ 建议通过")
        logs = mgr.recent_prediction_logs(limit=3)
        assert len(logs) == 3
        # 倒序：最新的 id 最大
        assert logs[0]["id"] > logs[-1]["id"]

    def test_log_disabled(self, tmp_path):
        cfg = _make_config(tmp_path / "off.db")
        cfg["prediction_log"]["enabled"] = False
        m = AuthManager(cfg)
        user = m.authenticate("admin", "admin123")
        m.log_prediction(user, "XGBoost", 0.5, "低风险", 10000, "✅ 建议通过")
        assert m.recent_prediction_logs() == []


# ======================================================================
# 数据库
# ======================================================================

class TestDatabase:
    def test_db_file_created(self, mgr, tmp_path):
        # mgr fixture 使用的是独立 tmp_path，直接检查文件存在
        db_files = list(tmp_path.glob("*.db"))
        assert len(db_files) == 1

    def test_tables_initialized(self, mgr):
        import sqlite3
        conn = sqlite3.connect(str(mgr.db.db_path))
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert {"users", "prediction_logs"} <= tables
