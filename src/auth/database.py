"""
认证数据库
==========
基于 SQLite 的用户与预测日志存储。

要点:
    1. 零依赖: 使用 Python 标准库 sqlite3，无需安装数据库服务
    2. 自动建表: 首次连接自动创建 users / prediction_logs 表
    3. 外键级联: 删除用户时同步清理其预测日志
    4. 连接即用即关: 每个操作独立事务，避免 Streamlit 多线程共享连接的问题
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger import get_logger

logger = get_logger("auth.database")

# 建表语句（IF NOT EXISTS 保证幂等）
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'user' CHECK(role IN ('admin', 'user')),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL,
    username       TEXT    NOT NULL,
    model_name     TEXT,
    probability    REAL,
    risk_level     TEXT,
    loan_amount    REAL,
    recommendation TEXT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_user ON prediction_logs (user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_time ON prediction_logs (created_at);
"""


class AuthDatabase:
    """SQLite 认证数据库封装。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"认证数据库就绪: {self.db_path}")

    def _init_schema(self) -> None:
        """建库建表（幂等）"""
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        """获取连接的上下文管理器：自动提交/回滚/关闭"""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------- 用户表 ----------------

    def insert_user(
        self, username: str, password_hash: str, role: str = "user"
    ) -> int:
        """新增用户，返回新用户 id；用户名重复时抛 ValueError"""
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, password_hash, role),
                )
                return int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError(f"用户名已存在: {username}") from e

    def get_user_by_name(self, username: str) -> Dict[str, Any] | None:
        """按用户名查询用户（不含密码哈希）"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, username, role, created_at, last_login_at "
                "FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def get_password_hash(self, username: str) -> str | None:
        """查询用户密码哈希（仅登录校验用）"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
        return row["password_hash"] if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """全部用户列表（不含密码哈希）"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, username, role, created_at, last_login_at "
                "FROM users ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_users(self) -> int:
        """用户总数（用于判断是否需要初始化默认管理员）"""
        with self._conn() as conn:
            return int(
                conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
            )

    def update_last_login(self, user_id: int) -> None:
        """更新最后登录时间"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = datetime('now', 'localtime') "
                "WHERE id = ?",
                (user_id,),
            )

    def update_password(self, user_id: int, password_hash: str) -> None:
        """更新密码哈希"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id),
            )

    def update_role(self, user_id: int, role: str) -> None:
        """更新角色"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET role = ? WHERE id = ?", (role, user_id)
            )

    def delete_user(self, user_id: int) -> None:
        """删除用户（预测日志级联删除）"""
        with self._conn() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # ---------------- 预测日志表 ----------------

    def insert_prediction_log(
        self,
        user_id: int,
        username: str,
        model_name: str,
        probability: float,
        risk_level: str,
        loan_amount: float,
        recommendation: str,
    ) -> None:
        """记录一次预测行为（审计用）"""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO prediction_logs "
                "(user_id, username, model_name, probability, risk_level, "
                " loan_amount, recommendation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, username, model_name, probability, risk_level,
                 loan_amount, recommendation),
            )

    def recent_prediction_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """最近预测日志（按时间倒序）"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, username, model_name, probability, risk_level, "
                "loan_amount, recommendation, created_at "
                "FROM prediction_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
