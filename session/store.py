"""SQLite persistence for complete per-user, per-session transcripts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StoredMessage:
    id: int
    user_id: str
    session_id: str
    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None

    def to_llm_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_name:
            message["tool_calls"] = [
                {
                    "id": self.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(self.tool_arguments or {}, ensure_ascii=False),
                    },
                }
            ]
        elif self.role == "tool":
            message["tool_call_id"] = self.tool_call_id
            if self.tool_name:
                message["name"] = self.tool_name
        return message


@dataclass(frozen=True)
class SessionState:
    user_id: str
    session_id: str
    summary: str
    summarized_through_id: int
    status: str


@dataclass(frozen=True)
class SessionInfo:
    user_id: str
    session_id: str
    title: str
    created_at: str
    updated_at: str


class SessionStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_through_id INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    custom_title TEXT,
                    status TEXT NOT NULL DEFAULT 'BOT',
                    PRIMARY KEY (user_id, session_id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_name TEXT,
                    tool_arguments TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id, session_id)
                        REFERENCES sessions(user_id, session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(user_id, session_id, id);

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product TEXT NOT NULL,
                    order_status TEXT NOT NULL,
                    payment_status TEXT NOT NULL,
                    completed_charge_count INTEGER NOT NULL DEFAULT 0,
                    shipping_status TEXT NOT NULL,
                    estimated_delivery TEXT,
                    tracking_number TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
                """
            )
            connection.executemany(
                """INSERT OR IGNORE INTO orders (
                    order_id, user_id, product, order_status, payment_status,
                    completed_charge_count, shipping_status, estimated_delivery,
                    tracking_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ("NS001", "user_001", "NovaSound Earbuds", "Shipped", "duplicate_completed", 2, "delayed", "2026-09-20", "NV100001"),
                    ("NS002", "user_001", "Smart Desk Lamp", "Shipped", "paid", 1, "in_transit", "2026-09-18", "NV100002"),
                    ("NS003", "user_001", "NovaHome Blender", "Packing", "paid", 1, "not_dispatched", None, None),
                    ("NS004", "user_001", "Travel Backpack", "Delivered", "paid", 1, "delivered", "2026-09-10", "NV100004"),
                    ("NS005", "user_001", "Ceramic Mug Set", "Delivered", "paid", 1, "delivered_damaged", "2026-09-11", "NV100005"),
                    ("NS006", "user_001", "Cotton Shirt", "Refunded", "refunded", 1, "returned", "2026-09-05", "NV100006"),
                    ("NS007", "user_001", "Phone Case", "Cancelled", "authorization_released", 0, "not_dispatched", None, None),
                    ("NS008", "user_002", "Gaming Mouse", "Shipped", "paid", 1, "in_transit", "2026-09-19", "NV200008"),
                    ("NS009", "user_002", "Digital Gift Card", "Completed", "paid", 1, "digital_delivered", "2026-09-12", None),
                ),
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
            if "custom_title" not in columns:
                connection.execute("ALTER TABLE sessions ADD COLUMN custom_title TEXT")
            if "status" not in columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'BOT'"
                )

    @staticmethod
    def _validate_ids(user_id: str, session_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")

    def ensure_session(self, user_id: str, session_id: str) -> None:
        self._validate_ids(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions(user_id, session_id) VALUES (?, ?)",
                (user_id, session_id),
            )

    def list_sessions(self, user_id: str) -> list[SessionInfo]:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.user_id, s.session_id, s.created_at, s.updated_at,
                       COALESCE(NULLIF(s.custom_title, ''),
                           (SELECT substr(m.content, 1, 60) FROM messages m
                            WHERE m.user_id = s.user_id AND m.session_id = s.session_id
                              AND m.role = 'user' ORDER BY m.id LIMIT 1),
                           'New chat'
                       ) AS title
                FROM sessions s WHERE s.user_id = ?
                ORDER BY s.updated_at DESC, s.created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [SessionInfo(**dict(row)) for row in rows]

    def rename_session(self, user_id: str, session_id: str, title: str) -> bool:
        self._validate_ids(user_id, session_id)
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title must be non-empty")
        if len(clean_title) > 80:
            raise ValueError("title must be at most 80 characters")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET custom_title = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND session_id = ?",
                (clean_title, user_id, session_id),
            )
        return cursor.rowcount > 0

    def delete_session(self, user_id: str, session_id: str) -> bool:
        self._validate_ids(user_id, session_id)
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                "DELETE FROM messages WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            connection.execute(
                "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
        return True

    def append_messages(
        self,
        user_id: str,
        session_id: str,
        messages: Iterable[dict[str, Any]],
    ) -> None:
        self.ensure_session(user_id, session_id)
        rows = []
        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            call_id = message.get("tool_call_id")
            tool_name = message.get("name")
            arguments = None
            tool_calls = message.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                call = tool_calls[0]
                call_id = call.get("id")
                function = call.get("function") or {}
                tool_name = function.get("name")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": function.get("arguments")}
            rows.append(
                (
                    user_id,
                    session_id,
                    role,
                    None if content is None else str(content),
                    call_id,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False) if arguments is not None else None,
                )
            )

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO messages(
                    user_id, session_id, role, content, tool_call_id, tool_name, tool_arguments
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )

    def messages(self, user_id: str, session_id: str, *, after_id: int = 0) -> list[StoredMessage]:
        self._validate_ids(user_id, session_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, session_id, role, content, tool_call_id,
                       tool_name, tool_arguments
                FROM messages
                WHERE user_id = ? AND session_id = ? AND id > ?
                ORDER BY id
                """,
                (user_id, session_id, after_id),
            ).fetchall()
        return [
            StoredMessage(
                id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                tool_call_id=row["tool_call_id"],
                tool_name=row["tool_name"],
                tool_arguments=json.loads(row["tool_arguments"]) if row["tool_arguments"] else None,
            )
            for row in rows
        ]

    def state(self, user_id: str, session_id: str) -> SessionState:
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary, summarized_through_id, status FROM sessions "
                "WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            ).fetchone()
        return SessionState(
            user_id,
            session_id,
            row["summary"],
            row["summarized_through_id"],
            row["status"],
        )

    def set_status(self, user_id: str, session_id: str, status: str) -> None:
        if status not in {"BOT", "WAITING_HUMAN", "HUMAN_ACTIVE"}:
            raise ValueError(f"Unknown session status: {status}")
        self.ensure_session(user_id, session_id)
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET status = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND session_id = ?",
                (status, user_id, session_id),
            )

    def update_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        through_id: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET summary = ?, summarized_through_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                """,
                (summary, through_id, user_id, session_id),
            )
