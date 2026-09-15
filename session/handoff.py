"""SQLite-backed simulated human-support queue."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .store import SessionStore


@dataclass(frozen=True)
class HumanHandoff:
    handoff_id: str
    user_id: str
    session_id: str
    reason: str
    issue_summary: str
    priority: str
    status: str
    created_at: str
    claimed_at: str | None
    closed_at: str | None


@dataclass(frozen=True)
class HumanMessage:
    id: int
    handoff_id: str
    sender: str
    content: str
    created_at: str


class HandoffStore:
    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store
        with self.session_store._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS human_handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    issue_summary TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    claimed_at TEXT,
                    closed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_handoffs_user
                    ON human_handoffs(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS human_handoff_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    handoff_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (handoff_id) REFERENCES human_handoffs(handoff_id)
                );
                CREATE INDEX IF NOT EXISTS idx_handoff_messages
                    ON human_handoff_messages(handoff_id, id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(human_handoffs)")
            }
            if "claimed_at" not in columns:
                connection.execute("ALTER TABLE human_handoffs ADD COLUMN claimed_at TEXT")
            if "closed_at" not in columns:
                connection.execute("ALTER TABLE human_handoffs ADD COLUMN closed_at TEXT")

    def create(
        self,
        user_id: str,
        session_id: str,
        reason: str,
        issue_summary: str,
        priority: str,
    ) -> HumanHandoff:
        handoff_id = f"ho_{uuid4().hex}"
        with self.session_store._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO sessions(user_id, session_id) VALUES (?, ?)",
                (user_id, session_id),
            )
            connection.execute(
                """
                INSERT INTO human_handoffs(
                    handoff_id, user_id, session_id, reason, issue_summary, priority
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (handoff_id, user_id, session_id, reason, issue_summary, priority),
            )
            connection.execute(
                "UPDATE sessions SET status = 'WAITING_HUMAN', "
                "updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            row = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
        return HumanHandoff(**dict(row))

    def list(
        self,
        *,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[HumanHandoff]:
        query = "SELECT * FROM human_handoffs"
        conditions: list[str] = []
        values: list[str] = []
        if user_id:
            conditions.append("user_id = ?")
            values.append(user_id)
        if status:
            conditions.append("status = ?")
            values.append(status)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += """
            ORDER BY CASE priority
                WHEN 'urgent' THEN 4
                WHEN 'high' THEN 3
                WHEN 'normal' THEN 2
                WHEN 'low' THEN 1
                ELSE 0
            END DESC, created_at ASC, handoff_id ASC
        """
        with self.session_store._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [HumanHandoff(**dict(row)) for row in rows]

    def pending_count(self, *, user_id: str | None = None) -> int:
        query = "SELECT COUNT(*) AS count FROM human_handoffs WHERE status = 'pending'"
        values: tuple[str, ...] = ()
        if user_id:
            query += " AND user_id = ?"
            values = (user_id,)
        with self.session_store._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return int(row["count"])

    def get(self, handoff_id: str) -> HumanHandoff | None:
        with self.session_store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
        return HumanHandoff(**dict(row)) if row else None

    def current_for_session(self, user_id: str, session_id: str) -> HumanHandoff | None:
        with self.session_store._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM human_handoffs
                WHERE user_id = ? AND session_id = ? AND status IN ('pending', 'active')
                ORDER BY created_at DESC, handoff_id DESC LIMIT 1
                """,
                (user_id, session_id),
            ).fetchone()
        return HumanHandoff(**dict(row)) if row else None

    def add_message(self, handoff_id: str, sender: str, content: str) -> HumanMessage:
        if sender not in {"customer", "human"}:
            raise ValueError("sender must be customer or human")
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("message content must be non-empty")
        with self.session_store._connect() as connection:
            handoff = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
            if handoff is None:
                raise LookupError("Handoff not found")
            if handoff["status"] != "active":
                raise ValueError("Human conversation is not active")
            cursor = connection.execute(
                "INSERT INTO human_handoff_messages(handoff_id, sender, content) "
                "VALUES (?, ?, ?)",
                (handoff_id, sender, clean_content),
            )
            row = connection.execute(
                "SELECT * FROM human_handoff_messages WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return HumanMessage(**dict(row))

    def messages(self, handoff_id: str, *, after_id: int = 0) -> list[HumanMessage]:
        with self.session_store._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM human_handoff_messages "
                "WHERE handoff_id = ? AND id > ? ORDER BY id",
                (handoff_id, after_id),
            ).fetchall()
        return [HumanMessage(**dict(row)) for row in rows]

    def claim(self, handoff_id: str) -> HumanHandoff | None:
        with self.session_store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None:
                return None
            updated_status = connection.execute(
                "UPDATE human_handoffs SET status = 'active', "
                "claimed_at = CURRENT_TIMESTAMP WHERE handoff_id = ? AND status = 'pending'",
                (handoff_id,),
            )
            if updated_status.rowcount != 1:
                return None
            connection.execute(
                "UPDATE sessions SET status = 'HUMAN_ACTIVE', updated_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND session_id = ?",
                (row["user_id"], row["session_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
        return HumanHandoff(**dict(updated))

    def close(self, handoff_id: str) -> HumanHandoff | None:
        with self.session_store._connect() as connection:
            row = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None:
                return None
            updated_status = connection.execute(
                "UPDATE human_handoffs SET status = 'closed', "
                "closed_at = CURRENT_TIMESTAMP WHERE handoff_id = ? AND status = 'active'",
                (handoff_id,),
            )
            if updated_status.rowcount != 1:
                return None
            connection.execute(
                "UPDATE sessions SET status = 'BOT', updated_at = CURRENT_TIMESTAMP "
                "WHERE user_id = ? AND session_id = ?",
                (row["user_id"], row["session_id"]),
            )
            updated = connection.execute(
                "SELECT * FROM human_handoffs WHERE handoff_id = ?", (handoff_id,)
            ).fetchone()
        return HumanHandoff(**dict(updated))
