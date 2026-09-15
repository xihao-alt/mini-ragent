"""User-scoped long-term memory storage and selective recall."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Protocol

from .store import SessionStore


@dataclass(frozen=True)
class LongTermMemory:
    id: int
    user_id: str
    key: str
    content: str
    source_session_id: str


class MemoryExtractor(Protocol):
    def extract_long_term_memory(self, user_message: str) -> dict[str, str] | None: ...


class LongTermMemoryStore:
    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "for", "from", "how", "i",
        "in", "is", "it", "me", "my", "of", "on", "or", "s", "that", "the",
        "this", "to", "was", "what", "when", "where", "who", "with", "you", "your",
    }
    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store
        self._initialize()

    def _initialize(self) -> None:
        with self.session_store._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, memory_key),
                    UNIQUE(user_id, normalized_content)
                );

                CREATE INDEX IF NOT EXISTS idx_long_term_memory_user
                    ON long_term_memories(user_id, updated_at DESC);
                """
            )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    def remember(self, user_id: str, session_id: str, key: str, content: str) -> None:
        SessionStore._validate_ids(user_id, session_id)
        if not key.strip() or not content.strip():
            raise ValueError("Memory key and content must be non-empty")
        normalized = self._normalize(content)
        with self.session_store._connect() as connection:
            duplicate = connection.execute(
                "SELECT id FROM long_term_memories WHERE user_id = ? AND normalized_content = ?",
                (user_id, normalized),
            ).fetchone()
            if duplicate:
                connection.execute(
                    "UPDATE long_term_memories SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (duplicate["id"],),
                )
                return
            connection.execute(
                """
                INSERT INTO long_term_memories(
                    user_id, memory_key, content, normalized_content, source_session_id
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) DO UPDATE SET
                    content = excluded.content,
                    normalized_content = excluded.normalized_content,
                    source_session_id = excluded.source_session_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, key.strip(), content.strip(), normalized, session_id),
            )

    def all_for_user(self, user_id: str) -> list[LongTermMemory]:
        with self.session_store._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, memory_key, content, source_session_id
                FROM long_term_memories WHERE user_id = ? ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            LongTermMemory(row["id"], row["user_id"], row["memory_key"], row["content"], row["source_session_id"])
            for row in rows
        ]

    @staticmethod
    def _tokens(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{2,}", value.casefold())
        return {token for token in tokens if token not in LongTermMemoryStore._STOP_WORDS}

    def relevant(self, user_id: str, query: str, *, limit: int = 3) -> list[LongTermMemory]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []
        scored = []
        for memory in self.all_for_user(user_id):
            memory_tokens = self._tokens(f"{memory.key} {memory.content}")
            overlap = len(query_tokens & memory_tokens)
            substring = any(token in memory.content.casefold() for token in query_tokens if len(token) > 2)
            score = overlap * 2 + int(substring)
            if score:
                scored.append((score, memory.id, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]


class LongTermMemoryManager:
    def __init__(self, store: LongTermMemoryStore, extractor: MemoryExtractor) -> None:
        self.store = store
        self.extractor = extractor

    def capture(self, user_id: str, session_id: str, user_message: str) -> LongTermMemory | None:
        # Most questions and short chat messages are session-only. Skipping the
        # extraction LLM call here removes an invisible second request from the
        # normal chat path while explicit/stable statements are still classified.
        normalized = user_message.strip().casefold()
        memory_signals = (
            "remember", "记住", "以后", "我喜欢", "我偏好", "我的项目", "当前项目",
            "my project", "current project", "i prefer", "i like", "always", "call me",
        )
        if not any(signal in normalized for signal in memory_signals):
            return None
        candidate = self.extractor.extract_long_term_memory(user_message)
        if not candidate:
            return None
        self.store.remember(user_id, session_id, candidate["key"], candidate["memory"])
        matches = [m for m in self.store.all_for_user(user_id) if m.key == candidate["key"]]
        return matches[0] if matches else None

    def context_messages(self, user_id: str, query: str) -> list[dict[str, str]]:
        memories = self.store.relevant(user_id, query)
        if not memories:
            return []
        content = "Relevant long-term memories about this user:\n" + "\n".join(
            f"- {memory.content}" for memory in memories
        )
        return [{"role": "system", "content": content}]
