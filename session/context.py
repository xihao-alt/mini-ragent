"""Build bounded LLM context while retaining complete SQLite history."""

from __future__ import annotations

import json
from typing import Protocol

from .store import SessionStore, StoredMessage


class HistorySummarizer(Protocol):
    def summarize_history(self, existing_summary: str, history: str) -> str: ...


class ContextManager:
    def __init__(
        self,
        store: SessionStore,
        summarizer: HistorySummarizer,
        *,
        compact_after_messages: int = 16,
        recent_user_turns: int = 4,
    ) -> None:
        if compact_after_messages < 2 or recent_user_turns < 1:
            raise ValueError("Context limits must be positive")
        self.store = store
        self.summarizer = summarizer
        self.compact_after_messages = compact_after_messages
        self.recent_user_turns = recent_user_turns

    def _recent_start(self, messages: list[StoredMessage]) -> int:
        user_indexes = [index for index, item in enumerate(messages) if item.role == "user"]
        if len(user_indexes) <= self.recent_user_turns:
            return 0
        return user_indexes[-self.recent_user_turns]

    @staticmethod
    def _summary_text(messages: list[StoredMessage]) -> str:
        entries = []
        for item in messages:
            detail = item.content or ""
            if item.tool_name:
                detail = f"{item.tool_name} {json.dumps(item.tool_arguments or {}, ensure_ascii=False)} {detail}"
            entries.append(f"{item.role}: {detail}")
        return "\n".join(entries)

    def build_messages(self, user_id: str, session_id: str) -> list[dict[str, object]]:
        state = self.store.state(user_id, session_id)
        unsummarized = self.store.messages(
            user_id,
            session_id,
            after_id=state.summarized_through_id,
        )

        if len(unsummarized) > self.compact_after_messages:
            cutoff = self._recent_start(unsummarized)
            older = unsummarized[:cutoff]
            if older:
                summary = self.summarizer.summarize_history(
                    state.summary,
                    self._summary_text(older),
                )
                self.store.update_summary(user_id, session_id, summary, older[-1].id)
                state = self.store.state(user_id, session_id)
                unsummarized = unsummarized[cutoff:]

        context: list[dict[str, object]] = []
        if state.summary:
            context.append(
                {
                    "role": "system",
                    "content": f"Summary of earlier messages in this session:\n{state.summary}",
                }
            )
        context.extend(item.to_llm_message() for item in unsummarized)
        return context
