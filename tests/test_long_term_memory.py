"""Tests for user-scoped long-term memory."""

from __future__ import annotations

from session import ContextManager, LongTermMemoryManager, LongTermMemoryStore, SessionStore


class FakeExtractor:
    def extract_long_term_memory(self, user_message: str) -> dict[str, str] | None:
        if "MiniRAGent" in user_message:
            return {"key": "current_agent_project", "memory": "The user's current Agent project is MiniRAGent."}
        return None


class FakeSummarizer:
    def summarize_history(self, existing_summary: str, history: str) -> str:
        return f"compressed: {history}"


def test_memory_crosses_sessions_and_deduplicates(tmp_path) -> None:
    session_store = SessionStore(tmp_path / "sessions.db")
    memory_store = LongTermMemoryStore(session_store)
    manager = LongTermMemoryManager(memory_store, FakeExtractor())

    manager.capture("user", "session-a", "My current project is called MiniRAGent.")
    manager.capture("user", "session-a", "My current project is called MiniRAGent.")

    memories = memory_store.all_for_user("user")
    assert len(memories) == 1
    assert memories[0].source_session_id == "session-a"
    recalled = manager.context_messages("user", "What is the name of my Agent project?")
    assert "MiniRAGent" in recalled[0]["content"]
    assert manager.context_messages("user", "What is today's weather in Shanghai?") == []


def test_temporary_search_is_not_saved(tmp_path) -> None:
    session_store = SessionStore(tmp_path / "sessions.db")
    memory_store = LongTermMemoryStore(session_store)
    manager = LongTermMemoryManager(memory_store, FakeExtractor())

    captured = manager.capture("user", "session-a", "Search the web for today's AI news")

    assert captured is None
    assert memory_store.all_for_user("user") == []


def test_session_compaction_does_not_change_long_term_memory(tmp_path) -> None:
    session_store = SessionStore(tmp_path / "sessions.db")
    memory_store = LongTermMemoryStore(session_store)
    memory_store.remember("user", "session-a", "project", "The project is MiniRAGent.")
    messages = []
    for number in range(5):
        messages.extend(
            [
                {"role": "user", "content": f"temporary question {number}"},
                {"role": "assistant", "content": f"temporary answer {number}"},
            ]
        )
    session_store.append_messages("user", "session-a", messages)

    ContextManager(
        session_store,
        FakeSummarizer(),
        compact_after_messages=6,
        recent_user_turns=2,
    ).build_messages("user", "session-a")

    assert session_store.state("user", "session-a").summary
    assert len(session_store.messages("user", "session-a")) == 10
    assert [m.content for m in memory_store.all_for_user("user")] == ["The project is MiniRAGent."]
