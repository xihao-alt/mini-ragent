"""Tests for SQLite sessions, context compaction, and isolation."""

from __future__ import annotations

from agent import AgentRunResult
from session import ContextManager, HandoffStore, SessionManager, SessionStore


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def summarize_history(self, existing_summary: str, history: str) -> str:
        self.calls.append((existing_summary, history))
        return f"summary:{history}"


class CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, message: str, *, history_messages=None) -> AgentRunResult:
        self.calls += 1
        return AgentRunResult("bot answer", [], None, [])


class EmptyContext:
    def build_messages(self, user_id: str, session_id: str) -> list[dict[str, object]]:
        return []


def test_store_isolates_user_and_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.append_messages("user-1", "session-a", [{"role": "user", "content": "alpha"}])
    store.append_messages("user-1", "session-b", [{"role": "user", "content": "beta"}])
    store.append_messages("user-2", "session-a", [{"role": "user", "content": "gamma"}])

    assert [m.content for m in store.messages("user-1", "session-a")] == ["alpha"]
    assert [m.content for m in store.messages("user-1", "session-b")] == ["beta"]
    assert [m.content for m in store.messages("user-2", "session-a")] == ["gamma"]


def test_tool_messages_round_trip(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "result"},
    ]
    store.append_messages("user", "session", messages)

    loaded = [item.to_llm_message() for item in store.messages("user", "session")]
    assert loaded == messages


def test_context_compacts_old_messages_without_deleting_them(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    original = []
    for number in range(6):
        original.extend(
            [
                {"role": "user", "content": f"question {number}"},
                {"role": "assistant", "content": f"answer {number}"},
            ]
        )
    store.append_messages("user", "session", original)
    summarizer = FakeSummarizer()
    context = ContextManager(
        store,
        summarizer,
        compact_after_messages=8,
        recent_user_turns=2,
    ).build_messages("user", "session")

    assert context[0]["role"] == "system"
    assert "summary:" in str(context[0]["content"])
    assert [message["content"] for message in context[1:] if message["role"] == "user"] == [
        "question 4",
        "question 5",
    ]
    assert len(store.messages("user", "session")) == len(original)
    assert store.state("user", "session").summarized_through_id > 0


def test_human_active_message_is_persisted_without_calling_llm(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    handoffs = HandoffStore(store)
    handoff = handoffs.create("user", "session", "requested", "Needs help", "normal")
    handoffs.claim(handoff.handoff_id)
    runtime = CountingRuntime()
    manager = SessionManager(runtime, store, EmptyContext(), handoffs=handoffs)

    result = manager.chat("user", "session", "I have another detail for the agent")

    assert runtime.calls == 0
    assert result.final_answer == "Your message was added to the human support conversation."
    assert [message.content for message in store.messages("user", "session")] == [
        "I have another detail for the agent"
    ]
    assert [message.content for message in handoffs.messages(handoff.handoff_id)] == [
        "I have another detail for the agent"
    ]
