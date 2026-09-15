"""FastAPI contract tests with lightweight injected services."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent import AgentRunResult, AgentStep
from app import ApplicationServices, create_app
from llm import LLMDecision
from session import HandoffStore, LongTermMemoryStore, SessionStore
from tests.test_pdf_loader import create_test_pdf
from tools import ToolRegistry


class FakeSessions:
    def chat(self, user_id: str, session_id: str, message: str) -> AgentRunResult:
        return AgentRunResult(
            "Human support is pending.",
            [
                AgentStep(
                    1,
                    LLMDecision.tool_call("handoff_to_human", {}, "call_1"),
                    tool_name="handoff_to_human",
                    tool_arguments={"reason": "requested"},
                    tool_result={"handoff_id": "ho_test", "status": "pending"},
                ),
                AgentStep(2, LLMDecision.final("Human support is pending."), final_answer="Human support is pending."),
            ],
        )


class FakeKnowledge:
    def __init__(self, docs_dir: Path) -> None:
        self.docs_dir = docs_dir
        self.docs_dir.mkdir()

    def documents(self) -> list[str]:
        return [path.name for path in self.docs_dir.glob("*.pdf")]

    def rebuild(self) -> int:
        return 3


def test_session_and_chat_endpoints(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    services = ApplicationServices(
        sessions=FakeSessions(),
        store=store,
        handoffs=HandoffStore(store),
        registry=ToolRegistry(),
        knowledge=FakeKnowledge(tmp_path / "docs"),
    )
    client = TestClient(create_app(services))

    created = client.post("/api/sessions", json={"user_id": "user-1"})
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    store.append_messages("user-1", session_id, [{"role": "user", "content": "Hello"}])

    assert client.get("/api/sessions", params={"user_id": "user-1"}).json()[0]["title"] == "Hello"
    assert client.get(
        f"/api/sessions/{session_id}/messages", params={"user_id": "user-1"}
    ).json()[0]["content"] == "Hello"

    renamed = client.patch(
        f"/api/sessions/{session_id}",
        json={"user_id": "user-1", "title": "Renamed chat"},
    )
    assert renamed.status_code == 200
    assert client.get("/api/sessions", params={"user_id": "user-1"}).json()[0]["title"] == "Renamed chat"

    chat = client.post(
        "/api/chat",
        json={"user_id": "user-1", "session_id": session_id, "message": "Human please"},
    )
    assert chat.status_code == 200
    assert chat.json()["handoff"] is True
    assert chat.json()["handoff_id"] == "ho_test"
    assert len(chat.json()["trace"]) == 2


def test_delete_session_removes_messages_but_not_user_memory(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    services = ApplicationServices(
        sessions=FakeSessions(), store=store, handoffs=HandoffStore(store),
        registry=ToolRegistry(), knowledge=FakeKnowledge(tmp_path / "docs"),
    )
    client = TestClient(create_app(services))
    store.append_messages("user-1", "session-a", [{"role": "user", "content": "Hello"}])
    memories = LongTermMemoryStore(store)
    memories.remember("user-1", "session-a", "preference", "The user likes Python.")
    response = client.delete("/api/sessions/session-a", params={"user_id": "user-1"})
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert store.list_sessions("user-1") == []
    assert store.messages("user-1", "session-a") == []
    assert memories.all_for_user("user-1")[0].content == "The user likes Python."


def test_pdf_upload_endpoint(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    knowledge = FakeKnowledge(tmp_path / "docs")
    services = ApplicationServices(
        sessions=FakeSessions(),
        store=store,
        handoffs=HandoffStore(store),
        registry=ToolRegistry(),
        knowledge=knowledge,
    )
    client = TestClient(create_app(services))
    valid_pdf = tmp_path / "valid.pdf"
    create_test_pdf(valid_pdf)

    response = client.post(
        "/api/upload",
        files={"file": ("guide.pdf", valid_pdf.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "guide.pdf"
    assert response.json()["chunk_count"] == 3


def test_chat_trace_is_bounded(tmp_path) -> None:
    class LongResultSessions(FakeSessions):
        def chat(self, user_id: str, session_id: str, message: str) -> AgentRunResult:
            return AgentRunResult(
                "Done",
                [
                    AgentStep(
                        1,
                        LLMDecision.tool_call("web_search", {}, "call_1"),
                        tool_name="web_search",
                        tool_result=[{"content": "x" * 2000}],
                    ),
                    AgentStep(2, LLMDecision.final("Done"), final_answer="Done"),
                ],
            )

    store = SessionStore(tmp_path / "sessions.db")
    services = ApplicationServices(
        sessions=LongResultSessions(),
        store=store,
        handoffs=HandoffStore(store),
        registry=ToolRegistry(),
        knowledge=FakeKnowledge(tmp_path / "docs"),
    )
    response = TestClient(create_app(services)).post(
        "/api/chat",
        json={"user_id": "u", "session_id": "s", "message": "search"},
    )
    content = response.json()["trace"][0]["tool_result"][0]["content"]
    assert len(content) < 700
    assert content.endswith("…")


def test_handoff_queue_claim_and_close_endpoints(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    handoffs = HandoffStore(store)
    low = handoffs.create("low-user", "low-session", "reason", "issue", "low")
    urgent = handoffs.create("urgent-user", "urgent-session", "reason", "issue", "urgent")
    high = handoffs.create("high-user", "high-session", "reason", "issue", "high")
    services = ApplicationServices(
        sessions=FakeSessions(), store=store, handoffs=handoffs,
        registry=ToolRegistry(), knowledge=FakeKnowledge(tmp_path / "docs"),
    )
    client = TestClient(create_app(services))

    queue = client.get("/api/handoffs").json()
    assert queue["pending_count"] == 3
    assert [item["handoff_id"] for item in queue["items"]] == [
        urgent.handoff_id, high.handoff_id, low.handoff_id,
    ]
    claimed = client.post(f"/api/handoffs/{urgent.handoff_id}/claim")
    assert claimed.status_code == 200 and claimed.json()["status"] == "active"
    assert client.get("/api/handoffs").json()["pending_count"] == 2

    current = client.get(
        "/api/handoffs/current",
        params={"user_id": "urgent-user", "session_id": "urgent-session"},
    )
    assert current.json()["handoff"]["handoff_id"] == urgent.handoff_id
    customer = client.post(
        f"/api/handoffs/{urgent.handoff_id}/customer-message",
        json={
            "user_id": "urgent-user",
            "session_id": "urgent-session",
            "content": "My order number is NS-100.",
        },
    )
    assert customer.status_code == 200 and customer.json()["sender"] == "customer"
    human = client.post(
        f"/api/handoffs/{urgent.handoff_id}/reply",
        json={"content": "I am checking order NS-100 now."},
    )
    assert human.status_code == 200 and human.json()["sender"] == "human"
    conversation = client.get(
        f"/api/handoffs/{urgent.handoff_id}/messages"
    ).json()["messages"]
    assert [message["sender"] for message in conversation] == ["customer", "human"]
    assert [message.content for message in store.messages("urgent-user", "urgent-session")] == [
        "My order number is NS-100.",
        "I am checking order NS-100 now.",
    ]

    closed = client.post(f"/api/handoffs/{urgent.handoff_id}/close")
    assert closed.status_code == 200 and closed.json()["status"] == "closed"
