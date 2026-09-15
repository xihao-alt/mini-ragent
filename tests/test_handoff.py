"""Tests for the human handoff queue tool."""

from __future__ import annotations

from tools.execution_context import ExecutionContext, reset_execution_context, set_execution_context
from tools.handoff_to_human import HandoffToHumanTool
from session import HandoffStore, SessionStore


def test_handoff_uses_runtime_identity_and_persists(tmp_path) -> None:
    session_store = SessionStore(tmp_path / "sessions.db")
    handoffs = HandoffStore(session_store)
    tool = HandoffToHumanTool(handoffs)
    token = set_execution_context(ExecutionContext("user-1", "session-a"))
    try:
        result = tool.execute(
            reason="User requested a human",
            issue_summary="The user wants human support.",
            priority="normal",
        )
    finally:
        reset_execution_context(token)

    assert result["status"] == "pending"
    queued = handoffs.list(user_id="user-1")
    assert len(queued) == 1
    assert queued[0].handoff_id == result["handoff_id"]
    assert queued[0].session_id == "session-a"


def test_handoff_requires_session_context(tmp_path) -> None:
    tool = HandoffToHumanTool(HandoffStore(SessionStore(tmp_path / "sessions.db")))
    result = tool.execute(reason="Need help", issue_summary="No active session")
    assert "error" in result


def test_pending_queue_priority_claim_and_close(tmp_path) -> None:
    session_store = SessionStore(tmp_path / "sessions.db")
    handoffs = HandoffStore(session_store)
    low = handoffs.create("user-low", "session-low", "low reason", "low issue", "low")
    urgent = handoffs.create(
        "user-urgent", "session-urgent", "urgent reason", "urgent issue", "urgent"
    )
    high = handoffs.create("user-high", "session-high", "high reason", "high issue", "high")

    assert [item.handoff_id for item in handoffs.list(status="pending")] == [
        urgent.handoff_id,
        high.handoff_id,
        low.handoff_id,
    ]
    assert handoffs.pending_count() == 3
    assert session_store.state("user-urgent", "session-urgent").status == "WAITING_HUMAN"

    claimed = handoffs.claim(urgent.handoff_id)
    assert claimed is not None and claimed.status == "active"
    assert claimed.claimed_at is not None
    assert handoffs.pending_count() == 2
    assert session_store.state("user-urgent", "session-urgent").status == "HUMAN_ACTIVE"

    closed = handoffs.close(urgent.handoff_id)
    assert closed is not None and closed.status == "closed"
    assert closed.closed_at is not None
    assert session_store.state("user-urgent", "session-urgent").status == "BOT"
