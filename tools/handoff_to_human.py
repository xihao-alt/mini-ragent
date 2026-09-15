"""Tool that places the current conversation in a human-support queue."""

from __future__ import annotations

from typing import Any, ClassVar

from .execution_context import current_execution_context

from .base import Tool


class HandoffToHumanTool(Tool):
    name = "handoff_to_human"
    description = (
        "Queue the current conversation for human support when the user explicitly asks "
        "for a human, clearly refuses further bot service in an angry or distressed "
        "message, or a retrieved policy requires immediate human escalation. For other "
        "issues requiring human review, explain the need and ask whether the customer "
        "wants a transfer before creating a ticket. Do not use this tool for an "
        "unconfirmed possible issue that the agent can still clarify."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why human support is required.",
                "minLength": 1,
            },
            "issue_summary": {
                "type": "string",
                "description": "A concise standalone summary for the human support agent.",
                "minLength": 1,
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high", "urgent"],
                "default": "normal",
            },
        },
        "required": ["reason", "issue_summary"],
        "additionalProperties": False,
    }

    def __init__(self, store: Any) -> None:
        self.store = store

    def execute(
        self,
        *,
        reason: str,
        issue_summary: str,
        priority: str = "normal",
        **kwargs: Any,
    ) -> dict[str, str]:
        if kwargs:
            return {"error": f"Unexpected arguments: {', '.join(kwargs)}"}
        if not isinstance(reason, str) or not reason.strip():
            return {"error": "reason must be a non-empty string"}
        if not isinstance(issue_summary, str) or not issue_summary.strip():
            return {"error": "issue_summary must be a non-empty string"}
        if priority not in {"low", "normal", "high", "urgent"}:
            return {"error": "priority must be low, normal, high, or urgent"}
        context = current_execution_context()
        if context is None:
            return {"error": "handoff_to_human requires an active session context"}
        handoff = self.store.create(
            context.user_id,
            context.session_id,
            reason.strip(),
            issue_summary.strip(),
            priority,
        )
        return {
            "handoff_id": handoff.handoff_id,
            "status": handoff.status,
            "message": "The conversation has been queued for human support.",
        }
