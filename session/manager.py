"""Coordinate persistent sessions, bounded context, and Agent Runtime runs."""

from __future__ import annotations

from typing import Any, Iterator

from agent import AgentRunResult, AgentRuntime
from tools.execution_context import ExecutionContext, reset_execution_context, set_execution_context

from .context import ContextManager
from .memory import LongTermMemoryManager
from .handoff import HandoffStore
from .store import SessionStore


class SessionManager:
    def __init__(
        self,
        runtime: AgentRuntime,
        store: SessionStore,
        context: ContextManager,
        memory: LongTermMemoryManager | None = None,
        handoffs: HandoffStore | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.context = context
        self.memory = memory
        self.handoffs = handoffs

    def chat(self, user_id: str, session_id: str, message: str) -> AgentRunResult:
        session_status = self.store.state(user_id, session_id).status
        if session_status in {"WAITING_HUMAN", "HUMAN_ACTIVE"}:
            self.store.append_messages(
                user_id,
                session_id,
                [{"role": "user", "content": message}],
            )
            current = self.handoffs.current_for_session(user_id, session_id) if self.handoffs else None
            if current is not None and current.status == "active":
                self.handoffs.add_message(current.handoff_id, "customer", message)
            acknowledgement = (
                "Your message was added to the human support conversation."
                if session_status == "HUMAN_ACTIVE"
                else "Your message was added while you wait for human support."
            )
            return AgentRunResult(acknowledgement, [], None, [])
        history = self.context.build_messages(user_id, session_id)
        if self.memory:
            history = self.memory.context_messages(user_id, message) + history
        token = set_execution_context(ExecutionContext(user_id=user_id, session_id=session_id))
        try:
            result = self.runtime.run(message, history_messages=history)
        finally:
            reset_execution_context(token)
        self.store.append_messages(user_id, session_id, result.new_messages)
        if self.memory:
            try:
                self.memory.capture(user_id, session_id, message)
            except Exception as exc:
                print(f"Long-term memory capture skipped: {exc}")
        return result

    def chat_stream(self, user_id: str, session_id: str, message: str) -> Iterator[dict[str, Any]]:
        session_status = self.store.state(user_id, session_id).status
        if session_status in {"WAITING_HUMAN", "HUMAN_ACTIVE"}:
            self.store.append_messages(
                user_id,
                session_id,
                [{"role": "user", "content": message}],
            )
            current = self.handoffs.current_for_session(user_id, session_id) if self.handoffs else None
            if current is not None and current.status == "active":
                self.handoffs.add_message(current.handoff_id, "customer", message)
            acknowledgement = (
                "Your message was added to the human support conversation."
                if session_status == "HUMAN_ACTIVE"
                else "Your message was added while you wait for human support."
            )
            result = AgentRunResult(acknowledgement, [], None, [])
            yield {"type": "done", "result": result}
            return
        history = self.context.build_messages(user_id, session_id)
        if self.memory:
            history = self.memory.context_messages(user_id, message) + history
        final_result = None
        runtime_events = self.runtime.run_stream(message, history_messages=history)
        while True:
            # Starlette may resume a synchronous streaming iterator in a different
            # worker context. Scope the ContextVar to each generator advance.
            token = set_execution_context(ExecutionContext(user_id=user_id, session_id=session_id))
            try:
                event = next(runtime_events)
            except StopIteration:
                break
            finally:
                reset_execution_context(token)
            if event.get("type") in {"done", "error"}:
                final_result = event.get("result")
            yield event
        if final_result is not None:
            self.store.append_messages(user_id, session_id, final_result.new_messages)
            if self.memory:
                try:
                    self.memory.capture(user_id, session_id, message)
                except Exception as exc:
                    print(f"Long-term memory capture skipped: {exc}")
