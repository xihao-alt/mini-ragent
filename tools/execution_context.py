"""Per-run metadata available to tools without exposing it in tool schemas."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionContext:
    user_id: str
    session_id: str


_current: ContextVar[ExecutionContext | None] = ContextVar("tool_execution_context", default=None)


def set_execution_context(context: ExecutionContext) -> Token[ExecutionContext | None]:
    return _current.set(context)


def reset_execution_context(token: Token[ExecutionContext | None]) -> None:
    _current.reset(token)


def current_execution_context() -> ExecutionContext | None:
    return _current.get()
