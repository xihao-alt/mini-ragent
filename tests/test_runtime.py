"""Tests for the generic Agent Runtime loop."""

from __future__ import annotations

from agent.runtime import AgentRuntime
from llm import LLMDecision
from tools import Tool, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo input."
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, **kwargs: object) -> object:
        return {"echo": kwargs["value"]}


class SequenceLLM:
    def __init__(self, decisions: list[LLMDecision]) -> None:
        self.decisions = decisions
        self.messages_seen: list[list[dict[str, object]]] = []

    def decide_messages(self, messages: list[dict[str, object]], registry: ToolRegistry) -> LLMDecision:
        self.messages_seen.append(list(messages))
        return self.decisions.pop(0)


def test_runtime_executes_tool_and_returns_final() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = SequenceLLM(
        [
            LLMDecision.tool_call("echo", {"value": "hello"}, "call_1"),
            LLMDecision.final("Finished"),
        ]
    )

    result = AgentRuntime(llm, registry).run("Start")

    assert result.succeeded
    assert result.final_answer == "Finished"
    assert result.steps[0].tool_result == {"echo": "hello"}
    assert llm.messages_seen[1][-2]["role"] == "assistant"
    assert llm.messages_seen[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "echo",
        "content": '{"echo": "hello"}',
    }


def test_runtime_stops_at_max_steps() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = SequenceLLM(
        [LLMDecision.tool_call("echo", {"value": "again"}, f"call_{n}") for n in range(2)]
    )

    result = AgentRuntime(llm, registry, max_steps=2).run("Loop")

    assert not result.succeeded
    assert result.error == "Agent stopped after reaching the maximum of 2 steps"
    assert len(result.steps) == 2
