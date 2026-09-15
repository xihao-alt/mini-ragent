"""Unit tests for LLM configuration and tool-schema conversion."""

from __future__ import annotations

from llm.client import LLMDecision, _openai_tools
from tools.base import Tool
from tools.registry import ToolRegistry


class ExampleTool(Tool):
    name = "example"
    description = "An example tool."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def execute(self, **kwargs: object) -> object:
        return kwargs


def test_registry_schema_converts_to_openai_tool() -> None:
    registry = ToolRegistry()
    registry.register(ExampleTool())

    assert _openai_tools(registry) == [
        {
            "type": "function",
            "function": {
                "name": "example",
                "description": "An example tool.",
                "parameters": ExampleTool.parameters,
            },
        }
    ]


def test_decision_types_are_distinct() -> None:
    assert LLMDecision.final("hello") == LLMDecision(type="final", text="hello")
    assert LLMDecision.tool_call("example", {"query": "docs"}) == LLMDecision(
        type="tool_call",
        tool_name="example",
        tool_arguments={"query": "docs"},
    )
