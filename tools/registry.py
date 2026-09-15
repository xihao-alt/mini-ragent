"""Name-based registry for tools available to a runtime or LLM."""

from __future__ import annotations

from copy import deepcopy

from .base import Tool


class DuplicateToolError(ValueError):
    """Raised when two tools use the same name."""


class ToolNotFoundError(LookupError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("Only Tool instances can be registered")
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(self._tools) or "none"
            raise ToolNotFoundError(
                f"Tool '{name}' is not registered. Available tools: {available}"
            ) from exc

    def schemas(self) -> list[dict[str, object]]:
        """Return only the metadata an LLM needs for tool selection."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.parameters),
            }
            for tool in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools)
