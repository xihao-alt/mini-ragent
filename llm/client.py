"""OpenAI-compatible chat client with Tool Registry integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from dotenv import dotenv_values
import httpx
from openai import OpenAI

from tools.registry import ToolRegistry


DecisionType = Literal["final", "tool_call"]


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "LLMConfig":
        """Load the OpenAI-compatible endpoint configuration from a .env file."""
        selected = env_path or os.getenv("LLM_ENV_FILE")
        path = Path(selected) if selected else Path(__file__).resolve().parents[1] / ".env"
        file_values = dotenv_values(path)
        def configured(primary: str, alias: str | None = None) -> str:
            value = file_values.get(primary)
            if not value and alias:
                value = file_values.get(alias)
            if not value:
                value = os.getenv(primary) or (os.getenv(alias) if alias else "")
            return str(value or "").strip()
        values = {
            "LLM_API_KEY": configured("LLM_API_KEY", "OPENAI_API_KEY"),
            "LLM_BASE_URL": configured("LLM_BASE_URL", "OPENAI_BASE_URL"),
            "LLM_MODEL": configured("LLM_MODEL"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing LLM configuration: {', '.join(missing)}")

        return cls(
            api_key=values["LLM_API_KEY"],
            base_url=values["LLM_BASE_URL"].rstrip("/"),
            model=values["LLM_MODEL"],
        )


@dataclass(frozen=True)
class LLMDecision:
    """One model decision: either final text or one requested tool call."""

    type: DecisionType
    text: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_call_id: str | None = None

    @classmethod
    def final(cls, text: str) -> "LLMDecision":
        return cls(type="final", text=text)

    @classmethod
    def tool_call(
        cls,
        name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
    ) -> "LLMDecision":
        return cls(
            type="tool_call",
            tool_name=name,
            tool_arguments=arguments,
            tool_call_id=call_id,
        )


def _openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Convert registry schemas to OpenAI-compatible function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in registry.schemas()
    ]


class LLMClient:
    """Ask an OpenAI-compatible model for one response or tool decision."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=120.0,
            max_retries=2,
            http_client=httpx.Client(trust_env=False),
        )

    def decide(
        self,
        user_message: str,
        registry: ToolRegistry,
        *,
        system_message: str | None = None,
    ) -> LLMDecision:
        """Make one model call without executing any requested tool."""
        if not user_message.strip():
            raise ValueError("user_message must be non-empty")

        messages: list[dict[str, Any]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.decide_messages(messages, registry)

    def decide_messages(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> LLMDecision:
        """Make one decision using the complete in-progress conversation."""
        tools = _openai_tools(registry)
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools:
            request.update({"tools": tools, "tool_choice": "auto"})
        response = self._client.chat.completions.create(
            **request,
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")

        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM returned invalid JSON arguments for tool '{call.function.name}'"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must decode to a JSON object")
            return LLMDecision.tool_call(call.function.name, arguments, call.id)

        return LLMDecision.final(message.content or "")

    def stream_decision_messages(
        self,
        messages: list[dict[str, Any]],
        registry: ToolRegistry,
    ) -> Iterator[tuple[str, str | LLMDecision]]:
        """Stream text deltas and finish with one normalized model decision."""
        tools = _openai_tools(registry)
        request: dict[str, Any] = {"model": self.config.model, "messages": messages, "stream": True}
        if tools:
            request.update({"tools": tools, "tool_choice": "auto"})
        stream = self._client.chat.completions.create(**request)
        content_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                yield ("text", delta.content)
            for call in delta.tool_calls or []:
                current = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    current["id"] += call.id
                if call.function:
                    if call.function.name:
                        current["name"] += call.function.name
                    if call.function.arguments:
                        current["arguments"] += call.function.arguments

        if calls:
            call = calls[min(calls)]
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM returned invalid JSON arguments for tool '{call['name']}'"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must decode to a JSON object")
            yield ("decision", LLMDecision.tool_call(call["name"], arguments, call["id"] or None))
            return
        yield ("decision", LLMDecision.final("".join(content_parts)))

    def complete_messages(self, messages: list[dict[str, Any]]) -> str:
        """Generate plain text without exposing tools to the model."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        return response.choices[0].message.content or ""

    def summarize_history(self, existing_summary: str, history: str) -> str:
        """Compress older conversation while preserving facts needed for follow-ups."""
        return self.complete_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "Create a compact factual conversation summary. Preserve names, "
                        "user preferences, decisions, unresolved tasks, tool findings, and "
                        "referents needed to understand later pronouns. Do not invent facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Existing summary:\n{existing_summary or '(none)'}\n\n"
                        f"Older conversation to merge:\n{history}"
                    ),
                },
            ]
        )

    def extract_long_term_memory(self, user_message: str) -> dict[str, str] | None:
        """Return one reusable user memory, or None for session-only information."""
        raw = self.complete_messages(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify one user message for long-term memory. Save explicit "
                        "remember requests, stable preferences/settings, and persistent "
                        "projects, goals, or important status. Do not save ordinary chat, "
                        "temporary questions, search requests/results, tool output, or "
                        "references such as 'that one'. Return JSON only. If saving, return "
                        '{"save":true,"key":"stable_snake_case_key","memory":"a concise '
                        'standalone fact"}. Otherwise return {"save":false}. Never include '
                        "hidden reasoning."
                    ),
                },
                {"role": "user", "content": user_message},
            ]
        ).strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(decision, dict) or decision.get("save") is not True:
            return None
        key = decision.get("key")
        memory = decision.get("memory")
        if not isinstance(key, str) or not key.strip() or not isinstance(memory, str) or not memory.strip():
            return None
        return {"key": key.strip(), "memory": memory.strip()}

    def respond_with_tool_result(
        self,
        user_message: str,
        tool_name: str,
        tool_result: Any,
    ) -> str:
        """Ask the model to summarize one already-executed tool result."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using the supplied tool result. "
                        "Be concise, preserve useful source links, and state clearly if "
                        "the tool returned an error or insufficient evidence."
                    ),
                },
                {"role": "user", "content": user_message},
                {
                    "role": "user",
                    "content": (
                        f"Tool {tool_name} returned:\n"
                        + json.dumps(tool_result, ensure_ascii=False)
                    ),
                },
            ],
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        return response.choices[0].message.content or ""
