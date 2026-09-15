"""Generic multi-step agent loop over an LLM and Tool Registry."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from llm import LLMClient, LLMDecision
from tools import ToolRegistry


MAX_STEPS = 6


@dataclass
class AgentStep:
    number: int
    decision: LLMDecision
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: Any = None
    final_answer: str | None = None


@dataclass
class AgentRunResult:
    final_answer: str | None
    steps: list[AgentStep]
    error: str | None = None
    new_messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.final_answer is not None


class AgentRuntime:
    """Run model decisions and arbitrary registered tools until a final answer."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_steps: int = MAX_STEPS,
        system_message: str | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.system_message = system_message

    @staticmethod
    def _print(label: str, value: Any) -> None:
        line = f"{label}: {json.dumps(value, ensure_ascii=False, default=str)}"
        try:
            print(line)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "ascii"
            print(line.encode(encoding, errors="backslashreplace").decode(encoding))

    def run(
        self,
        user_message: str,
        *,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> AgentRunResult:
        if not isinstance(user_message, str) or not user_message.strip():
            return AgentRunResult(None, [], "user_message must be a non-empty string")

        messages: list[dict[str, Any]] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        if history_messages:
            messages.extend(history_messages)
        new_messages: list[dict[str, Any]] = []
        user_entry = {"role": "user", "content": user_message.strip()}
        messages.append(user_entry)
        new_messages.append(user_entry)
        steps: list[AgentStep] = []

        for number in range(1, self.max_steps + 1):
            print(f"\n--- step {number} ---")
            try:
                decision = self.llm.decide_messages(messages, self.registry)
            except Exception as exc:
                error = f"LLM decision failed at step {number}: {exc}"
                self._print("error", error)
                return AgentRunResult(None, steps, error, new_messages)

            self._print("LLM decision", asdict(decision))
            step = AgentStep(number=number, decision=decision)
            steps.append(step)

            if decision.type == "final":
                step.final_answer = decision.text or ""
                assistant_entry = {"role": "assistant", "content": step.final_answer}
                messages.append(assistant_entry)
                new_messages.append(assistant_entry)
                self._print("final answer", step.final_answer)
                return AgentRunResult(step.final_answer, steps, None, new_messages)

            name = decision.tool_name or ""
            arguments = decision.tool_arguments or {}
            call_id = decision.tool_call_id or f"call_{number}"
            step.tool_name = name
            step.tool_arguments = arguments
            self._print("tool name", name)
            self._print("tool arguments", arguments)

            try:
                tool = self.registry.get(name)
                result = tool.execute(**arguments)
            except Exception as exc:
                result = {
                    "error": {
                        "type": "tool_execution_failed",
                        "message": str(exc),
                    }
                }
            step.tool_result = result
            self._print("tool result", result)

            assistant_entry = {
                    "role": "assistant",
                    "content": decision.text,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            tool_entry = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            messages.extend((assistant_entry, tool_entry))
            new_messages.extend((assistant_entry, tool_entry))

        error = f"Agent stopped after reaching the maximum of {self.max_steps} steps"
        self._print("error", error)
        return AgentRunResult(None, steps, error, new_messages)

    def run_stream(
        self,
        user_message: str,
        *,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the generic loop and stream final-answer text as it is generated."""
        if not isinstance(user_message, str) or not user_message.strip():
            yield {"type": "error", "message": "user_message must be a non-empty string"}
            return
        messages: list[dict[str, Any]] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        if history_messages:
            messages.extend(history_messages)
        new_messages: list[dict[str, Any]] = []
        user_entry = {"role": "user", "content": user_message.strip()}
        messages.append(user_entry)
        new_messages.append(user_entry)
        steps: list[AgentStep] = []

        for number in range(1, self.max_steps + 1):
            decision = None
            try:
                for event_type, value in self.llm.stream_decision_messages(messages, self.registry):
                    if event_type == "text":
                        yield {"type": "delta", "content": value}
                    else:
                        decision = value
            except Exception as exc:
                result = AgentRunResult(None, steps, f"LLM decision failed at step {number}: {exc}", new_messages)
                yield {"type": "error", "message": result.error, "result": result}
                return
            if not isinstance(decision, LLMDecision):
                result = AgentRunResult(None, steps, "LLM stream ended without a decision", new_messages)
                yield {"type": "error", "message": result.error, "result": result}
                return

            step = AgentStep(number=number, decision=decision)
            steps.append(step)
            if decision.type == "final":
                step.final_answer = decision.text or ""
                assistant_entry = {"role": "assistant", "content": step.final_answer}
                new_messages.append(assistant_entry)
                result = AgentRunResult(step.final_answer, steps, None, new_messages)
                yield {"type": "done", "result": result}
                return

            name = decision.tool_name or ""
            arguments = decision.tool_arguments or {}
            call_id = decision.tool_call_id or f"call_{number}"
            step.tool_name = name
            step.tool_arguments = arguments
            yield {"type": "tool_start", "step": number, "tool_name": name, "tool_arguments": arguments}
            try:
                tool = self.registry.get(name)
                tool_result = tool.execute(**arguments)
            except Exception as exc:
                tool_result = {"error": {"type": "tool_execution_failed", "message": str(exc)}}
            step.tool_result = tool_result
            yield {"type": "tool_result", "step": number, "tool_name": name, "tool_result": tool_result}
            assistant_entry = {
                "role": "assistant", "content": decision.text,
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
            }
            tool_entry = {"role": "tool", "tool_call_id": call_id, "name": name,
                          "content": json.dumps(tool_result, ensure_ascii=False, default=str)}
            messages.extend((assistant_entry, tool_entry))
            new_messages.extend((assistant_entry, tool_entry))

        result = AgentRunResult(None, steps, f"Agent stopped after reaching the maximum of {self.max_steps} steps", new_messages)
        yield {"type": "error", "message": result.error, "result": result}
