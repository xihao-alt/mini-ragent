"""Run the five requested real Agent behavior checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import build_services  # noqa: E402


def main() -> None:
    services = build_services()
    user_id = f"app-test-{uuid4().hex[:10]}"
    cases = {
        "ordinary_chat": "Hello, how are you?",
        "local_knowledge": "According to my project documents, how does the Agent Runtime work?",
        "web_search": "Search the web for recent developments in AI agents.",
        "human_handoff": "I don't want to talk to a bot anymore. Please connect me to a human agent.",
        "multi_step": (
            "First try to use my existing project documents to find the production database "
            "administrator password. If the documents cannot resolve this reliably, request "
            "human support."
        ),
    }
    output = {}
    for label, prompt in cases.items():
        session_id = f"{label}-{uuid4().hex[:8]}"
        result = services.sessions.chat(user_id, session_id, prompt)
        output[label] = {
            "session_id": session_id,
            "tools": [
                {
                    "step": step.number,
                    "name": step.tool_name,
                    "arguments": step.tool_arguments,
                    "result": step.tool_result,
                }
                for step in result.steps
                if step.tool_name
            ],
            "answer": result.final_answer,
            "error": result.error,
        }
        print(f"\n=== {label} ===")
        print(json.dumps(output[label], ensure_ascii=False, indent=2))

    handoffs = services.handoffs.list(user_id=user_id)
    print("\n=== persisted handoffs ===")
    print(json.dumps([item.__dict__ for item in handoffs], ensure_ascii=False, indent=2))

    expected = {
        "ordinary_chat": [],
        "local_knowledge": ["read_docs"],
        "web_search": ["web_search"],
        "human_handoff": ["handoff_to_human"],
    }
    for label, tools in expected.items():
        called = [item["name"] for item in output[label]["tools"]]
        if any(name not in called for name in tools) or output[label]["error"]:
            raise AssertionError(f"Unexpected {label} result: {called}")
    if not handoffs:
        raise AssertionError("No handoff rows were persisted")


if __name__ == "__main__":
    main()
