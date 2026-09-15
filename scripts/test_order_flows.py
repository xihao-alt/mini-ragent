"""Exercise two real order-policy Agent flows against a separate SQLite database."""

from __future__ import annotations

import json
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import AgentRuntime  # noqa: E402
from app import SYSTEM_MESSAGE  # noqa: E402
from llm import LLMClient  # noqa: E402
from rag import LocalEmbedder, Retriever, FaissVectorStore  # noqa: E402
from session import HandoffStore, SessionStore  # noqa: E402
from tools import HandoffToHumanTool, OrderLookupTool, ReadDocsTool, ToolRegistry  # noqa: E402
from tools.execution_context import ExecutionContext, reset_execution_context, set_execution_context  # noqa: E402


def main() -> None:
    database = ROOT / "data" / "order_flow_test.db"
    store = SessionStore(database)
    handoffs = HandoffStore(store)
    embedder = LocalEmbedder(ROOT / "models" / "all-MiniLM-L6-v2")
    retriever = Retriever(
        embedder,
        FaissVectorStore.load(
            ROOT / "data" / "index" / "novashop_support.faiss",
            ROOT / "data" / "index" / "novashop_support_chunks.json",
        ),
    )
    registry = ToolRegistry()
    registry.register(OrderLookupTool(store))
    registry.register(ReadDocsTool(retriever))
    registry.register(HandoffToHumanTool(handoffs))
    runtime = AgentRuntime(LLMClient(), registry, max_steps=6, system_message=SYSTEM_MESSAGE)

    cases = {
        "A": "我的 NS001 为什么还没到，而且好像被扣了两次钱？",
        "B": "我的 NS003 现在可以取消吗？",
    }
    for label, prompt in cases.items():
        session_id = f"order-flow-{label.lower()}"
        token = set_execution_context(ExecutionContext("user_001", session_id))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = runtime.run(prompt)
        finally:
            reset_execution_context(token)
        summary = {
            "case": label,
            "tools": [step.tool_name for step in result.steps if step.tool_name],
            "order": next(
                (step.tool_result for step in result.steps if step.tool_name == "order_lookup"),
                None,
            ),
            "handoff": next(
                (step.tool_result for step in result.steps if step.tool_name == "handoff_to_human"),
                None,
            ),
            "final_answer": result.final_answer,
            "error": result.error,
        }
        print("FLOW_RESULT " + json.dumps(summary, ensure_ascii=True))
        if result.error:
            raise RuntimeError(f"Case {label} failed: {result.error}")
        if label == "A":
            assert summary["tools"][0] == "order_lookup"
            assert "read_docs" in summary["tools"][1:-1]
            assert summary["tools"][-1] == "handoff_to_human"
            assert summary["order"]["completed_charge_count"] == 2
            assert summary["handoff"]["status"] == "pending"
        else:
            assert summary["tools"][:2] == ["order_lookup", "read_docs"]
            assert summary["order"]["order_status"] == "Packing"
            assert summary["handoff"] is None

    token = set_execution_context(ExecutionContext("user_001", "order-flow-c"))
    try:
        denied = OrderLookupTool(store).execute(order_id="NS008")
    finally:
        reset_execution_context(token)
    assert denied == {"error": "Order not found for the current user"}
    print("FLOW_RESULT " + json.dumps({"case": "C", "tools": ["order_lookup"], "result": denied}))


if __name__ == "__main__":
    main()
