"""Run a normal chat and a real multi-tool Agent Runtime task."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent import AgentRuntime  # noqa: E402
from llm import LLMClient  # noqa: E402
from rag.embedding import LocalEmbedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402
from tools import ReadDocsTool, ToolRegistry, WebSearchTool  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "test.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "test_chunks.json"
SYSTEM_MESSAGE = (
    "You are MiniRAGent. Complete every part of the user's request before giving the "
    "final answer. Use the available tools whenever they provide needed evidence. "
    "Questions about project documents require document evidence; questions about "
    "recent or external practices require web evidence. After each tool result, decide "
    "whether another tool is needed. Cite useful web URLs in the final answer."
)


def build_registry() -> ToolRegistry:
    embedder = LocalEmbedder(MODEL_PATH)
    store = FaissVectorStore.load(INDEX_PATH, MAPPING_PATH)
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, store)))
    registry.register(WebSearchTool())
    return registry


def print_run_summary(label: str, result: object) -> None:
    print(f"\n=== {label} summary ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    runtime = AgentRuntime(
        LLMClient(),
        build_registry(),
        max_steps=6,
        system_message=SYSTEM_MESSAGE,
    )

    chat = runtime.run("Hello, how are you?")
    print_run_summary(
        "ordinary chat",
        {"final_answer": chat.final_answer, "error": chat.error, "steps": len(chat.steps)},
    )

    task = runtime.run(
        "According to my project documents, explain how the Agent Runtime works, then "
        "search the web for recent Agent Runtime practices and compare them."
    )
    print_run_summary(
        "multi-tool task",
        {
            "final_answer": task.final_answer,
            "error": task.error,
            "tools": [
                {
                    "step": step.number,
                    "name": step.tool_name,
                    "arguments": step.tool_arguments,
                    "result": step.tool_result,
                }
                for step in task.steps
                if step.tool_name
            ],
        },
    )


if __name__ == "__main__":
    main()
