"""Exercise three real model decisions and one Tavily search/summarization."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm import LLMClient  # noqa: E402
from rag.embedding import LocalEmbedder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402
from tools import ReadDocsTool, ToolRegistry, WebSearchTool  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "test.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "test_chunks.json"
SYSTEM_MESSAGE = (
    "You are MiniRAGent. Answer ordinary conversation directly. Use read_docs for "
    "questions explicitly about the user's project documents. Use web_search for "
    "recent, current, or external public information. Choose autonomously from the "
    "available tools and do not claim a tool has run before receiving its result."
)


def build_registry() -> ToolRegistry:
    embedder = LocalEmbedder(MODEL_PATH)
    store = FaissVectorStore.load(INDEX_PATH, MAPPING_PATH)
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, store)))
    registry.register(WebSearchTool())
    return registry


def print_json(label: str, value: object) -> None:
    print(f"\n{label}:")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    client = LLMClient()
    registry = build_registry()
    cases = {
        "local_document": (
            "According to my project documents, which component coordinates model "
            "calls and tool execution?"
        ),
        "external_information": (
            "Search the web for recent information about AI agents and summarize the "
            "most relevant results."
        ),
        "ordinary_chat": "Hello, how are you?",
    }

    decisions = {}
    for label, prompt in cases.items():
        decision = client.decide(prompt, registry, system_message=SYSTEM_MESSAGE)
        decisions[label] = decision
        print_json(f"{label} decision", asdict(decision))

    web_decision = decisions["external_information"]
    if web_decision.type != "tool_call":
        raise RuntimeError("The model did not select a tool for the external-information test")
    tool = registry.get(web_decision.tool_name or "")
    result = tool.execute(**(web_decision.tool_arguments or {}))
    print_json("web_search result", result)
    summary = client.respond_with_tool_result(
        cases["external_information"],
        web_decision.tool_name or "",
        result,
    )
    print(f"\nfinal LLM summary:\n{summary}")


if __name__ == "__main__":
    main()
