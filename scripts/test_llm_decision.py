"""Make two real LLM decisions without executing requested tools."""

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
from tools import ReadDocsTool, ToolRegistry  # noqa: E402


MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support_chunks.json"
SYSTEM_MESSAGE = (
    "You are MiniRAGent. Answer ordinary conversation directly. When a user asks "
    "for information according to their project documents, use the available document "
    "search tool. Make only one decision; do not pretend that a tool was executed."
)


def build_registry() -> ToolRegistry:
    embedder = LocalEmbedder(MODEL_PATH)
    store = FaissVectorStore.load(INDEX_PATH, MAPPING_PATH)
    registry = ToolRegistry()
    registry.register(ReadDocsTool(Retriever(embedder, store)))
    return registry


def main() -> None:
    client = LLMClient()
    registry = build_registry()
    cases = {
        "ordinary_chat": "Hello, how are you?",
        "document_question": (
            "According to my project documents, which component coordinates model "
            "calls and tool execution?"
        ),
    }

    for label, prompt in cases.items():
        decision = client.decide(prompt, registry, system_message=SYSTEM_MESSAGE)
        print(f"{label}:")
        print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
