"""Integration tests for Tool Registry and read_docs."""

from __future__ import annotations

import json
from pathlib import Path

from rag.embedding import LocalEmbedder
from rag.retriever import Retriever
from rag.vector_store import FaissVectorStore
from tools.read_docs import ReadDocsTool
from tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support_chunks.json"
QUERY = "How long does an approved NovaShop refund take?"


def test_read_docs_registry() -> None:
    embedder = LocalEmbedder(MODEL_PATH)
    store = FaissVectorStore.load(INDEX_PATH, MAPPING_PATH)
    read_docs = ReadDocsTool(Retriever(embedder, store))
    registry = ToolRegistry()

    registry.register(read_docs)
    assert registry.names() == ["read_docs"]
    assert registry.schemas() == [
        {
            "name": read_docs.name,
            "description": read_docs.description,
            "parameters": read_docs.parameters,
        }
    ]
    assert registry.get("read_docs") is read_docs

    try:
        registry.register(read_docs)
        raise AssertionError("Duplicate registration should fail")
    except DuplicateToolError as exc:
        assert "read_docs" in str(exc)

    try:
        registry.get("missing_tool")
        raise AssertionError("Unknown tool lookup should fail")
    except ToolNotFoundError as exc:
        assert "missing_tool" in str(exc) and "read_docs" in str(exc)

    tool_name = "read_docs"
    results = registry.get(tool_name).execute(query=QUERY, top_k=2)
    assert len(results) == 2
    assert any("5-7 business days" in result["text"] for result in results)

    print("read_docs schema:")
    print(json.dumps(registry.schemas()[0], indent=2))
    print(f"Registry tools: {registry.names()}")
    print("read_docs result:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    test_read_docs_registry()
