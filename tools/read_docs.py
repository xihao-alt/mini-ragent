"""Document-search tool backed by the existing RAG Retriever."""

from __future__ import annotations

from typing import Any, ClassVar

from rag.retriever import Retriever

from .base import Tool


class ReadDocsTool(Tool):
    name = "read_docs"
    description = "Search the indexed project documents for passages relevant to a query."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural-language question or topic to search for.",
                "minLength": 1,
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of relevant document chunks to return.",
                "minimum": 1,
                "default": 3,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def execute(self, *, query: str, top_k: int = 3, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs:
            raise TypeError(f"Unexpected read_docs arguments: {', '.join(kwargs)}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        return [
            {
                "chunk_id": result.chunk.id,
                "text": result.chunk.text,
                "source": result.chunk.source,
                "score": result.score,
            }
            for result in self.retriever.retrieve(query.strip(), top_k=top_k)
        ]
