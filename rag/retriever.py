"""Semantic retrieval over embedded chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .chunker import Chunk
from .embedding import LocalEmbedder
from .vector_store import FaissVectorStore


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedder: LocalEmbedder, store: FaissVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> list[SearchResult]:
        vector = self.embedder.encode([query])[0]
        return [
            SearchResult(chunk=chunk, score=score)
            for chunk, score in self.store.search(vector, top_k)
        ]
