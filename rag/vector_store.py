"""FAISS vector storage with a persistent chunk-to-vector mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from .chunker import Chunk


class FaissVectorStore:
    """Cosine-similarity FAISS index whose row positions map to chunks."""

    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: Sequence[Chunk]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(f"Expected vectors with dimension {self.index.d}")
        if vectors.shape[0] != len(chunks):
            raise ValueError("Each vector must have exactly one chunk mapping")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, positions = self.index.search(query, min(top_k, self.index.ntotal))
        return [
            (self.chunks[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]

    def save(self, index_path: str | Path, mapping_path: str | Path) -> None:
        index_file = Path(index_path)
        mapping_file = Path(mapping_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        mapping_file.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: str | Path, mapping_path: str | Path) -> "FaissVectorStore":
        index = faiss.read_index(str(index_path))
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        store = cls(index.d)
        store.index = index
        store.chunks = [Chunk(**item) for item in mapping]
        if store.index.ntotal != len(store.chunks):
            raise ValueError("FAISS index and chunk mapping have different sizes")
        return store
