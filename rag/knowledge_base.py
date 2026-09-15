"""Reusable PDF-to-FAISS indexing service for uploads."""

from __future__ import annotations

from pathlib import Path

from .chunker import Chunk, chunk_text
from .embedding import LocalEmbedder
from .loader import load_pdf
from .retriever import Retriever
from .vector_store import FaissVectorStore


class KnowledgeBase:
    def __init__(
        self,
        docs_dir: str | Path,
        index_path: str | Path,
        mapping_path: str | Path,
        embedder: LocalEmbedder,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.embedder = embedder
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists() and self.mapping_path.exists():
            store = FaissVectorStore.load(self.index_path, self.mapping_path)
        else:
            store = FaissVectorStore(embedder.dimension)
        self.retriever = Retriever(embedder, store)

    def documents(self) -> list[str]:
        return [path.name for path in sorted(self.docs_dir.glob("*.pdf"))]

    def rebuild(self) -> int:
        chunks: list[Chunk] = []
        for pdf_path in sorted(self.docs_dir.glob("*.pdf")):
            text = load_pdf(pdf_path)
            for chunk in chunk_text(text, source=f"data/docs/{pdf_path.name}"):
                chunks.append(Chunk(id=len(chunks), text=chunk.text, source=chunk.source))
        store = FaissVectorStore(self.embedder.dimension)
        if chunks:
            store.add(self.embedder.encode([chunk.text for chunk in chunks]), chunks)
        store.save(self.index_path, self.mapping_path)
        self.retriever.store = store
        return len(chunks)
