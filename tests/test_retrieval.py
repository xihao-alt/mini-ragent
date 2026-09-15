"""End-to-end NovaShop policy retrieval tests."""

from __future__ import annotations

from pathlib import Path

from rag.chunker import Chunk, chunk_text
from rag.embedding import LocalEmbedder
from rag.loader import load_pdf
from rag.retriever import Retriever
from rag.vector_store import FaissVectorStore
from tests.test_pdf_loader import KNOWLEDGE_BASE_PDF, create_knowledge_base_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support.faiss"
MAPPING_PATH = PROJECT_ROOT / "data" / "index" / "novashop_support_chunks.json"
POLICY_CASES = (
    ("How many days do I have to return an unused product?", "30 calendar days"),
    ("How long does an approved refund take?", "5-7 business days"),
    ("Can I cancel an order that is already packing?", "cannot change the address or cancel"),
    ("What should I do if my item arrived damaged?", "within 48 hours"),
    ("When must a duplicate card charge go to a human agent?", "Two completed charges"),
)


def build_index() -> tuple[list[Chunk], Retriever]:
    create_knowledge_base_pdf(KNOWLEDGE_BASE_PDF)
    text = load_pdf(KNOWLEDGE_BASE_PDF)
    chunks = chunk_text(text, source=str(KNOWLEDGE_BASE_PDF.relative_to(PROJECT_ROOT)), max_chars=420)
    embedder = LocalEmbedder(MODEL_PATH)
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    store = FaissVectorStore(embedder.dimension)
    store.add(embeddings, chunks)
    store.save(INDEX_PATH, MAPPING_PATH)
    return chunks, Retriever(embedder, FaissVectorStore.load(INDEX_PATH, MAPPING_PATH))


def test_novashop_policy_retrieval() -> None:
    chunks, retriever = build_index()
    assert len(chunks) >= 14
    assert retriever.store.index.ntotal == len(chunks)
    for query, expected in POLICY_CASES:
        results = retriever.retrieve(query, top_k=3)
        matched = next((result for result in results if expected in result.chunk.text), None)
        assert matched is not None, f"Missing {expected!r} for {query!r}"
        print(f"QUERY: {query}\nSCORE: {matched.score:.4f}\nMATCH: {matched.chunk.text}\n")


if __name__ == "__main__":
    test_novashop_policy_retrieval()
