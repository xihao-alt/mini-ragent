"""Embedding generation with a local sentence-transformers model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Encode text using a model that must already exist on disk."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local embedding model not found: {path}")
        self.model_path = path
        self.model = SentenceTransformer(str(path), device="cpu", local_files_only=True)

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return normalized float32 vectors suitable for cosine search."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
