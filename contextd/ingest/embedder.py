from __future__ import annotations

from functools import lru_cache
from typing import Self

import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str, model: object) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model
        self._dim = 1024  # BGE-M3

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @classmethod
    def load(cls, *, model: str = "BAAI/bge-m3", device: str = "cpu") -> Self:
        # sentence-transformers replaces FlagEmbedding for dense vectors.
        # FlagEmbedding 1.3.x eagerly imports a reranker submodule that pulls
        # GEMMA2_START_DOCSTRING out of transformers — removed in 4.49+, so the
        # whole package is unimportable on modern transformers. BGE-M3 is the
        # canonical SentenceTransformer model, so this is a lateral move.
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model, device=device)
        m.max_seq_length = 8192  # BGE-M3 native context
        return cls(model, device, m)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        arr = self._model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    from contextd.config import get_settings

    s = get_settings()
    return Embedder.load(model=s.embedding_model, device=s.embedding_device)
