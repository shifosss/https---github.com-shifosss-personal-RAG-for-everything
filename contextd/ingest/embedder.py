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
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import-untyped]

        m = BGEM3FlagModel(model, use_fp16=False, device=device)
        return cls(model, device, m)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = self._model.encode(  # type: ignore[attr-defined]
            texts, batch_size=16, max_length=8192
        )["dense_vecs"]
        arr = np.asarray(out, dtype=np.float32)
        # BGE-M3 outputs are already L2-normalized; reassert defensively.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    from contextd.config import get_settings

    s = get_settings()
    return Embedder.load(model=s.embedding_model, device=s.embedding_device)
