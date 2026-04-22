"""Shared fixtures for privacy tests.

Two stubs side-step the HF hub:

* ``stub_embedder`` replaces ``default_embedder`` at both consuming
  import sites (``contextd.ingest.pipeline`` and ``contextd.retrieve.dense``).
  The real BGE-M3 is heavy and talks to the network on first load.
* ``stub_tokenizer`` patches ``tokenizers.Tokenizer.from_pretrained`` so
  the three adapters that lazy-load the BGE-M3 tokenizer
  (pdf / git_repo / claude_export) never hit the HF hub.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class _StubEmbedder:
    model_name = "stub"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


class _StubEncoding:
    """Mirrors the subset of ``tokenizers.Encoding`` our code uses."""

    def __init__(self, n_tokens: int) -> None:
        self.ids: list[int] = [0] * n_tokens


class _StubTokenizer:
    """Tokens ≈ whitespace-split words. Exact token count doesn't matter
    for privacy tests; only that we avoid the HF download.
    """

    def encode(self, text: str, *, add_special_tokens: bool = False) -> _StubEncoding:  # noqa: ARG002
        return _StubEncoding(n_tokens=max(1, len(text.split())))


@pytest.fixture
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> _StubEmbedder:
    """Patch default_embedder at both consuming import sites.

    dense.py and ingest.pipeline both do ``from contextd.ingest.embedder
    import default_embedder`` at module load time, so the name is already
    bound in each module. Patching the source module would be a no-op —
    we patch the consuming modules directly, same rule as elsewhere in
    the test suite.
    """
    emb = _StubEmbedder()
    monkeypatch.setattr("contextd.ingest.pipeline.default_embedder", lambda: emb)
    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: emb)
    return emb


@pytest.fixture
def stub_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block the HF download that ``Tokenizer.from_pretrained("BAAI/bge-m3")`` triggers."""

    def _fake_from_pretrained(*_: Any, **__: Any) -> _StubTokenizer:
        return _StubTokenizer()

    # All three adapters do `from tokenizers import Tokenizer`, so the classmethod
    # lookup goes through the canonical module. Patching the attribute on that
    # class covers every call site.
    import tokenizers

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", _fake_from_pretrained)
