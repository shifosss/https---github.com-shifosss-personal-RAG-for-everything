import numpy as np
import pytest

from contextd.ingest.embedder import Embedder

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_embed_returns_1024_dim_unit_vectors() -> None:
    e = Embedder.load(model="BAAI/bge-m3", device="cpu")
    vecs = e.embed(["hello world", "negation handling in clinical NLP"])
    assert vecs.shape == (2, 1024)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_embed_empty_batch_returns_empty() -> None:
    e = Embedder.load(model="BAAI/bge-m3", device="cpu")
    vecs = e.embed([])
    assert vecs.shape == (0, 1024)
