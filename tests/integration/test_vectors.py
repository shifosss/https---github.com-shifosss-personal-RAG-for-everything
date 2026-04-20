import numpy as np
import pytest

from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def test_upsert_then_ann_search_returns_closest(tmp_contextd_home):
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="test-4d")
    vecs = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # chunk 1
            [0.0, 1.0, 0.0, 0.0],  # chunk 2
            [0.7, 0.7, 0.0, 0.0],  # chunk 3 (between 1 and 2)
        ],
        dtype=np.float32,
    )
    vs.upsert([1, 2, 3], vecs)
    hits = vs.ann_search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert hits[0][0] == 1  # chunk 1 ranks first


def test_delete_removes_vectors(tmp_contextd_home):
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="test-4d")
    vecs = np.eye(4, dtype=np.float32)
    vs.upsert([1, 2, 3, 4], vecs)
    vs.delete([2, 4])
    hits = vs.ann_search(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), k=4)
    ids = {h[0] for h in hits}
    assert 2 not in ids
    assert 4 not in ids


def test_empty_store_returns_empty_list(tmp_contextd_home):
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="test-4d")
    hits = vs.ann_search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=3)
    assert hits == []
