import pytest

from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request
from contextd.retrieve.rerank import RerankUnavailable

pytestmark = pytest.mark.integration


async def test_rerank_failure_falls_through_to_rrf(monkeypatch, tmp_contextd_home):
    import numpy as np

    from tests.integration.retrieve.test_pipeline_end_to_end import _seed_corpus  # type: ignore

    _seed_corpus()

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())

    async def bad_rerank(**kw):
        raise RerankUnavailable("down")

    monkeypatch.setattr("contextd.retrieve.pipeline.rerank", bad_rerank)

    req = build_request(query="negation", corpus="personal", limit=2, rerank=True)
    results, trace = await retrieve(req)
    assert len(results) == 2
    assert trace.reranker_used is None
