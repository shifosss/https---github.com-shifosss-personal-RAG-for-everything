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


async def test_dense_failure_falls_through_to_sparse(monkeypatch, tmp_contextd_home):
    """Regression for asyncio.gather bubbling a single lane failure.

    Before return_exceptions=True, a dense_search crash (e.g. LanceDB
    missing-table on a brand-new corpus) aborted the whole retrieve with
    no results and no trace. PRD §15.6 requires graceful degradation.
    """
    import numpy as np

    from tests.integration.retrieve.test_pipeline_end_to_end import _seed_corpus  # type: ignore

    _seed_corpus()

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())

    async def dead_dense(**_kw):
        raise RuntimeError("lancedb table not found")

    monkeypatch.setattr("contextd.retrieve.pipeline.dense_search", dead_dense)

    req = build_request(query="negation", corpus="personal", limit=2, rerank=False)
    results, trace = await retrieve(req)
    # Sparse (FTS5) should still land the seeded chunk — dense failure must not
    # abort the whole pipeline.
    assert trace.dense_candidates == 0
    assert isinstance(results, list), "pipeline must return a result list even on lane failure"
