from datetime import UTC, datetime

import numpy as np
import pytest

from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed_corpus(corpus="personal"):
    conn = open_db(corpus)
    insert_corpus(
        conn,
        name=corpus,
        embed_model="t",
        embed_dim=4,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    sid = insert_source(
        conn,
        corpus=corpus,
        source_type="pdf",
        path="/a.pdf",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
        title="A",
    )
    c1 = insert_chunk(
        conn, source_id=sid, ordinal=0, token_count=5, content="negation handling clinical"
    )
    c2 = insert_chunk(
        conn, source_id=sid, ordinal=1, token_count=5, content="transformer architecture overview"
    )
    conn.commit()
    vs = VectorStore.open(corpus=corpus, embed_dim=4, model_name="t")
    vs.upsert([c1, c2], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    return c1, c2


async def test_retrieve_returns_rrf_ordered_without_rerank(monkeypatch, tmp_contextd_home):
    c1, c2 = _seed_corpus()

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())
    req = build_request(query="negation", corpus="personal", limit=2, rerank=False, rewrite=False)
    results, trace = await retrieve(req)
    assert results[0].chunk.id == c1  # top-ranked: matches both dense and sparse
    assert trace.reranker_used is None


async def test_retrieve_returns_at_most_limit(monkeypatch, tmp_contextd_home):
    c1, c2 = _seed_corpus()

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())
    req = build_request(query="architecture", corpus="personal", limit=1, rerank=False)
    results, _ = await retrieve(req)
    assert len(results) == 1
