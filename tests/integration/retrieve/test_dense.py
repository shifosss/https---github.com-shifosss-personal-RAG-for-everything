"""Integration tests for dense_search (LanceDB ANN)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from contextd.retrieve.dense import dense_search
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(home: object, embed_dim: int = 4) -> list[int]:
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="t",
        embed_dim=embed_dim,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    sid = insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path="/a.pdf",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    ids = [
        insert_chunk(conn, source_id=sid, ordinal=i, token_count=1, content=f"c{i}")
        for i in range(3)
    ]
    vs = VectorStore.open(corpus="personal", embed_dim=embed_dim, model_name="t")
    vs.upsert(
        ids,
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.7, 0.7, 0, 0]], dtype=np.float32),
    )
    conn.commit()
    return ids


@pytest.mark.asyncio
async def test_dense_returns_topk_by_cosine(tmp_contextd_home: object) -> None:
    ids = _seed(tmp_contextd_home)

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    hits = await dense_search(query="anything", corpus="personal", k=2, embedder=StubEmb())  # type: ignore[arg-type]
    assert [h[0] for h in hits] == [ids[0], ids[2]]
