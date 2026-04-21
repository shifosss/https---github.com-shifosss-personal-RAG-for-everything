"""Integration tests for sparse_search (FTS5 BM25)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from contextd.retrieve.sparse import sparse_search
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_bm25_returns_matching_chunk(tmp_contextd_home: object) -> None:
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="t",
        embed_dim=4,
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
    c1 = insert_chunk(
        conn,
        source_id=sid,
        ordinal=0,
        token_count=5,
        content="negation handling in clinical NLP",
    )
    c2 = insert_chunk(
        conn,
        source_id=sid,
        ordinal=1,
        token_count=5,
        content="transformer architecture overview",
    )
    conn.commit()

    hits = await sparse_search(query="negation", corpus="personal", k=5)
    ids = [h[0] for h in hits]
    assert c1 in ids and c2 not in ids


@pytest.mark.asyncio
async def test_empty_query_returns_empty(tmp_contextd_home: object) -> None:
    open_db("personal")
    hits = await sparse_search(query="", corpus="personal", k=5)
    assert hits == []
