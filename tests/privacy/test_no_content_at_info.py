"""Enforce: query text and chunk content never surface in INFO-level logs.

A future-proofing check: if anyone wires ``log.info("retrieve",
query=req.query, result=...)`` into the pipeline, this test fails.
contextd runs in clinical contexts where logs are shipped to third-party
aggregators — leaking retrieved content at INFO would be a PHIPA /
HIPAA breach vector.

Seeds a chunk containing a PHI-looking sentinel, runs a query that
matches it, then asserts the sentinel never appears in any captured
log record at INFO or below.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from contextd.logging_ import configure_logging
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.privacy

_SENTINELS = ("TOP SECRET", "PATIENT X42", "PHI-STRING-A1B2")


async def test_query_content_not_logged_at_info(
    tmp_contextd_home: Path,
    stub_embedder: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_logging()
    caplog.set_level(logging.INFO, logger="contextd")

    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="stub",
        embed_dim=4,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    sid = insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path="/phi.pdf",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=1,
        status="active",
    )
    cid = insert_chunk(
        conn,
        source_id=sid,
        ordinal=0,
        token_count=6,
        content="TOP SECRET PHI PATIENT X42 PHI-STRING-A1B2 admitted 2026-04-01",
    )
    conn.commit()

    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="stub")
    vs.upsert([cid], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    req = build_request(
        query="TOP SECRET PHI",
        corpus="personal",
        limit=1,
        rerank=False,
        rewrite=False,
    )
    results, _ = await retrieve(req)
    assert results, "retrieve() returned nothing — test fixture is broken, not the invariant"

    joined = " ".join(rec.getMessage() for rec in caplog.records)
    for sentinel in _SENTINELS:
        assert sentinel not in joined, (
            f"sensitive content {sentinel!r} leaked at INFO: {joined[:400]!r}"
        )
