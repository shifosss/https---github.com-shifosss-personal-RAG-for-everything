from datetime import UTC, datetime

import pytest

from contextd.retrieve.format import hydrate_results
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db

pytestmark = pytest.mark.integration


def test_hydrate_returns_chunkresult_with_source_and_meta(tmp_contextd_home):
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
        chunk_count=1,
        status="active",
        title="A",
    )
    cid = insert_chunk(
        conn,
        source_id=sid,
        ordinal=0,
        token_count=2,
        content="hi",
        section_label="methods",
    )
    conn.execute(
        "INSERT INTO chunk_meta(chunk_id, key, value) VALUES (?, 'pdf_page', '4')",
        (cid,),
    )
    conn.commit()

    results = hydrate_results(corpus="personal", scored=[(cid, 0.9)])

    assert len(results) == 1
    r = results[0]
    assert r.chunk.id == cid
    assert r.source.title == "A"
    assert r.metadata.get("pdf_page") == "4"
    assert r.rank == 1
    assert r.score == 0.9
