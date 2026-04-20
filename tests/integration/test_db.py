from datetime import UTC, datetime

import pytest

from contextd.storage.db import (
    fetch_chunks_by_ids,
    get_source_by_path,
    insert_chunk,
    insert_corpus,
    insert_source,
    open_db,
)

pytestmark = pytest.mark.integration


def test_insert_corpus_and_source_roundtrip(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="BAAI/bge-m3",
        embed_dim=1024,
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
        title="A",
    )
    got = get_source_by_path(conn, corpus="personal", path="/a.pdf")
    assert got is not None
    assert got.id == sid
    assert got.title == "A"


def test_insert_chunks_and_fetch_by_ids(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="BAAI/bge-m3",
        embed_dim=1024,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    sid = insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path="/b.pdf",
        content_hash="sha256:y",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=2, content="hello world")
    c2 = insert_chunk(conn, source_id=sid, ordinal=1, token_count=3, content="goodbye world today")
    rows = fetch_chunks_by_ids(conn, [c1, c2])
    assert {r.id for r in rows} == {c1, c2}


def test_unique_corpus_path_conflict_raises(tmp_contextd_home):
    import sqlite3

    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="BAAI/bge-m3",
        embed_dim=1024,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path="/a.pdf",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    with pytest.raises(sqlite3.IntegrityError):
        insert_source(
            conn,
            corpus="personal",
            source_type="pdf",
            path="/a.pdf",
            content_hash="sha256:y",
            ingested_at=datetime.now(UTC),
            chunk_count=0,
            status="active",
        )
