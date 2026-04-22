from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from typer.testing import CliRunner

from contextd.cli.main import app
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def test_forget_removes_source_and_chunks(tmp_contextd_home):
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
        path="/forget-me.pdf",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=1, content="bye")
    conn.commit()
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="t")
    vs.upsert([c1], np.array([[1, 0, 0, 0]], dtype=np.float32))

    r = CliRunner().invoke(app, ["forget", "/forget-me.pdf", "--corpus", "personal", "--yes"])
    assert r.exit_code == 0, r.output

    conn2 = open_db("personal")
    rows = conn2.execute("SELECT id FROM source WHERE path = ?", ("/forget-me.pdf",)).fetchall()
    assert rows == []
    rows2 = conn2.execute("SELECT id FROM chunk WHERE source_id = ?", (sid,)).fetchall()
    assert rows2 == []


def test_forget_resolves_relative_path(tmp_contextd_home, tmp_path, monkeypatch):
    """Regression: `contextd forget foo.pdf` (relative) must match the
    absolute path stored in the DB. Previously it was passed verbatim
    to SQL and silently failed."""
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="t",
        embed_dim=4,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    target = tmp_path / "relative-me.pdf"
    target.write_bytes(b"x")
    insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path=str(target.resolve()),
        content_hash="sha256:r",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    conn.commit()

    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["forget", "relative-me.pdf", "--corpus", "personal", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "would delete" in r.output


def test_forget_dry_run_does_not_delete(tmp_contextd_home):
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
        path="/keep.pdf",
        content_hash="sha256:k",
        ingested_at=datetime.now(UTC),
        chunk_count=0,
        status="active",
    )
    conn.commit()
    r = CliRunner().invoke(app, ["forget", "/keep.pdf", "--corpus", "personal", "--dry-run"])
    assert r.exit_code == 0
    assert "would delete" in r.output
    conn2 = open_db("personal")
    row = conn2.execute("SELECT id FROM source WHERE id = ?", (sid,)).fetchone()
    assert row is not None
