from __future__ import annotations

import json as _json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from contextd.cli.main import app
from contextd.storage.db import insert_corpus, insert_source, open_db

pytestmark = pytest.mark.integration


def test_list_shows_source_rows(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="t",
        embed_dim=4,
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
        chunk_count=1,
        status="active",
        title="A",
    )
    conn.commit()
    r = CliRunner().invoke(app, ["list", "--corpus", "personal"])
    assert r.exit_code == 0, r.output
    assert "/a.pdf" in r.output


def test_list_json_output(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(
        conn,
        name="personal",
        embed_model="t",
        embed_dim=4,
        created_at=datetime.now(UTC),
        schema_version=1,
    )
    insert_source(
        conn,
        corpus="personal",
        source_type="pdf",
        path="/b.pdf",
        content_hash="sha256:y",
        ingested_at=datetime.now(UTC),
        chunk_count=2,
        status="active",
        title="B",
    )
    conn.commit()
    r = CliRunner().invoke(app, ["list", "--corpus", "personal", "--json"])
    assert r.exit_code == 0
    data = _json.loads(r.stdout)
    assert data[0]["path"] == "/b.pdf"
