"""Integration tests for the `contextd query` CLI subcommand."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from typer.testing import CliRunner

from contextd.cli.main import app
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed a minimal corpus with one chunk and its vector."""
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
        title="A",
    )
    c1 = insert_chunk(
        conn,
        source_id=sid,
        ordinal=0,
        token_count=5,
        content="negation handling clinical",
    )
    conn.commit()

    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="t")
    vs.upsert([c1], np.array([[1, 0, 0, 0]], dtype=np.float32))

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts: list[str]) -> np.ndarray:
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    # Patch the import site in dense.py — same fix applied in T8:
    # dense.py does `from contextd.ingest.embedder import default_embedder` at
    # module level, so patching the source module wouldn't intercept the already-
    # bound name.  We patch the name as it exists in the consuming module instead.
    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())


def test_query_json_output_shape(
    tmp_contextd_home: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(monkeypatch)
    r = CliRunner().invoke(
        app,
        ["query", "negation", "--corpus", "personal", "--limit", "1", "--no-rerank", "--json"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert "results" in data and "trace" in data
    assert data["results"][0]["chunk"]["content"] == "negation handling clinical"


def test_query_rich_output_includes_source_path(
    tmp_contextd_home: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(monkeypatch)
    r = CliRunner().invoke(
        app,
        ["query", "negation", "--corpus", "personal", "--limit", "1", "--no-rerank"],
    )
    assert r.exit_code == 0, r.output
    assert "/a.pdf" in r.output
