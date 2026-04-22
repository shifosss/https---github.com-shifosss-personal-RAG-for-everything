from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from contextd.mcp.api import create_app
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(monkeypatch):
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
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=2, content="negation clinical")
    conn.commit()
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="t")
    vs.upsert([c1], np.array([[1, 0, 0, 0]], dtype=np.float32))

    class StubEmb:
        model_name = "t"
        dim = 4

        def embed(self, texts):
            return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)

    # IMPORTANT: Patch the import site, not the source module (T8 lesson). Also patch embedder
    # source for safety since other call sites may differ.
    monkeypatch.setattr("contextd.retrieve.dense.default_embedder", lambda: StubEmb())
    return c1, sid


def test_post_search_returns_results(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.post(
        "/v1/search",
        json={"query": "negation", "corpus": "personal", "limit": 1, "rerank": False},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["chunk"]["content"] == "negation clinical"
    assert body["trace"]["trace_id"]


def test_get_chunk_by_id(tmp_contextd_home, monkeypatch):
    cid, _ = _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.get(f"/v1/chunks/{cid}?corpus=personal")
    assert r.status_code == 200, r.text
    assert r.json()["chunk"]["chunk"]["id"] == cid


def test_get_chunk_honors_include_edges_false(tmp_contextd_home, monkeypatch):
    """Regression P4#1: the TS Zod schema advertises include_edges and
    include_metadata as controllable; the Python route used to ignore both
    and always return full payloads."""
    cid, _ = _seed(monkeypatch)
    client = TestClient(create_app())

    r = client.get(f"/v1/chunks/{cid}?corpus=personal&include_edges=false")
    assert r.status_code == 200, r.text
    assert r.json()["chunk"]["edges"] == []

    r = client.get(f"/v1/chunks/{cid}?corpus=personal&include_metadata=false")
    assert r.status_code == 200, r.text
    assert r.json()["chunk"]["metadata"] == {}

    r = client.get(f"/v1/chunks/{cid}?corpus=personal&include_edges=false&include_metadata=false")
    assert r.status_code == 200, r.text
    body = r.json()["chunk"]
    assert body["edges"] == [] and body["metadata"] == {}
    # Core fields still present
    assert body["chunk"]["id"] == cid and body["source"]["path"] == "/a.pdf"


def test_list_corpora(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.get("/v1/corpora")
    assert r.status_code == 200
    names = {c["name"] for c in r.json()["corpora"]}
    assert "personal" in names


def test_search_unknown_corpus_returns_404(tmp_contextd_home):
    client = TestClient(create_app())
    r = client.post("/v1/search", json={"query": "x", "corpus": "nope"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "CORPUS_NOT_FOUND"


def test_healthz_returns_200(tmp_contextd_home):
    client = TestClient(create_app())
    assert client.get("/v1/healthz").status_code == 200
