from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from contextd.ingest.embedder import Embedder
from contextd.ingest.pipeline import IngestionPipeline
from contextd.ingest.protocol import ChunkDraft, SourceCandidate

pytestmark = pytest.mark.integration


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self._model_name = "fake-4d"
        self._device = "cpu"
        self._model = None
        self._dim = 4

    def embed(self, texts):
        return (
            np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
            if texts
            else np.zeros((0, 4), np.float32)
        )


class TwoChunkAdapter:
    source_type = "pdf"

    def can_handle(self, p):
        return True

    def sources(self, p):
        yield SourceCandidate(
            path=Path("/tmp/f.pdf"),
            source_type="pdf",
            canonical_id="/tmp/f.pdf",
            content_hash="sha256:deadbeef",
            title="F",
            source_mtime=datetime(2026, 4, 1, tzinfo=UTC),
        )

    def parse(self, source):
        yield ChunkDraft(
            ordinal=0, content="intro text", token_count=2, section_label="introduction"
        )
        yield ChunkDraft(ordinal=1, content="method text", token_count=2, section_label="methods")

    def metadata(self, source):
        return {"pdf_authors_list": "Fu, Smith"}

    def edges(self, chunks):
        yield from ()


def test_ingest_writes_source_chunks_embeddings(tmp_contextd_home):
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunkAdapter()])
    report = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    assert report.sources_ingested == 1
    assert report.chunks_written == 2
    assert report.sources_skipped == 0


def test_ingest_is_idempotent_on_unchanged_hash(tmp_contextd_home):
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunkAdapter()])
    r1 = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    r2 = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    assert r1.sources_ingested == 1 and r1.sources_skipped == 0
    assert r2.sources_ingested == 0 and r2.sources_skipped == 1


class FailingVectorStore:
    """Simulates a VectorStore that fails on upsert after SQLite writes."""

    def __init__(self) -> None:
        self.calls = 0

    def upsert(self, ids, vecs) -> None:
        raise RuntimeError("lancedb write failed")

    def delete(self, ids) -> None:
        pass


class UpsertOkButCommitFailsVS:
    """Upsert succeeds, but a later exception forces rollback — tests the
    compensating vs.delete() path."""

    def __init__(self) -> None:
        self.upserted: list[int] = []
        self.deleted: list[int] = []

    def upsert(self, ids, vecs) -> None:
        self.upserted.extend(ids)

    def delete(self, ids) -> None:
        self.deleted.extend(ids)


def test_ingest_rolls_back_on_vector_failure(tmp_contextd_home, monkeypatch):
    from contextd.ingest import pipeline as pipe_mod
    from contextd.storage.db import get_source_by_path, open_db

    monkeypatch.setattr(
        pipe_mod.VectorStore,
        "open",
        classmethod(lambda cls, **kw: FailingVectorStore()),
    )
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunkAdapter()])
    report = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    assert report.sources_ingested == 0
    assert report.sources_failed == 1
    # Verify the source row was rolled back: a second ingest with a fixed vs would succeed
    conn = open_db("personal")
    src = get_source_by_path(conn, corpus="personal", path="/tmp/f.pdf")
    assert src is None, f"rollback should have removed the source row, got {src}"


# ---------------------------------------------------------------------------
# Multi-source-per-file regression
# ---------------------------------------------------------------------------


class _FakeEncoding:
    def __init__(self, n: int) -> None:
        self.ids = [0] * n


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> _FakeEncoding:  # noqa: ARG002
        return _FakeEncoding(max(1, len(text.split())))


def test_multi_conversation_claude_export_does_not_collide_on_path(
    tmp_contextd_home, monkeypatch
) -> None:
    """Regression: fixture contains 3 conversations in one JSON file. Before the
    pipeline started keying source.path on canonical_id, the second and third
    conversations crashed on UNIQUE(source.corpus, source.path) because all
    three SourceCandidates carried the same file path."""
    import tokenizers

    from contextd.ingest.adapters.claude_export import ClaudeExportAdapter
    from contextd.storage.db import open_db

    monkeypatch.setattr(
        tokenizers.Tokenizer,
        "from_pretrained",
        lambda *_a, **_k: _FakeTokenizer(),
    )

    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "claude" / "export.json"
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[ClaudeExportAdapter()])
    report = pipe.ingest(path=fixture, corpus="personal")

    assert report.sources_failed == 0, report.errors
    assert (
        report.sources_ingested == 3
    ), f"expected 3 conversation sources, got {report.sources_ingested}"

    conn = open_db("personal")
    paths = [
        row[0] for row in conn.execute("SELECT path FROM source WHERE status='active' ORDER BY id")
    ]
    assert len(paths) == len(set(paths)) == 3
    assert all("#conversations/" in p for p in paths)


# ---------------------------------------------------------------------------
# Compensating vector delete on mid-transaction failure
# ---------------------------------------------------------------------------


class TwoChunksBadEdgeAdapter:
    """Runs vs.upsert() then raises KeyError from an edge pointing at a
    non-existent ordinal — simulates any failure that lands between the
    upsert call and the COMMIT."""

    source_type = "pdf"

    def can_handle(self, p):
        return True

    def sources(self, p):
        from contextd.ingest.protocol import SourceCandidate

        yield SourceCandidate(
            path=Path("/tmp/bad.pdf"),
            source_type="pdf",
            canonical_id="/tmp/bad.pdf",
            content_hash="sha256:bad",
            title="Bad",
        )

    def parse(self, source):
        from contextd.ingest.protocol import ChunkDraft

        yield ChunkDraft(ordinal=0, content="a", token_count=1)
        yield ChunkDraft(ordinal=1, content="b", token_count=1)

    def metadata(self, source):
        return {}

    def edges(self, chunks):
        from contextd.ingest.protocol import EdgeDraft

        # source_ordinal=99 is not in ordinal_to_id, so the pipeline's
        # `ordinal_to_id[e.source_ordinal]` raises KeyError AFTER vs.upsert()
        # already ran — this is the orphan-vector code path.
        yield EdgeDraft(source_ordinal=99, target_ordinal=0, edge_type="conversation_next")


def test_vs_upsert_rolled_back_when_commit_fails(tmp_contextd_home, monkeypatch):
    from contextd.ingest import pipeline as pipe_mod

    vs = UpsertOkButCommitFailsVS()
    monkeypatch.setattr(
        pipe_mod.VectorStore,
        "open",
        classmethod(lambda cls, **kw: vs),
    )
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunksBadEdgeAdapter()])
    report = pipe.ingest(path=Path("/tmp/bad.pdf"), corpus="personal")

    assert report.sources_failed == 1
    assert report.sources_ingested == 0
    assert vs.upserted, "test assumes upsert ran before the failure"
    assert vs.deleted == vs.upserted, (
        f"compensating delete must remove exactly the ids that were upserted; "
        f"upserted={vs.upserted}, deleted={vs.deleted}"
    )
