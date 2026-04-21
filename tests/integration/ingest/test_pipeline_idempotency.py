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
