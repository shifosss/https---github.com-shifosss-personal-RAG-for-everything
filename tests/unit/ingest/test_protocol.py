from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from contextd.ingest.protocol import Adapter, ChunkDraft, SourceCandidate


def test_source_candidate_is_frozen():
    sc = SourceCandidate(
        path=Path("/a.pdf"), source_type="pdf", canonical_id="/a.pdf", content_hash="sha256:x"
    )
    with pytest.raises(FrozenInstanceError):
        sc.path = Path("/b.pdf")  # type: ignore[misc]


def test_chunk_draft_defaults():
    c = ChunkDraft(ordinal=0, content="hi", token_count=1)
    assert c.section_label is None


def test_adapter_is_runtime_checkable():
    class StubPDF:
        source_type: str = "pdf"

        def can_handle(self, path: Path) -> bool:
            return True

        def sources(self, path: Path):
            yield from ()

        def parse(self, source: SourceCandidate):
            yield from ()

        def metadata(self, source: SourceCandidate):
            return {}

        def edges(self, chunks):
            yield from ()

    assert isinstance(StubPDF(), Adapter)
