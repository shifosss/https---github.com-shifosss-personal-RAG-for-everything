from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from contextd.storage.models import Chunk, Corpus, Edge, Source


def test_source_is_frozen() -> None:
    s = Source(
        id=1,
        corpus="personal",
        source_type="pdf",
        path="/tmp/a.pdf",
        content_hash="sha256:abc",
        ingested_at=datetime.now(),
        chunk_count=5,
        status="active",
    )
    with pytest.raises(FrozenInstanceError):
        s.path = "/tmp/b.pdf"  # type: ignore[misc]


def test_chunk_defaults() -> None:
    c = Chunk(id=1, source_id=1, ordinal=0, content="hello", token_count=1)
    assert c.section_label is None
    assert c.role is None


def test_edge_accepts_hint_without_target() -> None:
    e = Edge(id=1, source_chunk_id=5, edge_type="wikilink", target_hint="Fu 2024")
    assert e.target_chunk_id is None
    assert e.target_hint == "Fu 2024"


def test_corpus_requires_embed_model() -> None:
    with pytest.raises(TypeError):
        Corpus(name="x", embed_dim=1024, created_at=datetime.now(), schema_version=1)  # type: ignore[call-arg]
