"""Tests for QueryFilter application (blocker P3#4).

Before the filter was wired, QueryFilter fields were declared on the
public API (SearchRequest → QueryFilter) but the retrieve pipeline
ignored them. Requests asking for ``source_types=("pdf",)`` silently
returned results from every source type.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from contextd.retrieve.filters import apply_filter
from contextd.retrieve.preprocess import QueryFilter
from contextd.storage.models import Chunk, ChunkResult, Source

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.unit


def _chunk(cid: int, content: str = "x", section_label: str | None = None) -> Chunk:
    return Chunk(
        id=cid,
        source_id=100 + cid,
        ordinal=0,
        content=content,
        token_count=1,
        section_label=section_label,
    )


def _source(sid: int, source_type: str, path: str, ingested_at: datetime) -> Source:
    return Source(
        id=sid,
        corpus="eval",
        source_type=source_type,  # type: ignore[arg-type]
        path=path,
        content_hash="sha256:x",
        ingested_at=ingested_at,
        chunk_count=1,
        status="active",
    )


def _result(
    cid: int,
    source_type: str = "pdf",
    path: str = "/a.pdf",
    ingested: datetime | None = None,
    section_label: str | None = None,
    meta: dict[str, str] | None = None,
) -> ChunkResult:
    return ChunkResult(
        chunk=_chunk(cid, section_label=section_label),
        source=_source(100 + cid, source_type, path, ingested or datetime(2026, 1, 1, tzinfo=UTC)),
        score=1.0 / cid,
        rank=cid,
        metadata=meta or {},
        edges=(),
    )


def test_empty_filter_keeps_all_except_reference_sections() -> None:
    """Default QueryFilter has exclude_reference_sections=True."""
    results = [
        _result(1, section_label=None),
        _result(2, section_label="abstract"),
        _result(3, section_label="references"),
        _result(4, section_label="Bibliography"),  # case-insensitive
    ]
    f = QueryFilter()
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [1, 2]


def test_source_types_allow_list() -> None:
    results = [
        _result(1, source_type="pdf"),
        _result(2, source_type="git_repo"),
        _result(3, source_type="claude_export"),
    ]
    f = QueryFilter(source_types=("pdf", "claude_export"))
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [1, 3]


def test_source_path_prefix() -> None:
    results = [
        _result(1, path="/papers/a.pdf"),
        _result(2, path="/code/b.py"),
        _result(3, path="/papers/c.pdf"),
    ]
    f = QueryFilter(source_path_prefix="/papers/")
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [1, 3]


def test_date_range() -> None:
    old = datetime(2025, 1, 1, tzinfo=UTC)
    mid = datetime(2026, 1, 1, tzinfo=UTC)
    new = datetime(2027, 1, 1, tzinfo=UTC)
    results = [
        _result(1, ingested=old),
        _result(2, ingested=mid),
        _result(3, ingested=new),
    ]
    f = QueryFilter(
        date_from=datetime(2025, 6, 1, tzinfo=UTC),
        date_to=datetime(2026, 6, 1, tzinfo=UTC),
    )
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [2]


def test_metadata_exact_match() -> None:
    results = [
        _result(1, meta={"author": "fu"}),
        _result(2, meta={"author": "kaster"}),
        _result(3, meta={}),
    ]
    f = QueryFilter(metadata={"author": "fu"})
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [1]


def test_exclude_reference_sections_can_be_disabled() -> None:
    results = [
        _result(1, section_label="methods"),
        _result(2, section_label="references"),
    ]
    f = QueryFilter(exclude_reference_sections=False)
    kept = apply_filter(results, f)
    assert [r.chunk.id for r in kept] == [1, 2]


def test_combined_filters() -> None:
    results = [
        _result(1, source_type="pdf", path="/papers/a.pdf", section_label="methods"),
        _result(2, source_type="pdf", path="/papers/a.pdf", section_label="references"),
        _result(3, source_type="git_repo", path="/papers/a.pdf"),
        _result(4, source_type="pdf", path="/elsewhere/b.pdf", section_label="methods"),
    ]
    f = QueryFilter(source_types=("pdf",), source_path_prefix="/papers/")
    kept = apply_filter(results, f)
    # Only #1 — #2 excluded by default exclude_reference_sections,
    # #3 excluded by source_type, #4 excluded by path prefix.
    assert [r.chunk.id for r in kept] == [1]


def test_preserves_input_order() -> None:
    """Filter must not reorder — upstream rank/score semantics depend on it."""
    results = [_result(i) for i in (5, 3, 1, 2, 4)]
    kept = apply_filter(results, QueryFilter())
    assert [r.chunk.id for r in kept] == [5, 3, 1, 2, 4]
