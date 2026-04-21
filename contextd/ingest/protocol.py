from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from contextd.storage.models import EdgeType, Role, SourceType


@dataclass(frozen=True)
class SourceCandidate:
    path: Path  # canonical path; for multi-source files, path#fragment
    source_type: SourceType
    canonical_id: str  # equal to str(path) for single-source files; "path#frag" otherwise
    content_hash: str  # "sha256:..." computed over canonical bytes
    title: str | None = None
    source_mtime: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkDraft:
    ordinal: int
    content: str
    token_count: int
    offset_start: int | None = None
    offset_end: int | None = None
    section_label: str | None = None
    scope: str | None = None
    role: Role | None = None
    chunk_timestamp: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)  # lands in chunk_meta


@dataclass(frozen=True)
class EdgeDraft:
    source_ordinal: int
    edge_type: EdgeType
    target_ordinal: int | None = None  # resolved within the same source
    target_hint: str | None = None  # unresolved (e.g., wikilink text)
    label: str | None = None
    weight: float | None = None


@runtime_checkable
class Adapter(Protocol):
    source_type: SourceType

    def can_handle(self, path: Path) -> bool: ...
    def sources(self, path: Path) -> Iterable[SourceCandidate]: ...
    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]: ...
    def metadata(self, source: SourceCandidate) -> dict[str, str]: ...
    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]: ...
