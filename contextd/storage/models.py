from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

SourceType = Literal[
    "pdf",
    "claude_export",
    "git_repo",
    "markdown",
    "notion",
    "gmail",
    "arxiv_bookmark",
    "web_page",
]
SourceStatus = Literal["active", "deleted", "failed"]
EdgeType = Literal[
    "wikilink",
    "conversation_next",
    "conversation_prev",
    "code_imports",
    "pdf_cites",
    "email_reply_to",
    "email_thread",
]
Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Corpus:
    name: str
    embed_model: str
    embed_dim: int
    created_at: datetime
    schema_version: int
    root_path: str | None = None


@dataclass(frozen=True)
class Source:
    id: int
    corpus: str
    source_type: SourceType
    path: str
    content_hash: str
    ingested_at: datetime
    chunk_count: int
    status: SourceStatus
    title: str | None = None
    source_mtime: datetime | None = None


@dataclass(frozen=True)
class Chunk:
    id: int
    source_id: int
    ordinal: int
    content: str
    token_count: int
    offset_start: int | None = None
    offset_end: int | None = None
    section_label: str | None = None
    scope: str | None = None
    role: Role | None = None
    chunk_timestamp: datetime | None = None


@dataclass(frozen=True)
class Edge:
    id: int
    source_chunk_id: int
    edge_type: EdgeType
    target_chunk_id: int | None = None
    target_hint: str | None = None
    label: str | None = None
    weight: float | None = None


@dataclass(frozen=True)
class ChunkResult:
    """What the retrieval pipeline returns; superset of Chunk with source + score."""

    chunk: Chunk
    source: Source
    score: float
    rank: int
    metadata: dict[str, str]
    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class QueryTrace:
    trace_id: str  # ULID
    latency_ms: int
    dense_candidates: int
    sparse_candidates: int
    reranker_used: str | None  # null when skipped / failed
    rewriter_used: str | None
