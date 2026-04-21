from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contextd.storage.models import EdgeType, SourceType  # noqa: TC001


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_types: list[SourceType] = Field(default_factory=list)
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_path_prefix: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    corpus: str = "personal"
    limit: int = 10
    rewrite: bool = False
    rerank: bool = True
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must be non-empty")
        return v

    @field_validator("limit")
    @classmethod
    def _clamp_limit(cls, v: int) -> int:
        return max(1, min(v, 100))


class ChunkView(BaseModel):
    id: int
    source_id: int
    ordinal: int
    content: str
    token_count: int
    section_label: str | None = None
    scope: str | None = None
    role: str | None = None
    chunk_timestamp: str | None = None
    offset_start: int | None = None
    offset_end: int | None = None


class SourceView(BaseModel):
    id: int
    corpus: str
    source_type: SourceType
    path: str
    title: str | None = None
    content_hash: str
    ingested_at: str
    chunk_count: int
    status: str


class EdgeView(BaseModel):
    id: int
    source_chunk_id: int
    target_chunk_id: int | None = None
    target_hint: str | None = None
    edge_type: EdgeType
    label: str | None = None
    weight: float | None = None


class ChunkResultView(BaseModel):
    chunk: ChunkView
    source: SourceView
    score: float
    rank: int
    metadata: dict[str, str]
    edges: list[EdgeView]


class QueryTraceView(BaseModel):
    trace_id: str
    latency_ms: int
    dense_candidates: int
    sparse_candidates: int
    reranker_used: str | None
    rewriter_used: str | None


class SearchResponse(BaseModel):
    results: list[ChunkResultView]
    query: dict[str, object]
    trace: QueryTraceView


class FetchChunkResponse(BaseModel):
    chunk: ChunkResultView


class ExpandContextResponse(BaseModel):
    chunks: list[ChunkResultView]


class GetEdgesResponse(BaseModel):
    edges: list[EdgeView]
    targets: list[ChunkResultView] | None = None


class ListSourcesResponse(BaseModel):
    sources: list[SourceView]
    total: int
    has_more: bool


class GetSourceResponse(BaseModel):
    source: SourceView
    metadata: dict[str, str]


class CorpusStats(BaseModel):
    name: str
    embed_model: str
    embed_dim: int
    source_count: int
    chunk_count: int
    created_at: str


class ListCorporaResponse(BaseModel):
    corpora: list[CorpusStats]


class ErrorEnvelope(BaseModel):
    code: Literal["BAD_REQUEST", "NOT_FOUND", "CORPUS_NOT_FOUND", "RERANK_UNAVAILABLE", "INTERNAL"]
    message: str
    trace_id: str | None = None
