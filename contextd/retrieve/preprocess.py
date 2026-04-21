from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ulid import ULID

if TYPE_CHECKING:
    from datetime import datetime

    from contextd.storage.models import SourceType

_MAX_QUERY_LEN = 2000


@dataclass(frozen=True)
class QueryFilter:
    source_types: tuple[SourceType, ...] = ()
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_path_prefix: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    exclude_reference_sections: bool = True


@dataclass(frozen=True)
class QueryRequest:
    query: str
    corpus: str
    limit: int
    trace_id: str
    filters: QueryFilter
    rewrite: bool
    rerank: bool


def build_request(
    *,
    query: str,
    corpus: str,
    limit: int = 10,
    rewrite: bool = False,
    rerank: bool = True,
    filters: QueryFilter | None = None,
) -> QueryRequest:
    if not corpus:
        raise ValueError("corpus must be a non-empty string")
    q = unicodedata.normalize("NFC", query).strip()
    if len(q) > _MAX_QUERY_LEN:
        q = q[:_MAX_QUERY_LEN]
    if not q:
        raise ValueError("query must be non-empty after normalization")
    lim = max(1, min(limit, 100))
    return QueryRequest(
        query=q,
        corpus=corpus,
        limit=lim,
        trace_id=str(ULID()),
        filters=filters or QueryFilter(),
        rewrite=rewrite,
        rerank=rerank,
    )
