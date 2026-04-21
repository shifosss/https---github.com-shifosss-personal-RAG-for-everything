from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from contextd.config import get_settings
from contextd.logging_ import configure_logging, get_logger
from contextd.mcp.schemas import (
    ChunkResultView,
    ChunkView,
    CorpusStats,
    EdgeView,
    ErrorEnvelope,
    ExpandContextResponse,
    FetchChunkResponse,
    GetEdgesResponse,
    GetSourceResponse,
    ListCorporaResponse,
    ListSourcesResponse,
    QueryTraceView,
    SearchRequest,
    SearchResponse,
    SourceView,
)
from contextd.retrieve.format import hydrate_results
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import QueryFilter, build_request
from contextd.storage.db import fetch_chunks_by_ids, open_db

if TYPE_CHECKING:
    import sqlite3

    from contextd.storage.models import ChunkResult

log = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="contextd", version="0.1.0")

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": ErrorEnvelope(code="BAD_REQUEST", message=str(exc)).model_dump()},
        )

    @app.get("/v1/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        _require_corpus(req.corpus)
        f = req.filters
        qfilter = QueryFilter(
            source_types=tuple(f.source_types) if f else (),
            date_from=f.date_from if f else None,
            date_to=f.date_to if f else None,
            source_path_prefix=f.source_path_prefix if f else None,
            metadata=f.metadata if f else {},
        )
        qreq = build_request(
            query=req.query,
            corpus=req.corpus,
            limit=req.limit,
            rewrite=req.rewrite,
            rerank=req.rerank,
            filters=qfilter,
        )
        results, trace = await retrieve(qreq)
        return SearchResponse(
            results=[_cr_to_view(r) for r in results],
            query={
                "original": qreq.query,
                "rewritten": [],
                "corpus": qreq.corpus,
                "filters_applied": qfilter.__dict__,
            },
            trace=QueryTraceView(**asdict(trace)),
        )

    @app.get("/v1/chunks/{chunk_id}", response_model=FetchChunkResponse)
    async def fetch_chunk(chunk_id: int, corpus: str = Query("personal")) -> FetchChunkResponse:
        _require_corpus(corpus)
        conn = open_db(corpus)
        chunks = fetch_chunks_by_ids(conn, [chunk_id])
        if not chunks:
            raise HTTPException(
                404,
                detail=ErrorEnvelope(code="NOT_FOUND", message=f"chunk_id={chunk_id}").model_dump(),
            )
        result = hydrate_results(corpus=corpus, scored=[(chunk_id, 1.0)])
        return FetchChunkResponse(chunk=_cr_to_view(result[0]))

    @app.get("/v1/chunks/{chunk_id}/context", response_model=ExpandContextResponse)
    async def expand_context(
        chunk_id: int,
        before: int = 2,
        after: int = 2,
        corpus: str = Query("personal"),
    ) -> ExpandContextResponse:
        _require_corpus(corpus)
        conn = open_db(corpus)
        row = conn.execute(
            "SELECT source_id, ordinal FROM chunk WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(
                404,
                detail=ErrorEnvelope(code="NOT_FOUND", message=f"chunk_id={chunk_id}").model_dump(),
            )
        lo, hi = max(0, row["ordinal"] - before), row["ordinal"] + after
        neighbors = conn.execute(
            "SELECT id FROM chunk WHERE source_id = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (row["source_id"], lo, hi),
        ).fetchall()
        hydrated = hydrate_results(corpus=corpus, scored=[(int(r["id"]), 1.0) for r in neighbors])
        return ExpandContextResponse(chunks=[_cr_to_view(r) for r in hydrated])

    @app.get("/v1/chunks/{chunk_id}/edges", response_model=GetEdgesResponse)
    async def get_edges(
        chunk_id: int,
        direction: str = "both",
        edge_types: Annotated[list[str] | None, Query()] = None,
        include_target_chunks: bool = False,
        limit: int = 50,
        corpus: str = Query("personal"),
    ) -> GetEdgesResponse:
        _require_corpus(corpus)
        conn = open_db(corpus)
        where: list[str] = []
        params: list[Any] = []
        if direction == "outbound":
            where.append("source_chunk_id = ?")
            params.append(chunk_id)
        elif direction == "inbound":
            where.append("target_chunk_id = ?")
            params.append(chunk_id)
        else:  # both
            where.append("(source_chunk_id = ? OR target_chunk_id = ?)")
            params.extend([chunk_id, chunk_id])
        if edge_types:
            where.append("edge_type IN (" + ",".join("?" * len(edge_types)) + ")")
            params.extend(edge_types)
        sql = "SELECT * FROM edge WHERE " + " AND ".join(where) + " LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        edges = [EdgeView(**{k: r[k] for k in r}) for r in rows]
        targets = None
        if include_target_chunks:
            ids = [e.target_chunk_id for e in edges if e.target_chunk_id]
            if ids:
                hydrated = hydrate_results(corpus=corpus, scored=[(i, 1.0) for i in ids])
                targets = [_cr_to_view(r) for r in hydrated]
        return GetEdgesResponse(edges=edges, targets=targets)

    @app.get("/v1/sources", response_model=ListSourcesResponse)
    async def list_sources(
        corpus: str = Query("personal"),
        source_types: Annotated[list[str] | None, Query()] = None,
        ingested_since: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ListSourcesResponse:
        _require_corpus(corpus)
        conn = open_db(corpus)
        where = "status = 'active' AND corpus = ?"
        params: list[Any] = [corpus]
        if source_types:
            where += " AND source_type IN (" + ",".join("?" * len(source_types)) + ")"
            params.extend(source_types)
        if ingested_since:
            where += " AND ingested_at >= ?"
            params.append(ingested_since)
        total = conn.execute(f"SELECT COUNT(*) FROM source WHERE {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM source WHERE {where} ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        sources = [_row_to_sourceview(r) for r in rows]
        return ListSourcesResponse(
            sources=sources,
            total=total,
            has_more=(offset + len(sources)) < total,
        )

    @app.get("/v1/sources/{source_id}", response_model=GetSourceResponse)
    async def get_source(source_id: int, corpus: str = Query("personal")) -> GetSourceResponse:
        _require_corpus(corpus)
        conn = open_db(corpus)
        row = conn.execute(
            "SELECT * FROM source WHERE id = ? AND corpus = ?", (source_id, corpus)
        ).fetchone()
        if row is None:
            raise HTTPException(
                404,
                detail=ErrorEnvelope(
                    code="NOT_FOUND", message=f"source_id={source_id}"
                ).model_dump(),
            )
        meta = {
            r["key"]: r["value"]
            for r in conn.execute(
                "SELECT key, value FROM source_meta WHERE source_id = ?", (source_id,)
            )
        }
        return GetSourceResponse(source=_row_to_sourceview(row), metadata=meta)

    @app.get("/v1/corpora", response_model=ListCorporaResponse)
    async def list_corpora() -> ListCorporaResponse:
        root = get_settings().data_root / "corpora"
        if not root.exists():
            return ListCorporaResponse(corpora=[])
        corpora: list[CorpusStats] = []
        for corpus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            name = corpus_dir.name
            conn = open_db(name)
            row = conn.execute("SELECT * FROM corpus WHERE name = ?", (name,)).fetchone()
            if not row:
                continue
            src_n = conn.execute(
                "SELECT COUNT(*) FROM source WHERE status='active' AND corpus=?",
                (name,),
            ).fetchone()[0]
            chk_n = conn.execute(
                "SELECT COUNT(*) FROM chunk c JOIN source s ON c.source_id = s.id"
                " WHERE s.status='active' AND s.corpus=?",
                (name,),
            ).fetchone()[0]
            created_at = row["created_at"]
            if not isinstance(created_at, str):
                created_at = str(created_at)
            corpora.append(
                CorpusStats(
                    name=name,
                    embed_model=row["embed_model"],
                    embed_dim=row["embed_dim"],
                    source_count=src_n,
                    chunk_count=chk_n,
                    created_at=created_at,
                )
            )
        return ListCorporaResponse(corpora=corpora)

    return app


def _require_corpus(name: str) -> None:
    p = get_settings().data_root / "corpora" / name
    if not p.exists():
        raise HTTPException(
            404,
            detail=ErrorEnvelope(code="CORPUS_NOT_FOUND", message=f"corpus={name}").model_dump(),
        )


def _cr_to_view(r: ChunkResult) -> ChunkResultView:
    return ChunkResultView(
        chunk=ChunkView(
            id=r.chunk.id,
            source_id=r.chunk.source_id,
            ordinal=r.chunk.ordinal,
            content=r.chunk.content,
            token_count=r.chunk.token_count,
            section_label=r.chunk.section_label,
            scope=r.chunk.scope,
            role=r.chunk.role,
            chunk_timestamp=(
                r.chunk.chunk_timestamp.isoformat() if r.chunk.chunk_timestamp else None
            ),
            offset_start=r.chunk.offset_start,
            offset_end=r.chunk.offset_end,
        ),
        source=SourceView(
            id=r.source.id,
            corpus=r.source.corpus,
            source_type=r.source.source_type,
            path=r.source.path,
            title=r.source.title,
            content_hash=r.source.content_hash,
            ingested_at=(
                r.source.ingested_at.isoformat()
                if hasattr(r.source.ingested_at, "isoformat")
                else str(r.source.ingested_at)
            ),
            chunk_count=r.source.chunk_count,
            status=r.source.status,
        ),
        score=r.score,
        rank=r.rank,
        metadata=dict(r.metadata),
        edges=[
            EdgeView(
                id=e.id,
                source_chunk_id=e.source_chunk_id,
                target_chunk_id=e.target_chunk_id,
                target_hint=e.target_hint,
                edge_type=e.edge_type,
                label=e.label,
                weight=e.weight,
            )
            for e in r.edges
        ],
    )


def _row_to_sourceview(r: sqlite3.Row) -> SourceView:
    ingested_at = r["ingested_at"]
    if not isinstance(ingested_at, str):
        ingested_at = str(ingested_at)
    return SourceView(
        id=r["id"],
        corpus=r["corpus"],
        source_type=r["source_type"],
        path=r["path"],
        title=r["title"],
        content_hash=r["content_hash"],
        ingested_at=ingested_at,
        chunk_count=r["chunk_count"],
        status=r["status"],
    )
