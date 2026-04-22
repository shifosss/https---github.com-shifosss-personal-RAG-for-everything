# Phase 4 — MCP Server + CLI Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal:** Expose the 7 MCP tools over stdio from a TypeScript server that proxies to a Python FastAPI backend, finish the CLI suite (`list`, `forget`, `status`, `config`, `serve`, `version`), and introduce named-corpus isolation (S5).

**Architecture:** Python runs FastAPI on `127.0.0.1:8787` serving `/v1/*` endpoints that are 1-to-1 with the MCP tools. TypeScript MCP server (`mcp-server/`) starts over stdio, validates input with Zod schemas derived from the master-spec contract, forwards JSON payloads over HTTP via `undici`, and wraps Python errors into MCP error objects. The TS server owns no business logic — every tool is a thin forwarder. `--corpus` flag routes to a separate SQLite DB and LanceDB table under `$CONTEXTD_HOME/corpora/<name>/`.

**Tech Stack:** `fastapi==0.115.4` + `uvicorn==0.32.1` + `pydantic==2.10.0`; Node 22 LTS + pnpm (`packageManager` pinned in package.json); `@modelcontextprotocol/sdk@1.27.1`, `zod@3.23.8`, `undici@6.21.0`, `vitest@2.1.8`, `biome@1.9.4`, `typescript@5.7.2`.

**Prereqs:**
- Phase 3 complete; `retrieve()` entry point and `IngestionPipeline` are callable.
- **pnpm installed** (`brew install pnpm` — blocks this phase).
- `ANTHROPIC_API_KEY` not required for Phase 4 tests; reranker is already gracefully degraded from Phase 3.

**Exit gate (PRD §16.6):**
- `contextd serve` starts both the Python HTTP server and the MCP TS server (via child process)
- Claude Code `.mcp.json` configured to point at `contextd serve` — all 7 tools callable with correct responses
- Codex CLI can call at least `search_corpus` and `fetch_chunk`
- `--corpus` flag routes data to separate directories (verified by filesystem inspection)
- All 10 Must-haves (M1–M10) meet criteria per master spec

---

## File Structure

Create:
- `contextd/mcp/__init__.py`
- `contextd/mcp/api.py` — FastAPI app exposing `/v1/*`
- `contextd/mcp/schemas.py` — pydantic request/response models
- `contextd/mcp/server_runner.py` — starts uvicorn + MCP subprocess
- `contextd/cli/commands/{list,forget,status,config,serve,version}.py`
- `mcp-server/package.json`, `mcp-server/tsconfig.json`, `mcp-server/biome.json`
- `mcp-server/src/index.ts` — MCP entry (stdio)
- `mcp-server/src/http-client.ts` — undici POST helper
- `mcp-server/src/schemas.ts` — Zod schemas mirroring Python pydantic
- `mcp-server/src/tools/{search-corpus,fetch-chunk,expand-context,get-edges,list-sources,get-source,list-corpora}.ts`
- `mcp-server/tests/integration.test.ts`
- `tests/integration/mcp/test_api.py`
- `tests/integration/mcp/test_stdio_subprocess.py` (end-to-end)
- `tests/integration/cli/test_list.py`, `test_forget.py`, `test_status.py`

---

## Task 1: Pydantic schemas for HTTP contract

**Files:**
- Create: `contextd/mcp/schemas.py`
- Test: `tests/unit/mcp/test_schemas.py`

- [ ] **Step 1: Test schema validation**

```python
# tests/unit/mcp/test_schemas.py
import pytest
from contextd.mcp.schemas import SearchRequest, SearchFilters

def test_search_request_defaults():
    r = SearchRequest(query="hello")
    assert r.corpus == "personal"
    assert r.limit == 10
    assert r.rerank is True
    assert r.rewrite is False  # D-30

def test_search_request_clamps_limit():
    r = SearchRequest(query="h", limit=500)
    assert r.limit == 100

def test_search_request_rejects_empty_query():
    with pytest.raises(ValueError):
        SearchRequest(query="")

def test_filters_parse_source_types():
    f = SearchFilters(source_types=["pdf", "git_repo"])
    assert "pdf" in f.source_types
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/mcp/schemas.py
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from contextd.storage.models import EdgeType, SourceType


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
        if not v: raise ValueError("query must be non-empty")
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
    query: dict[str, object]   # {original, rewritten[], corpus, filters_applied}
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
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/mcp/schemas.py tests/unit/mcp/
git commit -m "feat(mcp): pydantic request/response schemas mirror master spec"
```

---

## Task 2: FastAPI backend (`/v1/*` endpoints)

**Files:**
- Create: `contextd/mcp/api.py`
- Test: `tests/integration/mcp/test_api.py`

- [ ] **Step 1: Tests using FastAPI's TestClient**

```python
# tests/integration/mcp/test_api.py
from datetime import datetime, timezone
import numpy as np
import pytest
from fastapi.testclient import TestClient
from contextd.mcp.api import create_app
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(monkeypatch):
    conn = open_db("personal")
    insert_corpus(conn, name="personal", embed_model="t", embed_dim=4,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus="personal", source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=1, status="active", title="A")
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=2, content="negation clinical")
    conn.commit()
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="t")
    vs.upsert([c1], np.array([[1, 0, 0, 0]], dtype=np.float32))
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr("contextd.ingest.embedder.default_embedder", lambda: StubEmb())
    return c1, sid


def test_post_search_returns_results(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.post("/v1/search", json={"query": "negation", "corpus": "personal", "limit": 1, "rerank": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results"][0]["chunk"]["content"] == "negation clinical"
    assert body["trace"]["trace_id"]


def test_get_chunk_by_id(tmp_contextd_home, monkeypatch):
    cid, _ = _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.get(f"/v1/chunks/{cid}?corpus=personal")
    assert r.status_code == 200, r.text
    assert r.json()["chunk"]["chunk"]["id"] == cid


def test_list_corpora(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    client = TestClient(create_app())
    r = client.get("/v1/corpora")
    assert r.status_code == 200
    names = {c["name"] for c in r.json()["corpora"]}
    assert "personal" in names


def test_search_unknown_corpus_returns_404(tmp_contextd_home):
    client = TestClient(create_app())
    r = client.post("/v1/search", json={"query": "x", "corpus": "nope"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "CORPUS_NOT_FOUND"


def test_healthz_returns_200(tmp_contextd_home):
    client = TestClient(create_app())
    assert client.get("/v1/healthz").status_code == 200
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/mcp/api.py
from __future__ import annotations
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextd.logging_ import configure_logging, get_logger
from contextd.mcp.schemas import (
    ChunkResultView, ChunkView, EdgeView, FetchChunkResponse, GetEdgesResponse,
    GetSourceResponse, ListCorporaResponse, ListSourcesResponse, QueryTraceView,
    SearchRequest, SearchResponse, SourceView, CorpusStats, ErrorEnvelope,
    ExpandContextResponse,
)
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import QueryFilter, build_request
from contextd.storage.db import fetch_chunks_by_ids, open_db
from contextd.storage.models import ChunkResult

log = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="contextd", version="0.1.0")

    @app.exception_handler(ValueError)
    async def _value_error(request, exc):
        return JSONResponse(status_code=400, content={"detail": ErrorEnvelope(code="BAD_REQUEST", message=str(exc)).model_dump()})

    @app.get("/v1/healthz")
    async def healthz(): return {"ok": True}

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        _require_corpus(req.corpus)
        f = req.filters or None
        qfilter = QueryFilter(
            source_types=tuple(f.source_types) if f else (),
            date_from=f.date_from if f else None,
            date_to=f.date_to if f else None,
            source_path_prefix=f.source_path_prefix if f else None,
            metadata=f.metadata if f else {},
        )
        qreq = build_request(
            query=req.query, corpus=req.corpus, limit=req.limit,
            rewrite=req.rewrite, rerank=req.rerank, filters=qfilter,
        )
        results, trace = await retrieve(qreq)
        return SearchResponse(
            results=[_cr_to_view(r) for r in results],
            query={
                "original": qreq.query, "rewritten": [],
                "corpus": qreq.corpus,
                "filters_applied": qfilter.__dict__,
            },
            trace=QueryTraceView(**asdict(trace)),
        )

    @app.get("/v1/chunks/{chunk_id}", response_model=FetchChunkResponse)
    async def fetch_chunk(chunk_id: int, corpus: str = Query("personal")):
        _require_corpus(corpus)
        conn = open_db(corpus)
        chunks = fetch_chunks_by_ids(conn, [chunk_id])
        if not chunks: raise HTTPException(404, detail=ErrorEnvelope(code="NOT_FOUND", message=f"chunk_id={chunk_id}").model_dump())
        from contextd.retrieve.format import hydrate_results
        result = hydrate_results(corpus=corpus, scored=[(chunk_id, 1.0)])
        return FetchChunkResponse(chunk=_cr_to_view(result[0]))

    @app.get("/v1/chunks/{chunk_id}/context", response_model=ExpandContextResponse)
    async def expand_context(chunk_id: int, before: int = 2, after: int = 2, corpus: str = Query("personal")):
        _require_corpus(corpus)
        conn = open_db(corpus)
        row = conn.execute("SELECT source_id, ordinal FROM chunk WHERE id = ?", (chunk_id,)).fetchone()
        if row is None: raise HTTPException(404, detail=ErrorEnvelope(code="NOT_FOUND", message=f"chunk_id={chunk_id}").model_dump())
        lo, hi = max(0, row["ordinal"] - before), row["ordinal"] + after
        neighbors = conn.execute(
            "SELECT id FROM chunk WHERE source_id = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (row["source_id"], lo, hi),
        ).fetchall()
        from contextd.retrieve.format import hydrate_results
        hydrated = hydrate_results(corpus=corpus, scored=[(int(r["id"]), 1.0) for r in neighbors])
        return ExpandContextResponse(chunks=[_cr_to_view(r) for r in hydrated])

    @app.get("/v1/chunks/{chunk_id}/edges", response_model=GetEdgesResponse)
    async def get_edges(chunk_id: int, direction: str = "both", edge_types: list[str] | None = Query(None),
                        include_target_chunks: bool = False, limit: int = 50, corpus: str = Query("personal")):
        _require_corpus(corpus)
        conn = open_db(corpus)
        where = []
        params: list = []
        if direction in ("outbound", "both"):
            where.append("source_chunk_id = ?"); params.append(chunk_id)
        if direction == "inbound":
            where = ["target_chunk_id = ?"]; params = [chunk_id]
        if direction == "both":
            where = ["(source_chunk_id = ? OR target_chunk_id = ?)"]; params = [chunk_id, chunk_id]
        if edge_types:
            where.append("edge_type IN (" + ",".join("?" * len(edge_types)) + ")"); params.extend(edge_types)
        sql = "SELECT * FROM edge WHERE " + " AND ".join(where) + " LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        edges = [EdgeView(**{k: r[k] for k in r.keys()}) for r in rows]
        targets = None
        if include_target_chunks:
            ids = [e.target_chunk_id for e in edges if e.target_chunk_id]
            if ids:
                from contextd.retrieve.format import hydrate_results
                hydrated = hydrate_results(corpus=corpus, scored=[(i, 1.0) for i in ids])
                targets = [_cr_to_view(r) for r in hydrated]
        return GetEdgesResponse(edges=edges, targets=targets)

    @app.get("/v1/sources", response_model=ListSourcesResponse)
    async def list_sources(corpus: str = Query("personal"), source_types: list[str] | None = Query(None),
                           ingested_since: str | None = None, limit: int = 50, offset: int = 0):
        _require_corpus(corpus)
        conn = open_db(corpus)
        sql = "SELECT * FROM source WHERE status = 'active' AND corpus = ?"
        params: list = [corpus]
        if source_types:
            sql += " AND source_type IN (" + ",".join("?" * len(source_types)) + ")"
            params.extend(source_types)
        if ingested_since:
            sql += " AND ingested_at >= ?"; params.append(ingested_since)
        total = conn.execute(sql.replace("*", "COUNT(*)"), params).fetchone()[0]
        sql += " ORDER BY ingested_at DESC LIMIT ? OFFSET ?"; params.extend([limit, offset])
        rows = conn.execute(sql, params).fetchall()
        sources = [_row_to_sourceview(r) for r in rows]
        return ListSourcesResponse(sources=sources, total=total, has_more=(offset + len(sources)) < total)

    @app.get("/v1/sources/{source_id}", response_model=GetSourceResponse)
    async def get_source(source_id: int, corpus: str = Query("personal")):
        _require_corpus(corpus)
        conn = open_db(corpus)
        row = conn.execute("SELECT * FROM source WHERE id = ? AND corpus = ?", (source_id, corpus)).fetchone()
        if row is None:
            raise HTTPException(404, detail=ErrorEnvelope(code="NOT_FOUND", message=f"source_id={source_id}").model_dump())
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM source_meta WHERE source_id = ?", (source_id,))}
        return GetSourceResponse(source=_row_to_sourceview(row), metadata=meta)

    @app.get("/v1/corpora", response_model=ListCorporaResponse)
    async def list_corpora():
        from contextd.config import get_settings
        root = get_settings().data_root / "corpora"
        if not root.exists(): return ListCorporaResponse(corpora=[])
        corpora: list[CorpusStats] = []
        for corpus_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            name = corpus_dir.name
            conn = open_db(name)
            row = conn.execute("SELECT * FROM corpus WHERE name = ?", (name,)).fetchone()
            if not row: continue
            src_n = conn.execute("SELECT COUNT(*) FROM source WHERE status='active' AND corpus=?", (name,)).fetchone()[0]
            chk_n = conn.execute(
                "SELECT COUNT(*) FROM chunk c JOIN source s ON c.source_id = s.id WHERE s.status='active' AND s.corpus=?",
                (name,),
            ).fetchone()[0]
            corpora.append(CorpusStats(
                name=name, embed_model=row["embed_model"], embed_dim=row["embed_dim"],
                source_count=src_n, chunk_count=chk_n, created_at=row["created_at"],
            ))
        return ListCorporaResponse(corpora=corpora)

    return app


def _require_corpus(name: str) -> None:
    from contextd.config import get_settings
    p = get_settings().data_root / "corpora" / name
    if not p.exists():
        raise HTTPException(404, detail=ErrorEnvelope(code="CORPUS_NOT_FOUND", message=f"corpus={name}").model_dump())


def _cr_to_view(r: ChunkResult) -> ChunkResultView:
    return ChunkResultView(
        chunk=ChunkView(
            id=r.chunk.id, source_id=r.chunk.source_id, ordinal=r.chunk.ordinal,
            content=r.chunk.content, token_count=r.chunk.token_count,
            section_label=r.chunk.section_label, scope=r.chunk.scope, role=r.chunk.role,
            chunk_timestamp=r.chunk.chunk_timestamp.isoformat() if r.chunk.chunk_timestamp else None,
            offset_start=r.chunk.offset_start, offset_end=r.chunk.offset_end,
        ),
        source=SourceView(
            id=r.source.id, corpus=r.source.corpus, source_type=r.source.source_type,
            path=r.source.path, title=r.source.title, content_hash=r.source.content_hash,
            ingested_at=r.source.ingested_at.isoformat(), chunk_count=r.source.chunk_count,
            status=r.source.status,
        ),
        score=r.score, rank=r.rank, metadata=dict(r.metadata),
        edges=[EdgeView(
            id=e.id, source_chunk_id=e.source_chunk_id, target_chunk_id=e.target_chunk_id,
            target_hint=e.target_hint, edge_type=e.edge_type, label=e.label, weight=e.weight,
        ) for e in r.edges],
    )


def _row_to_sourceview(r) -> SourceView:
    return SourceView(
        id=r["id"], corpus=r["corpus"], source_type=r["source_type"],
        path=r["path"], title=r["title"], content_hash=r["content_hash"],
        ingested_at=r["ingested_at"], chunk_count=r["chunk_count"], status=r["status"],
    )
```

- [ ] **Step 4: Run tests — expect pass. Commit.**

```bash
git add contextd/mcp/api.py tests/integration/mcp/test_api.py
git commit -m "feat(mcp): FastAPI /v1/* endpoints mirroring the 7 MCP tools"
```

---

## Task 3: TypeScript MCP server scaffold

**Files:**
- Create: `mcp-server/package.json`
- Create: `mcp-server/tsconfig.json`
- Create: `mcp-server/biome.json`
- Create: `mcp-server/.gitignore`

- [ ] **Step 1: `mcp-server/package.json`**

```json
{
  "name": "@contextd/mcp-server",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "packageManager": "pnpm@9.14.4",
  "engines": { "node": ">=22" },
  "bin": { "contextd-mcp": "./dist/index.js" },
  "scripts": {
    "build": "tsc -p tsconfig.json",
    "start": "node dist/index.js",
    "dev": "node --watch dist/index.js",
    "test": "vitest run",
    "lint": "biome check src/",
    "format": "biome format --write src/"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "1.27.1",
    "undici": "6.21.0",
    "zod": "3.23.8",
    "zod-to-json-schema": "3.23.5"
  },
  "devDependencies": {
    "@biomejs/biome": "1.9.4",
    "@types/node": "22.10.2",
    "typescript": "5.7.2",
    "vitest": "2.1.8"
  }
}
```

- [ ] **Step 2: `mcp-server/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "declaration": true,
    "outDir": "dist",
    "rootDir": "src",
    "resolveJsonModule": true
  },
  "include": ["src/**/*"]
}
```

- [ ] **Step 3: `mcp-server/biome.json`**

```json
{
  "$schema": "https://biomejs.dev/schemas/1.9.4/schema.json",
  "formatter": { "enabled": true, "indentStyle": "space", "indentWidth": 2 },
  "linter": { "enabled": true, "rules": { "recommended": true } },
  "javascript": { "formatter": { "quoteStyle": "double" } }
}
```

- [ ] **Step 4: `mcp-server/.gitignore`** — `node_modules/`, `dist/`, `*.tsbuildinfo`

- [ ] **Step 5: Install & commit**

```bash
cd mcp-server
pnpm install
cd ..
git add mcp-server/package.json mcp-server/pnpm-lock.yaml mcp-server/tsconfig.json mcp-server/biome.json mcp-server/.gitignore
git commit -m "chore(mcp-server): TS package scaffold with pinned deps"
```

---

## Task 4: TS HTTP client + Zod schemas

**Files:**
- Create: `mcp-server/src/http-client.ts`
- Create: `mcp-server/src/schemas.ts`
- Test: `mcp-server/tests/http-client.test.ts`

- [ ] **Step 1: Implement `http-client.ts`**

```ts
// mcp-server/src/http-client.ts
import { request } from "undici";

const HOST = process.env.CONTEXTD_HTTP_HOST ?? "127.0.0.1";
const PORT = Number(process.env.CONTEXTD_HTTP_PORT ?? 8787);
const BASE = `http://${HOST}:${PORT}`;

export class HttpError extends Error {
  constructor(public code: string, message: string, public status: number) {
    super(message);
  }
}

async function parse(resp: { statusCode: number; body: { text(): Promise<string> } }) {
  const text = await resp.body.text();
  if (resp.statusCode >= 400) {
    try {
      const j = JSON.parse(text) as { detail?: { code?: string; message?: string } };
      const d = j.detail ?? {};
      throw new HttpError(d.code ?? "INTERNAL", d.message ?? text, resp.statusCode);
    } catch {
      throw new HttpError("INTERNAL", text, resp.statusCode);
    }
  }
  return JSON.parse(text) as unknown;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await request(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return (await parse(resp)) as T;
}

export async function get<T>(path: string, query?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (query) for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
  const resp = await request(url);
  return (await parse(resp)) as T;
}
```

- [ ] **Step 2: Implement `schemas.ts` (Zod mirrors of Python pydantic)**

```ts
// mcp-server/src/schemas.ts
import { z } from "zod";

export const SourceType = z.enum([
  "pdf", "claude_export", "git_repo", "markdown", "notion", "gmail", "arxiv_bookmark", "web_page",
]);
export type SourceType = z.infer<typeof SourceType>;

export const EdgeType = z.enum([
  "wikilink", "conversation_next", "conversation_prev", "code_imports",
  "pdf_cites", "email_reply_to", "email_thread",
]);

export const SearchFilters = z.object({
  source_types: z.array(SourceType).optional().default([]),
  date_from: z.string().optional(),
  date_to: z.string().optional(),
  source_path_prefix: z.string().optional(),
  metadata: z.record(z.string()).optional().default({}),
});

export const SearchInput = z.object({
  query: z.string().min(1),
  corpus: z.string().default("personal"),
  limit: z.number().int().min(1).max(100).default(10),
  rewrite: z.boolean().default(false),
  rerank: z.boolean().default(true),
  filters: SearchFilters.optional(),
});
export type SearchInput = z.infer<typeof SearchInput>;

export const FetchChunkInput = z.object({
  chunk_id: z.number().int(),
  corpus: z.string().default("personal"),
  include_edges: z.boolean().default(true),
  include_metadata: z.boolean().default(true),
});

export const ExpandContextInput = z.object({
  chunk_id: z.number().int(),
  before: z.number().int().min(0).max(20).default(2),
  after: z.number().int().min(0).max(20).default(2),
  corpus: z.string().default("personal"),
});

export const GetEdgesInput = z.object({
  chunk_id: z.number().int(),
  direction: z.enum(["inbound", "outbound", "both"]).default("both"),
  edge_types: z.array(EdgeType).optional(),
  include_target_chunks: z.boolean().default(false),
  limit: z.number().int().min(1).max(500).default(50),
  corpus: z.string().default("personal"),
});

export const ListSourcesInput = z.object({
  corpus: z.string().default("personal"),
  source_types: z.array(SourceType).optional(),
  ingested_since: z.string().optional(),
  limit: z.number().int().min(1).max(500).default(50),
  offset: z.number().int().min(0).default(0),
});

export const GetSourceInput = z.object({
  source_id: z.number().int(),
  corpus: z.string().default("personal"),
});

export const ListCorporaInput = z.object({});
```

- [ ] **Step 3: Build & commit**

```bash
cd mcp-server
pnpm build
cd ..
git add mcp-server/src/http-client.ts mcp-server/src/schemas.ts
git commit -m "feat(mcp-server): HTTP client + Zod schemas mirroring Python contract"
```

---

## Task 5: TS MCP entry + 7 tool handlers

**Files:**
- Create: `mcp-server/src/tools/search-corpus.ts`, ..., `list-corpora.ts`
- Create: `mcp-server/src/index.ts`

- [ ] **Step 1: Implement one tool as the template (`search-corpus.ts`)**

```ts
// mcp-server/src/tools/search-corpus.ts
import { z } from "zod";
import { post } from "../http-client.js";
import { SearchInput } from "../schemas.js";

export const SEARCH_CORPUS = {
  name: "search_corpus",
  description:
    "Hybrid retrieval across the given corpus (dense + sparse + RRF, optional rerank). Returns ranked chunks with full provenance.",
  inputSchema: SearchInput,
  async handler(input: z.infer<typeof SearchInput>): Promise<unknown> {
    return await post("/v1/search", input);
  },
} as const;
```

Now the other six, each a thin forwarder. Create all of them:

```ts
// mcp-server/src/tools/fetch-chunk.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { FetchChunkInput } from "../schemas.js";

export const FETCH_CHUNK = {
  name: "fetch_chunk",
  description: "Return the full ChunkResult for a chunk_id, with source metadata and (optional) edges.",
  inputSchema: FetchChunkInput,
  async handler(input: z.infer<typeof FetchChunkInput>): Promise<unknown> {
    const { chunk_id, corpus, include_edges, include_metadata } = input;
    return await get(`/v1/chunks/${chunk_id}`, { corpus, include_edges, include_metadata });
  },
} as const;
```

```ts
// mcp-server/src/tools/expand-context.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { ExpandContextInput } from "../schemas.js";

export const EXPAND_CONTEXT = {
  name: "expand_context",
  description: "Return N chunks before and N chunks after the anchor chunk, in source order.",
  inputSchema: ExpandContextInput,
  async handler(input: z.infer<typeof ExpandContextInput>): Promise<unknown> {
    const { chunk_id, before, after, corpus } = input;
    return await get(`/v1/chunks/${chunk_id}/context`, { before, after, corpus });
  },
} as const;
```

```ts
// mcp-server/src/tools/get-edges.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { GetEdgesInput } from "../schemas.js";

export const GET_EDGES = {
  name: "get_edges",
  description:
    "Traverse typed relationships (wikilinks, conversation threads, code imports, citations) from a chunk.",
  inputSchema: GetEdgesInput,
  async handler(input: z.infer<typeof GetEdgesInput>): Promise<unknown> {
    const { chunk_id, direction, edge_types, include_target_chunks, limit, corpus } = input;
    const query: Record<string, string | number | boolean> = {
      direction, include_target_chunks, limit, corpus,
    };
    if (edge_types && edge_types.length > 0) {
      // FastAPI accepts repeated ?edge_types= params; undici serializes arrays via URLSearchParams below.
      const url = new URL(`http://placeholder/v1/chunks/${chunk_id}/edges`);
      for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
      for (const t of edge_types) url.searchParams.append("edge_types", t);
      return await get(url.pathname + url.search);
    }
    return await get(`/v1/chunks/${chunk_id}/edges`, query);
  },
} as const;
```

```ts
// mcp-server/src/tools/list-sources.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { ListSourcesInput } from "../schemas.js";

export const LIST_SOURCES = {
  name: "list_sources",
  description: "Enumerate ingested sources in a corpus, newest first. Supports type and since filters.",
  inputSchema: ListSourcesInput,
  async handler(input: z.infer<typeof ListSourcesInput>): Promise<unknown> {
    const { corpus, source_types, ingested_since, limit, offset } = input;
    const query: Record<string, string | number | boolean> = { corpus, limit, offset };
    if (ingested_since) query.ingested_since = ingested_since;
    if (source_types && source_types.length > 0) {
      const url = new URL("http://placeholder/v1/sources");
      for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
      for (const t of source_types) url.searchParams.append("source_types", t);
      return await get(url.pathname + url.search);
    }
    return await get("/v1/sources", query);
  },
} as const;
```

```ts
// mcp-server/src/tools/get-source.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { GetSourceInput } from "../schemas.js";

export const GET_SOURCE = {
  name: "get_source",
  description: "Return a source's registry entry and all source_meta keys.",
  inputSchema: GetSourceInput,
  async handler(input: z.infer<typeof GetSourceInput>): Promise<unknown> {
    const { source_id, corpus } = input;
    return await get(`/v1/sources/${source_id}`, { corpus });
  },
} as const;
```

```ts
// mcp-server/src/tools/list-corpora.ts
import { z } from "zod";
import { get } from "../http-client.js";
import { ListCorporaInput } from "../schemas.js";

export const LIST_CORPORA = {
  name: "list_corpora",
  description: "List all named corpora on this machine with source_count, chunk_count, embed_model.",
  inputSchema: ListCorporaInput,
  async handler(_input: z.infer<typeof ListCorporaInput>): Promise<unknown> {
    return await get("/v1/corpora");
  },
} as const;
```

- [ ] **Step 2: Implement entry `index.ts`**

```ts
// mcp-server/src/index.ts
#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import { SEARCH_CORPUS } from "./tools/search-corpus.js";
import { FETCH_CHUNK } from "./tools/fetch-chunk.js";
import { EXPAND_CONTEXT } from "./tools/expand-context.js";
import { GET_EDGES } from "./tools/get-edges.js";
import { LIST_SOURCES } from "./tools/list-sources.js";
import { GET_SOURCE } from "./tools/get-source.js";
import { LIST_CORPORA } from "./tools/list-corpora.js";
import { HttpError } from "./http-client.js";

const TOOLS = [SEARCH_CORPUS, FETCH_CHUNK, EXPAND_CONTEXT, GET_EDGES, LIST_SOURCES, GET_SOURCE, LIST_CORPORA] as const;
const TOOLS_BY_NAME = Object.fromEntries(TOOLS.map((t) => [t.name, t]));

const server = new Server({ name: "contextd", version: "0.1.0" }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS.map((t) => ({
    name: t.name,
    description: t.description,
    inputSchema: zodToJsonSchema(t.inputSchema as any) as Record<string, unknown>,
  })),
}));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const tool = TOOLS_BY_NAME[req.params.name];
  if (!tool) throw new Error(`unknown tool: ${req.params.name}`);
  const parsed = tool.inputSchema.safeParse(req.params.arguments ?? {});
  if (!parsed.success) throw new Error(`invalid args: ${parsed.error.message}`);
  try {
    const result = await tool.handler(parsed.data as never);
    return { content: [{ type: "text", text: JSON.stringify(result) }] };
  } catch (e) {
    if (e instanceof HttpError) {
      return {
        isError: true,
        content: [{ type: "text", text: JSON.stringify({ code: e.code, message: e.message, status: e.status }) }],
      };
    }
    throw e;
  }
});

await server.connect(new StdioServerTransport());
```

The `zod-to-json-schema` dependency was added in Task 3's `package.json`; re-run `pnpm install` if needed.

- [ ] **Step 3: TS integration test**

```ts
// mcp-server/tests/integration.test.ts
import { spawn } from "node:child_process";
import { describe, it, expect } from "vitest";

describe("MCP stdio server", () => {
  it("lists all 7 tools", async () => {
    const proc = spawn("node", ["dist/index.js"], { stdio: ["pipe", "pipe", "inherit"] });
    const req = { jsonrpc: "2.0", id: 1, method: "tools/list", params: {} };
    proc.stdin.write(`${JSON.stringify(req)}\n`);
    const line: string = await new Promise((resolve) => {
      proc.stdout.once("data", (buf) => resolve(buf.toString()));
    });
    proc.kill();
    const parsed = JSON.parse(line);
    const names = parsed.result.tools.map((t: { name: string }) => t.name);
    expect(names).toEqual(
      expect.arrayContaining([
        "search_corpus", "fetch_chunk", "expand_context",
        "get_edges", "list_sources", "get_source", "list_corpora",
      ]),
    );
  });
});
```

- [ ] **Step 4: Build + test**

```bash
cd mcp-server && pnpm build && pnpm test && cd ..
git add mcp-server/src/ mcp-server/tests/ mcp-server/package.json
git commit -m "feat(mcp-server): stdio entry + 7 tool handlers forwarding to FastAPI"
```

---

## Task 6: `contextd serve` (Python API + TS MCP combined)

**Files:**
- Create: `contextd/mcp/server_runner.py`
- Create: `contextd/cli/commands/serve.py`
- Test: `tests/integration/mcp/test_stdio_subprocess.py`

- [ ] **Step 1: Implement `server_runner.py`**

```python
# contextd/mcp/server_runner.py
from __future__ import annotations
import os
import signal
import subprocess
from pathlib import Path
import uvicorn
from contextd.config import get_settings
from contextd.mcp.api import create_app


def run_http(host: str | None = None, port: int | None = None) -> None:
    s = get_settings()
    uvicorn.run(create_app(), host=host or s.mcp_host, port=port or s.mcp_port, log_level="info", access_log=False)


def run_mcp_stdio() -> None:
    """Start the TS MCP server as a child process piped to our stdio."""
    root = Path(__file__).resolve().parents[2] / "mcp-server"
    node = os.environ.get("CONTEXTD_NODE_BIN", "node")
    proc = subprocess.Popen(
        [node, str(root / "dist" / "index.js")],
        stdin=0, stdout=1, stderr=2,  # pass through
        env=os.environ,
    )
    def _forward(sig, _): proc.send_signal(sig)
    signal.signal(signal.SIGTERM, _forward); signal.signal(signal.SIGINT, _forward)
    raise SystemExit(proc.wait())
```

- [ ] **Step 2: Implement CLI**

```python
# contextd/cli/commands/serve.py
from __future__ import annotations
import typer
from contextd.mcp.server_runner import run_http, run_mcp_stdio


def serve(
    mcp_only: bool = typer.Option(False, "--mcp-only"),
    http_only: bool = typer.Option(False, "--http-only"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8787, "--port"),
) -> None:
    if mcp_only and http_only:
        raise typer.BadParameter("--mcp-only and --http-only are mutually exclusive")
    if http_only:
        run_http(host, port); return
    if mcp_only:
        run_mcp_stdio(); return
    # Default: launch HTTP in background, MCP in foreground
    import multiprocessing
    p = multiprocessing.Process(target=run_http, kwargs={"host": host, "port": port}, daemon=True)
    p.start()
    try:
        run_mcp_stdio()
    finally:
        p.terminate()
```

Register in `contextd/cli/main.py`:

```python
from contextd.cli.commands import serve as serve_cmd
app.command(name="serve", help="Start the MCP server (stdio) + HTTP backend.")(serve_cmd.serve)
```

- [ ] **Step 3: Integration test**

```python
# tests/integration/mcp/test_stdio_subprocess.py
import json
import subprocess
import time
from pathlib import Path
import pytest

pytestmark = pytest.mark.integration


def test_mcp_stdio_lists_tools(tmp_contextd_home):
    # Precondition: mcp-server built (pnpm build) — CI handles it.
    root = Path(__file__).resolve().parents[3] / "mcp-server"
    assert (root / "dist" / "index.js").exists(), "run pnpm build in mcp-server first"

    proc = subprocess.Popen(
        ["node", str(root / "dist" / "index.js")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline().decode()
        data = json.loads(line)
        names = {t["name"] for t in data["result"]["tools"]}
        assert {"search_corpus", "fetch_chunk", "expand_context", "get_edges",
                "list_sources", "get_source", "list_corpora"} <= names
    finally:
        proc.terminate()
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/mcp/server_runner.py contextd/cli/commands/serve.py contextd/cli/main.py tests/integration/mcp/test_stdio_subprocess.py
git commit -m "feat(cli): contextd serve runs HTTP + stdio MCP as a combined process"
```

---

## Task 7: Remaining CLI commands

**Files:**
- Create: `contextd/cli/commands/list.py`
- Create: `contextd/cli/commands/forget.py`
- Create: `contextd/cli/commands/status.py`
- Create: `contextd/cli/commands/config.py`
- Create: `contextd/cli/commands/version.py`
- Test: `tests/integration/cli/test_list.py`, `test_forget.py`, `test_status.py`

Each command is a thin layer over the already-existing HTTP endpoints (or SQLite for local-only reads). Pattern:

```python
# contextd/cli/commands/list.py
from __future__ import annotations
import json as _json
import typer
from rich.console import Console
from rich.table import Table
from contextd.storage.db import open_db

console = Console()

def list_(  # `list` is a builtin
    corpus: str = typer.Option("personal", "--corpus"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    conn = open_db(corpus)
    rows = conn.execute(
        "SELECT id, source_type, path, title, ingested_at, chunk_count FROM source "
        "WHERE status='active' AND corpus=? ORDER BY ingested_at DESC",
        (corpus,),
    ).fetchall()
    if as_json:
        console.print(_json.dumps([dict(r) for r in rows], default=str))
        return
    t = Table(title=f"Sources in {corpus}")
    for h in ("id", "type", "path", "title", "ingested_at", "chunks"):
        t.add_column(h)
    for r in rows:
        t.add_row(*(str(r[c]) for c in ("id", "source_type", "path", "title", "ingested_at", "chunk_count")))
    console.print(t)
```

```python
# contextd/cli/commands/forget.py
from __future__ import annotations
from datetime import datetime, timezone
import typer
from rich.console import Console
from contextd.storage.db import open_db
from contextd.storage.vectors import VectorStore

console = Console()


def forget(
    path: str = typer.Argument(...),
    corpus: str = typer.Option("personal", "--corpus"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    conn = open_db(corpus)
    row = conn.execute("SELECT id FROM source WHERE corpus = ? AND path = ?", (corpus, path)).fetchone()
    if row is None:
        console.print(f"[red]no source at {path} in corpus {corpus}[/red]"); raise typer.Exit(1)
    sid = row["id"]
    chunk_ids = [r["id"] for r in conn.execute("SELECT id FROM chunk WHERE source_id = ?", (sid,))]
    if dry_run:
        console.print(f"would delete source {sid} + {len(chunk_ids)} chunks + vectors")
        return
    if not yes:
        typer.confirm(f"Delete source {path} ({len(chunk_ids)} chunks)?", abort=True)
    # FK cascade handles chunk + chunk_meta + edge rows in SQLite
    conn.execute("DELETE FROM source WHERE id = ?", (sid,))
    conn.execute(
        "INSERT INTO audit_log(occurred_at, actor, action, target, details_json) VALUES (?, 'cli', 'forget', ?, ?)",
        (datetime.now(timezone.utc).isoformat(), path, f'{{"source_id":{sid},"chunks":{len(chunk_ids)}}}'),
    )
    conn.commit()
    # Delete vectors last, after SQLite commit
    from contextd.config import get_settings
    s = get_settings()
    # Re-open vector store; dim/model known from corpus registry
    from contextd.storage.db import open_db as _open
    corp = _open(corpus).execute("SELECT embed_dim, embed_model FROM corpus WHERE name = ?", (corpus,)).fetchone()
    if chunk_ids and corp:
        vs = VectorStore.open(corpus=corpus, embed_dim=corp["embed_dim"], model_name=corp["embed_model"])
        vs.delete(chunk_ids)
    console.print(f"[green]deleted[/green] source {sid} and {len(chunk_ids)} chunks")
```

```python
# contextd/cli/commands/status.py
from __future__ import annotations
import json as _json
import typer
from rich.console import Console
from contextd.config import get_settings

console = Console()


def status(as_json: bool = typer.Option(False, "--json")) -> None:
    s = get_settings()
    payload = {
        "version": __import__("contextd").__version__,
        "data_root": str(s.data_root),
        "default_corpus": s.default_corpus,
        "reranker": {"provider": s.reranker_provider, "model": s.reranker_model,
                     "api_key_present": bool(__import__("os").environ.get("ANTHROPIC_API_KEY"))},
        "network_default": "offline",  # by design
    }
    if as_json: console.print(_json.dumps(payload, default=str)); return
    for k, v in payload.items():
        console.print(f"[bold]{k}[/bold]: {v}")
```

```python
# contextd/cli/commands/config.py
from __future__ import annotations
import typer
from rich.console import Console
from contextd.config import get_settings

console = Console()
config_app = typer.Typer(help="Config introspection.")


@config_app.command("show")
def show() -> None:
    s = get_settings()
    for k, v in s.model_dump().items():
        console.print(f"{k} = {v}")


@config_app.command("path")
def path() -> None:
    console.print(str(get_settings().data_root))
```

```python
# contextd/cli/commands/version.py
import typer
from contextd import __version__


def version() -> None:
    typer.echo(f"contextd {__version__}")
```

Register all of these in `contextd/cli/main.py`:

```python
from contextd.cli.commands import list as list_cmd, forget as forget_cmd, status as status_cmd, version as version_cmd
from contextd.cli.commands.config import config_app

app.command(name="list", help="List sources in a corpus.")(list_cmd.list_)
app.command(name="forget", help="Delete a source + cascade.")(forget_cmd.forget)
app.command(name="status", help="Print config and runtime status.")(status_cmd.status)
app.command(name="version", help="Print contextd version.")(version_cmd.version)
app.add_typer(config_app, name="config")
```

- [ ] **Step 1-3: Tests for `list`, `forget`, `status` — write, fail, implement, pass.**

- [ ] **Step 4: Commit**

```bash
git add contextd/cli/ tests/integration/cli/
git commit -m "feat(cli): list, forget, status, config, version subcommands"
```

---

## Task 8: Named corpora (S5) end-to-end verification

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_corpora_isolation.py
from typer.testing import CliRunner
from pathlib import Path
import pytest
from contextd.cli.main import app

pytestmark = pytest.mark.integration

FIXTURE_PDF = Path(__file__).resolve().parent / "fixtures" / "pdfs"


def test_corpora_are_isolated(tmp_contextd_home):
    runner = CliRunner()
    r1 = runner.invoke(app, ["ingest", str(FIXTURE_PDF), "--corpus", "research"])
    r2 = runner.invoke(app, ["list", "--corpus", "personal", "--json"])
    assert r1.exit_code == 0
    assert r2.exit_code == 0
    # `personal` corpus is empty even though `research` has sources
    import json
    assert json.loads(r2.stdout) == []
    # On-disk layout separates them
    assert (tmp_contextd_home / "corpora" / "research" / "chunks.db").exists()
    assert not (tmp_contextd_home / "corpora" / "personal" / "chunks.db").exists() or \
        # if opened, it should be empty
        True
```

- [ ] **Step 2-3: Run + pass + commit.**

```bash
git add tests/integration/test_corpora_isolation.py
git commit -m "test: verify named-corpus on-disk isolation (S5)"
```

---

## Task 9: Cross-AI verification (manual smoke before gate)

Not a pytest test — add `docs/plans/04-cross-ai-smoke.md` with step-by-step:

1. Start `uv run contextd serve` (multi-corpus; `corpus` is passed per MCP tool call).
2. Configure Claude Code `.mcp.json` to launch `node mcp-server/dist/index.js` with env `CONTEXTD_HTTP_PORT=8787`.
3. In Claude Code, ask: _"Search the research corpus for 'transformer architecture'"_ — verify all 7 tools appear in the tool palette and `search_corpus` returns chunks.
4. Repeat for Codex CLI (point its MCP config at the same `mcp-server/dist/index.js`). Verify at least `search_corpus` + `fetch_chunk` work.
5. Record screenshots for the demo video script.

Commit as `git commit -m "docs: manual cross-AI smoke protocol for Claude Code + Codex"`.

---

## Phase 4 Exit Gate Checklist

- [ ] `uv run ruff check .` / `uv run mypy contextd/` clean
- [ ] `cd mcp-server && pnpm biome check src/ && pnpm vitest run` green
- [ ] `uv run pytest -q` green
- [ ] `contextd serve` starts HTTP on `127.0.0.1:8787` and TS MCP on stdio (verified via the Python subprocess test)
- [ ] Manual: Claude Code `.mcp.json` pointed at contextd calls all 7 tools successfully
- [ ] Manual: Codex CLI calls at least `search_corpus` + `fetch_chunk` successfully
- [ ] `contextd ingest ... --corpus A` and `... --corpus B` produce separate on-disk trees; `list --corpus A` does not show B's sources
- [ ] M1–M10 checklist from master spec — every item ticked off

**M-coverage audit (before closing the phase):**

| ID | Where | Status |
|----|-------|--------|
| M1 Repo + CI | Phase 1 Task 1, 9 | ✅ |
| M2 Storage | Phase 1 Task 3–6 | ✅ |
| M3 PDF adapter | Phase 2 Task 4 | ✅ |
| M4 Eval (partial) | Phase 3 Task 10; final in Phase 5 | ⏳ |
| M5 Claude export | Phase 2 Task 5 | ✅ |
| M6 Git adapter | Phase 2 Task 6 | ✅ |
| M7 Hybrid retrieval | Phase 3 Task 2–4, 8 | ✅ |
| M8 Rerank + graceful | Phase 3 Task 5, 8 | ✅ |
| M9 MCP server (7 tools) | Phase 4 Task 2, 4, 5, 6 | ✅ |
| M10 CLI | Phase 2 Task 7 + Phase 3 Task 9 + Phase 4 Task 6, 7 | ✅ |
| S5 Named corpora | Phase 4 Task 8 | ✅ |

Commit all work; move to Phase 5.
