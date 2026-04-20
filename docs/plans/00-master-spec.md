# contextd v0.1 — Master Spec & Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement the per-phase plans task-by-task. All PRD references are to `docs/PRDs/part*_*.md`.

**Goal:** Ship a local-first, MCP-first personal RAG server unifying PDFs, Claude exports, and git repos behind seven MCP tools, in 18 engineering hours across five strict-order phases.

**Architecture:** Python owns storage/embeddings/ingestion/retrieval (FastAPI on localhost); TypeScript owns the MCP surface (stdio transport, Zod-validated tools) and forwards every call over localhost HTTP. Hybrid retrieval = BGE-M3 dense (LanceDB ANN) + BM25 sparse (SQLite FTS5), fused with RRF(k=60), optionally reranked by Claude Haiku-4.5. Every chunk carries immutable provenance (`source_id`, `source_path`, `content_hash`, `ingest_ts`).

**Tech Stack:** Python 3.12 (uv-managed, ruff 0.8, pytest 8.3, pytest-asyncio 0.24); FastAPI 0.115 + uvicorn 0.32 + pydantic 2.10; sentence-transformers 3.3.1 + FlagEmbedding 1.3.4 + torch 2.5.1+cpu; LanceDB 0.17.0 + pysqlite3-binary 0.5.3; pymupdf4llm 0.0.17, tree-sitter 0.23.2, pygit2 1.16, markdown-it-py 3.0; anthropic 0.50 (Haiku-4.5); typer 0.13 + rich 13.9 + structlog 24.4. Node 22 LTS + pnpm: @modelcontextprotocol/sdk 1.27.1, zod 3.23.8, undici 6.21, vitest 2.1.8, biome 1.9.4.

---

## Plan Index

Execute strictly in order. Each phase commits to `main` at its exit gate.

| # | Plan | PRD Ref | Budget | Delivers |
|---|------|---------|--------|----------|
| 1 | [01-phase1-bootstrap.md](./01-phase1-bootstrap.md) | §16.3 | 2h | Repo, CI, SQLite schema, LanceDB wrapper — M1, M2 |
| 2 | [02-phase2-ingestion.md](./02-phase2-ingestion.md) | §16.4 | 5h | Adapter protocol, PDF/Claude/git adapters, CLI `ingest` — M3, M5, M6 |
| 3 | [03-phase3-retrieval.md](./03-phase3-retrieval.md) | §16.5 | 4h | Hybrid pipeline, reranker, CLI `query`, eval v0 — M7, M8, M4 (partial) |
| 4 | [04-phase4-mcp-cli.md](./04-phase4-mcp-cli.md) | §16.6 | 4h | FastAPI backend, TS MCP server (7 tools), CLI suite, corpora — M9, M10, S5 |
| 5 | [05-phase5-polish.md](./05-phase5-polish.md) | §16.7 | 3h | README, demo video, 30-query eval ≥0.80 Recall@5, privacy CI — M4 (final), S6 |

**Scope gate (non-negotiable, PRD §16.2):** M1–M10 Must-haves + S5 named corpora + S6 demo. All other S/C items are v0.2+.

---

## Repo Layout (target end of Phase 4)

```
personal-RAG-for-everything/
├── pyproject.toml              # uv-managed, name="contextd"
├── uv.lock
├── README.md
├── LICENSE                     # MIT
├── .gitignore                  # Python, Node, data/, .contextd-dev/, *.lance/
├── .github/workflows/ci.yml    # lint, typecheck, test, install-smoke
├── contextd/                   # Python package
│   ├── __init__.py
│   ├── config.py               # pydantic settings; CONTEXTD_HOME env override
│   ├── storage/
│   │   ├── schema.py           # DDL strings; migrations
│   │   ├── db.py               # SQLite connection (WAL, FKs on, JSON1)
│   │   ├── vectors.py          # LanceDB wrapper, embedding upsert/ann
│   │   └── models.py           # frozen dataclasses: Corpus, Source, Chunk, Edge
│   ├── ingest/
│   │   ├── protocol.py         # Adapter Protocol, SourceCandidate, Edge DTO
│   │   ├── registry.py         # source_type → Adapter class
│   │   ├── pipeline.py         # orchestrate parse→embed→write→edges
│   │   ├── embedder.py         # BGE-M3 wrapper (lazy load, batch encode)
│   │   └── adapters/
│   │       ├── pdf.py
│   │       ├── claude_export.py
│   │       ├── git_repo.py
│   │       └── markdown.py     # Phase 5 stub per PRD §14.5
│   ├── retrieve/
│   │   ├── preprocess.py       # QueryRequest build, NFC normalize, trace_id
│   │   ├── rewrite.py          # Haiku query expansion (disabled by default in v0.1 per D-30)
│   │   ├── dense.py            # LanceDB ANN, k=50
│   │   ├── sparse.py           # FTS5 BM25, k=50
│   │   ├── fusion.py           # RRF(k=60, top_n=50)
│   │   ├── rerank.py           # Haiku rerank, 5s timeout, graceful fallback
│   │   ├── format.py           # ChunkResult hydration, trace population
│   │   └── pipeline.py         # glue; async-gather across queries
│   ├── mcp/
│   │   └── api.py              # FastAPI routes mirroring the 7 MCP tools
│   ├── eval/
│   │   ├── harness.py          # score Recall@k, MRR; LLM-as-judge
│   │   └── seed_queries.json   # 30 queries with expected chunk IDs
│   ├── cli/
│   │   ├── main.py             # typer app; command registry
│   │   └── commands/{ingest,query,list,forget,status,config,serve,version}.py
│   └── logging_.py             # structlog config; INFO default, no content at INFO
├── mcp-server/                 # TypeScript subpackage
│   ├── package.json            # "packageManager": "pnpm@9.x"
│   ├── tsconfig.json
│   ├── biome.json
│   ├── src/
│   │   ├── index.ts            # MCP server entry (stdio)
│   │   ├── http-client.ts      # undici POST to localhost:8787
│   │   └── tools/              # one file per tool: search_corpus, fetch_chunk, ...
│   └── tests/*.test.ts
├── tests/                      # Python tests
│   ├── unit/
│   ├── integration/
│   ├── privacy/                # no-network, non-mutation, no-content-at-INFO
│   └── fixtures/               # tiny PDFs, sanitized Claude export, mini git repo
└── docs/
    ├── PRDs/                   # existing
    └── plans/                  # this directory
```

---

## Cross-phase invariants (every plan enforces)

1. **Immutability.** All DTOs are `@dataclass(frozen=True)` or `NamedTuple`. Update = new object. No `obj.field = x`.
2. **Provenance.** `source_id`, `source_path`, `content_hash` (SHA-256), `ingest_ts` must round-trip through every pipeline. Drop one → test fails.
3. **Non-mutation.** Ingestion never writes to source paths. Tested via pre/post SHA-256 hash diff on the source directory.
4. **No outbound network by default.** The Anthropic client is the only permitted caller; gated on `ANTHROPIC_API_KEY` presence AND `reranker.provider == "anthropic"` in config. Privacy CI asserts no other sockets open (see Phase 5).
5. **Corpus isolation.** Every storage call takes a `corpus: str`; no cross-corpus default joins. Clinical data would live in a separate corpus; v0.1 enforces the default `personal` corpus.
6. **Graceful degradation.** Reranker/rewriter LLM failure → log + continue with RRF order + set `trace.reranker_used = null`. Never raises to caller.
7. **Trace IDs.** ULID per retrieval, propagated through every stage, persisted to `AUDIT_LOG`.
8. **Dev data path.** All code reads `CONTEXTD_HOME` env; default to `~/.contextd` in prod, `./.contextd-dev/` in dev (repo-local, gitignored).

---

## Shared data types (defined in Phase 1, imported by all later phases)

`contextd/storage/models.py` — these signatures are load-bearing across plans; later phases MUST match names exactly.

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

SourceType = Literal["pdf", "claude_export", "git_repo", "markdown", "notion", "gmail", "arxiv_bookmark", "web_page"]
SourceStatus = Literal["active", "deleted", "failed"]
EdgeType = Literal["wikilink", "conversation_next", "conversation_prev", "code_imports", "pdf_cites", "email_reply_to", "email_thread"]
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
    trace_id: str                 # ULID
    latency_ms: int
    dense_candidates: int
    sparse_candidates: int
    reranker_used: str | None     # null when skipped / failed
    rewriter_used: str | None
```

Python↔TS JSON envelope for every MCP tool (Phase 4 contract):

```json
{ "ok": true, "data": { ... } }
{ "ok": false, "error": { "code": "NOT_FOUND", "message": "...", "trace_id": "01JH..." } }
```

Error codes fixed in Phase 4: `BAD_REQUEST`, `NOT_FOUND`, `CORPUS_NOT_FOUND`, `RERANK_UNAVAILABLE`, `INTERNAL`.

---

## SQLite DDL (authoritative; Phase 1 writes this verbatim)

```sql
-- contextd/storage/schema.py renders these inside a transaction on first open.
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS corpus (
    name            TEXT PRIMARY KEY,
    root_path       TEXT,
    embed_model     TEXT NOT NULL,
    embed_dim       INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS source (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus          TEXT NOT NULL REFERENCES corpus(name) ON DELETE CASCADE,
    source_type     TEXT NOT NULL,
    path            TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    title           TEXT,
    ingested_at     TEXT NOT NULL,
    source_mtime    TEXT,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'active',
    UNIQUE (corpus, path)
);
CREATE INDEX IF NOT EXISTS idx_source_type ON source(source_type);
CREATE INDEX IF NOT EXISTS idx_source_hash ON source(content_hash);

CREATE TABLE IF NOT EXISTS chunk (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id       INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    offset_start    INTEGER,
    offset_end      INTEGER,
    token_count     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    section_label   TEXT,
    scope           TEXT,
    role            TEXT,
    chunk_timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_source_ordinal ON chunk(source_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunk_timestamp ON chunk(chunk_timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    content,
    content='chunk',
    content_rowid='id',
    tokenize='unicode61'
);
-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
    INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunk_au AFTER UPDATE ON chunk BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE IF NOT EXISTS edge (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_chunk_id   INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
    target_chunk_id   INTEGER REFERENCES chunk(id) ON DELETE CASCADE,
    target_hint       TEXT,
    edge_type         TEXT NOT NULL,
    label             TEXT,
    weight            REAL
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edge(source_chunk_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edge_tgt ON edge(target_chunk_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_edge_hint ON edge(target_hint);

CREATE TABLE IF NOT EXISTS chunk_meta (
    chunk_id    INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (chunk_id, key)
);
CREATE TABLE IF NOT EXISTS source_meta (
    source_id   INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    PRIMARY KEY (source_id, key)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at   TEXT NOT NULL,
    actor         TEXT NOT NULL,
    action        TEXT NOT NULL,
    target        TEXT NOT NULL,
    details_json  TEXT NOT NULL DEFAULT '{}'
);
```

LanceDB table `embedding`: columns `chunk_id INT64 PRIMARY`, `vector FLOAT32[1024]`, `model_name STRING`. One LanceDB dir per corpus (`<data_root>/corpora/<name>/vectors.lance`).

---

## MCP Tool Contract (authoritative; Phase 4 implements)

Seven tools. Names, inputs, outputs are PRD-stable and therefore a v0.1 breaking-change boundary. **Do not rename.**

| Tool | Input (required / optional) | Output |
|------|----|----|
| `search_corpus` | `query: str` / `corpus: str = "personal"`, `limit: int = 10` (max 100), `source_types: SourceType[]`, `date_range: {from, to}`, `source_path_prefix: str`, `metadata_filters: {key: value}`, `rewrite: bool = false`, `rerank: bool = true` | `{ results: ChunkResult[], query: {original, rewritten[], corpus, filters_applied}, trace: QueryTrace }` |
| `fetch_chunk` | `chunk_id: int` / `include_edges: bool = true`, `include_metadata: bool = true` | `{ chunk: ChunkResult }` |
| `expand_context` | `chunk_id: int` / `before: int = 2`, `after: int = 2` | `{ chunks: ChunkResult[] }` (source-ordered) |
| `get_edges` | `chunk_id: int` / `direction: "inbound" \| "outbound" \| "both" = "both"`, `edge_types: EdgeType[]`, `include_target_chunks: bool = false`, `limit: int = 50` | `{ edges: Edge[], targets?: ChunkResult[] }` |
| `list_sources` | — / `corpus: str`, `source_types: SourceType[]`, `ingested_since: datetime`, `limit: int = 50`, `offset: int = 0` | `{ sources: Source[], total: int, has_more: bool }` |
| `get_source` | `source_id: int` XOR `path: str + corpus: str` | `{ source: Source, metadata: {key: value} }` |
| `list_corpora` | — | `{ corpora: Corpus[] }` (with `source_count`, `chunk_count`) |

HTTP surface mirrors these at `/v1/*` (Phase 4 `contextd/mcp/api.py`).

---

## Environment prereqs (AlexZ to handle before Phase 4 starts)

- [x] `uv` already at `~/.local/bin/uv`
- [ ] **Install pnpm** before Phase 4: `brew install pnpm` (Node 22 LTS + pnpm pinned via `packageManager` field in `mcp-server/package.json`)
- [ ] `ANTHROPIC_API_KEY` exported in shell for Phase 3 reranker integration tests (graceful-degrade path must also be tested without it — CI runs that matrix)

---

## Quality gates (applied at each phase exit)

- `uv run ruff format --check .` — clean
- `uv run ruff check .` — clean (ignore `E501` line length for DDL strings only)
- `uv run mypy contextd/` — clean (strict-optional, disallow-untyped-defs)
- `uv run pytest -q` — green, coverage per-phase targets below

| Subsystem | Coverage target (v0.1) |
|-----------|---|
| `contextd/storage/` | ≥ 70% |
| `contextd/ingest/` | ≥ 60% |
| `contextd/retrieve/` | ≥ 60% |
| Overall | ≥ 50% |

Phase 5 adds: privacy CI (`tests/privacy/test_no_network.py`, `test_non_mutation.py`), 30-query eval ≥ 0.80 Recall@5.

---

## Execution handoff (after reviewing master + phase plans)

Recommended path: **subagent-driven-development, one phase at a time**, with human review at each exit gate. Alternative: inline execution per phase.

Start with `01-phase1-bootstrap.md`. Do not skip ahead — later phases hard-depend on the storage schema and the `models.py` dataclass shapes defined in Phase 1.
