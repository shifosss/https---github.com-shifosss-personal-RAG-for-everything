# Phase 3 — Retrieval Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal:** Wire the 6-stage retrieval pipeline (preprocess → rewrite → hybrid dense+sparse → RRF fuse → rerank → format) behind a single async entry point `retrieve()`. Ship `contextd query` CLI and a 10-query baseline eval harness.

**Architecture:** Six small modules in `contextd/retrieve/` each own one stage. `pipeline.retrieve(query, opts)` dispatches dense and sparse in parallel via `asyncio.gather`, merges with RRF(k=60), optionally reranks via Anthropic Haiku-4.5, and returns a tuple `(list[ChunkResult], QueryTrace)`. Graceful degradation is a hard requirement: rewriter/reranker failure never raises — it logs, sets `trace.rewriter_used=None` or `trace.reranker_used=None`, and falls through to the next stage.

**Tech Stack:** anthropic 0.50.0 (Haiku-4.5 JSON mode, 3s rewriter timeout, 5s reranker timeout), reused BGE-M3 embedder from Phase 2, SQLite FTS5 (stdlib), LanceDB 0.17 (phase 1 wrapper), python-ulid 3.0 (trace IDs).

**Prereqs:** Phase 1 (storage) and Phase 2 (ingestion) complete. `ANTHROPIC_API_KEY` exported for integration tests with rerank on; CI also runs without the key to verify graceful-degrade.

**Exit gate (PRD §16.5):**
- `contextd query "X" --limit 5` returns results in < 2 s (no rerank) and < 4 s (with rerank)
- 10-query baseline eval: Recall@5 ≥ 0.60 (full 30-query ≥ 0.80 gate is Phase 5)
- JSON output (`--json`) validates against `ChunkResult` schema from master spec
- Reranker disabled / API unreachable → returns RRF order + `trace.reranker_used=null` (no exception)

---

## File Structure

Create:
- `contextd/retrieve/__init__.py`
- `contextd/retrieve/preprocess.py` — `QueryRequest`, NFC normalize, trace_id
- `contextd/retrieve/rewrite.py` — Haiku sub-query expansion
- `contextd/retrieve/dense.py` — LanceDB ANN per query
- `contextd/retrieve/sparse.py` — FTS5 BM25 per query
- `contextd/retrieve/fusion.py` — RRF implementation
- `contextd/retrieve/rerank.py` — Haiku reranker
- `contextd/retrieve/format.py` — `ChunkResult` hydration (joins chunk + source + meta + edges)
- `contextd/retrieve/pipeline.py` — orchestration (`retrieve()` entry)
- `contextd/retrieve/filters.py` — filter model + SQL WHERE + Lance filter string builders
- `contextd/eval/harness.py`, `contextd/eval/seed_queries.json`
- `contextd/cli/commands/query.py`
- `tests/unit/retrieve/test_*.py` (one per module)
- `tests/integration/retrieve/test_pipeline_end_to_end.py`
- `tests/integration/retrieve/test_graceful_degradation.py`
- `tests/integration/cli/test_query.py`

---

## Task 1: QueryRequest + preprocessing

**Files:**
- Create: `contextd/retrieve/preprocess.py`
- Test: `tests/unit/retrieve/test_preprocess.py`

- [ ] **Step 1: Test**

```python
# tests/unit/retrieve/test_preprocess.py
import pytest
from contextd.retrieve.preprocess import QueryRequest, build_request, QueryFilter

def test_nfc_normalizes():
    req = build_request(query="café", corpus="personal")  # NFC of "cafe\u0301"
    # The NFC-normalized form should be a single 4-char string
    assert len(req.query) == 4

def test_length_cap_truncates_at_2000():
    req = build_request(query="x" * 5000, corpus="personal")
    assert len(req.query) == 2000

def test_trace_id_is_ulid_like():
    req = build_request(query="x", corpus="personal")
    assert len(req.trace_id) == 26  # ULID canonical length

def test_filter_defaults_empty():
    req = build_request(query="x", corpus="personal")
    assert req.filters.source_types == ()
    assert req.filters.metadata == {}

def test_invalid_corpus_raises():
    with pytest.raises(ValueError):
        build_request(query="x", corpus="")
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/preprocess.py
from __future__ import annotations
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from ulid import ULID
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
        query=q, corpus=corpus, limit=lim, trace_id=str(ULID()),
        filters=filters or QueryFilter(),
        rewrite=rewrite, rerank=rerank,
    )
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/preprocess.py tests/unit/retrieve/
git commit -m "feat(retrieve): QueryRequest with NFC normalize + ULID trace"
```

---

## Task 2: Dense retrieval

**Files:**
- Create: `contextd/retrieve/dense.py`
- Test: `tests/integration/retrieve/test_dense.py`

- [ ] **Step 1: Test**

```python
# tests/integration/retrieve/test_dense.py
from datetime import datetime, timezone
import numpy as np
import pytest
from contextd.retrieve.dense import dense_search
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(home, embed_dim=4):
    conn = open_db("personal")
    insert_corpus(conn, name="personal", embed_model="t", embed_dim=embed_dim,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus="personal", source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=0, status="active")
    ids = [insert_chunk(conn, source_id=sid, ordinal=i, token_count=1, content=f"c{i}") for i in range(3)]
    vs = VectorStore.open(corpus="personal", embed_dim=embed_dim, model_name="t")
    vs.upsert(ids, np.array([[1,0,0,0],[0,1,0,0],[0.7,0.7,0,0]], dtype=np.float32))
    conn.commit()
    return ids


async def test_dense_returns_topk_by_cosine(tmp_contextd_home):
    ids = _seed(tmp_contextd_home)
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    hits = await dense_search(query="anything", corpus="personal", k=2, embedder=StubEmb())
    assert [h[0] for h in hits] == [ids[0], ids[2]]  # closest to [1,0,0,0]
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/dense.py
from __future__ import annotations
import asyncio
from contextd.ingest.embedder import Embedder, default_embedder
from contextd.storage.vectors import VectorStore


async def dense_search(
    *, query: str, corpus: str, k: int, embedder: Embedder | None = None,
) -> list[tuple[int, float]]:
    emb = embedder or default_embedder()
    vec = await asyncio.to_thread(lambda: emb.embed([query])[0])
    vs = VectorStore.open(corpus=corpus, embed_dim=emb.dim, model_name=emb.model_name)
    return await asyncio.to_thread(vs.ann_search, vec, k)
```

- [ ] **Step 4: Run test — expect pass. Commit.**

```bash
git add contextd/retrieve/dense.py tests/integration/retrieve/test_dense.py
git commit -m "feat(retrieve): async dense_search via LanceDB ANN"
```

---

## Task 3: Sparse retrieval (FTS5 BM25)

**Files:**
- Create: `contextd/retrieve/sparse.py`
- Test: `tests/integration/retrieve/test_sparse.py`

- [ ] **Step 1: Test**

```python
# tests/integration/retrieve/test_sparse.py
from datetime import datetime, timezone
import pytest
from contextd.retrieve.sparse import sparse_search
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db

pytestmark = pytest.mark.integration


async def test_bm25_returns_matching_chunk(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(conn, name="personal", embed_model="t", embed_dim=4,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus="personal", source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=0, status="active")
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=5, content="negation handling in clinical NLP")
    c2 = insert_chunk(conn, source_id=sid, ordinal=1, token_count=5, content="transformer architecture overview")
    conn.commit()

    hits = await sparse_search(query="negation", corpus="personal", k=5)
    ids = [h[0] for h in hits]
    assert c1 in ids and c2 not in ids


async def test_empty_query_returns_empty(tmp_contextd_home):
    open_db("personal")
    hits = await sparse_search(query="", corpus="personal", k=5)
    assert hits == []
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/sparse.py
from __future__ import annotations
import asyncio
import re
from contextd.storage.db import open_db


def _sanitize_fts(q: str) -> str:
    # Strip punctuation that confuses FTS5; quote tokens to keep them literal
    tokens = re.findall(r"\w+", q)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else ""


async def sparse_search(*, query: str, corpus: str, k: int) -> list[tuple[int, float]]:
    fts_q = _sanitize_fts(query)
    if not fts_q:
        return []

    def _run() -> list[tuple[int, float]]:
        conn = open_db(corpus)
        rows = conn.execute(
            "SELECT rowid, bm25(chunk_fts) AS score FROM chunk_fts WHERE chunk_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (fts_q, k),
        ).fetchall()
        # FTS5 bm25() returns lower = better; flip sign so higher = better to match dense convention.
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    return await asyncio.to_thread(_run)
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/sparse.py tests/integration/retrieve/test_sparse.py
git commit -m "feat(retrieve): async sparse_search via FTS5 bm25"
```

---

## Task 4: RRF fusion

**Files:**
- Create: `contextd/retrieve/fusion.py`
- Test: `tests/unit/retrieve/test_fusion.py`

- [ ] **Step 1: Test**

```python
# tests/unit/retrieve/test_fusion.py
from contextd.retrieve.fusion import reciprocal_rank_fusion

def test_rrf_merges_two_lists():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(3, 5.0), (2, 4.0), (4, 1.0)]
    out = reciprocal_rank_fusion([(dense, sparse)], k=60, top_n=4)
    ids = [c for c, _ in out]
    # Chunks appearing in both lists at high ranks should rank higher
    assert ids[:2] == [2, 3] or ids[:2] == [3, 2]
    assert 4 in ids
    # Scores strictly decreasing
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)

def test_rrf_single_list_preserves_order():
    dense = [(10, 0.9), (20, 0.8), (30, 0.1)]
    sparse: list = []
    out = reciprocal_rank_fusion([(dense, sparse)], k=60, top_n=3)
    assert [c for c, _ in out] == [10, 20, 30]

def test_rrf_multiple_queries_accumulate():
    q1 = ([(1, 0.0)], [])
    q2 = ([(1, 0.0), (2, 0.0)], [])
    out = reciprocal_rank_fusion([q1, q2], k=60, top_n=2)
    ids = [c for c, _ in out]
    assert ids[0] == 1  # chunk 1 appears in both -> higher cumulative score
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/fusion.py
from __future__ import annotations
from collections import defaultdict


def reciprocal_rank_fusion(
    per_query: list[tuple[list[tuple[int, float]], list[tuple[int, float]]]],
    *,
    k: int = 60,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """Per PRD §15.5: score = Σ 1 / (k + rank). Rank is 1-based."""
    acc: dict[int, float] = defaultdict(float)
    for dense, sparse in per_query:
        for rank_idx, (cid, _) in enumerate(dense):
            acc[cid] += 1.0 / (k + rank_idx + 1)
        for rank_idx, (cid, _) in enumerate(sparse):
            acc[cid] += 1.0 / (k + rank_idx + 1)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_n]
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/fusion.py tests/unit/retrieve/test_fusion.py
git commit -m "feat(retrieve): reciprocal rank fusion (k=60, top_n=50)"
```

---

## Task 5: Reranker with graceful degradation

**Files:**
- Create: `contextd/retrieve/rerank.py`
- Test: `tests/integration/retrieve/test_rerank.py`

- [ ] **Step 1: Test — include the graceful-degrade case**

```python
# tests/integration/retrieve/test_rerank.py
import asyncio
import pytest
from contextd.retrieve.rerank import rerank, RerankUnavailable

pytestmark = pytest.mark.integration


async def test_rerank_returns_ordered_ids_from_anthropic(monkeypatch):
    calls: list[dict] = []

    class FakeMessages:
        def create(self, *, model, max_tokens, temperature, system, messages, response_format=None):
            calls.append({"model": model, "messages": messages})
            # Respond with scores reversing input order
            class R:
                content = [type("b", (), {"text": '[{"id": 2, "score": 10}, {"id": 1, "score": 5}]'})()]
            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())

    chunks = [
        (1, "chunk one"),
        (2, "chunk two"),
    ]
    out = await rerank(query="q", candidates=chunks, model="claude-haiku-4-5", timeout_ms=5000)
    assert [c for c, _ in out] == [2, 1]


async def test_rerank_unavailable_raises_typed_error(monkeypatch):
    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                raise ConnectionError("api down")
    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())
    with pytest.raises(RerankUnavailable):
        await rerank(query="q", candidates=[(1, "t")], model="m", timeout_ms=5000)


async def test_rerank_timeout_raises(monkeypatch):
    class SlowMessages:
        def create(self, **kw):
            import time; time.sleep(2.0)
            class R: content = [type("b", (), {"text": "[]"})()]
            return R()
    class FakeClient: messages = SlowMessages()
    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())
    with pytest.raises(RerankUnavailable):
        await rerank(query="q", candidates=[(1, "t")], model="m", timeout_ms=500)
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/rerank.py
from __future__ import annotations
import asyncio
import json
import os
from functools import lru_cache
from anthropic import Anthropic

_SYS_PROMPT = (
    "You are a reranker for a personal retrieval system. Score each candidate "
    "chunk 0-10 for relevance to the user's query. Reply ONLY with a JSON array "
    'like [{"id": 12345, "score": 8}, ...]. No prose.'
)


class RerankUnavailable(Exception):
    """Raised when the reranker API is unreachable, times out, or returns invalid output after retry."""


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def rerank(
    *,
    query: str,
    candidates: list[tuple[int, str]],
    model: str,
    timeout_ms: int,
    truncate_tokens: int = 800,
) -> list[tuple[int, float]]:
    if not candidates:
        return []

    payload = {
        "query": query,
        "candidates": [
            {"id": cid, "content": _truncate(text, truncate_tokens)}
            for cid, text in candidates
        ],
    }
    user_msg = f"Query: {payload['query']}\n\nCandidates:\n" + json.dumps(payload["candidates"], ensure_ascii=False)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _anthropic_client().messages.create,
                model=model, max_tokens=1200, temperature=0.0,
                system=_SYS_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ),
            timeout=timeout_ms / 1000.0,
        )
    except (asyncio.TimeoutError, ConnectionError, OSError) as e:
        raise RerankUnavailable(f"rerank API unreachable: {e!r}") from e
    except Exception as e:
        raise RerankUnavailable(f"rerank failed: {e!r}") from e

    try:
        text = result.content[0].text.strip()
        data = json.loads(text)
        scored = [(int(x["id"]), float(x["score"])) for x in data]
    except Exception as e:
        raise RerankUnavailable(f"invalid rerank JSON: {e!r}") from e

    return sorted(scored, key=lambda x: x[1], reverse=True)


def _truncate(s: str, tokens: int) -> str:
    # Cheap char-based truncation (4 chars ~ 1 token average).
    cap = tokens * 4
    return s if len(s) <= cap else s[:cap]
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/rerank.py tests/integration/retrieve/test_rerank.py
git commit -m "feat(retrieve): Haiku reranker with timeout-based graceful-degrade error"
```

---

## Task 6: Result formatter (hydration)

**Files:**
- Create: `contextd/retrieve/format.py`
- Test: `tests/integration/retrieve/test_format.py`

- [ ] **Step 1: Test**

```python
# tests/integration/retrieve/test_format.py
from datetime import datetime, timezone
import pytest
from contextd.retrieve.format import hydrate_results
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db

pytestmark = pytest.mark.integration


def test_hydrate_returns_chunkresult_with_source_and_meta(tmp_contextd_home):
    conn = open_db("personal")
    insert_corpus(conn, name="personal", embed_model="t", embed_dim=4,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus="personal", source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=1, status="active", title="A")
    cid = insert_chunk(conn, source_id=sid, ordinal=0, token_count=2, content="hi",
                       section_label="methods")
    conn.execute("INSERT INTO chunk_meta(chunk_id, key, value) VALUES (?, 'pdf_page', '4')", (cid,))
    conn.commit()
    results = hydrate_results(corpus="personal", scored=[(cid, 0.9)])
    assert len(results) == 1
    r = results[0]
    assert r.chunk.id == cid
    assert r.source.title == "A"
    assert r.metadata.get("pdf_page") == "4"
    assert r.rank == 1
    assert r.score == 0.9
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/format.py
from __future__ import annotations
from contextd.storage.db import fetch_chunks_by_ids, open_db, row_to_source
from contextd.storage.models import ChunkResult, Edge


def hydrate_results(*, corpus: str, scored: list[tuple[int, float]]) -> list[ChunkResult]:
    if not scored:
        return []
    conn = open_db(corpus)
    ids = [cid for cid, _ in scored]
    chunks = {c.id: c for c in fetch_chunks_by_ids(conn, ids)}
    if not chunks:
        return []
    # Sources
    q_placeholders = ",".join("?" for _ in chunks)
    src_rows = conn.execute(
        f"SELECT DISTINCT s.* FROM source s JOIN chunk c ON c.source_id = s.id WHERE c.id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    sources_by_id = {r["id"]: row_to_source(r) for r in src_rows}
    # Metadata
    meta_rows = conn.execute(
        f"SELECT chunk_id, key, value FROM chunk_meta WHERE chunk_id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    meta_by_chunk: dict[int, dict[str, str]] = {}
    for r in meta_rows:
        meta_by_chunk.setdefault(int(r["chunk_id"]), {})[r["key"]] = r["value"]
    # Edges (outbound only for format; expand_context handles inbound)
    edge_rows = conn.execute(
        f"SELECT * FROM edge WHERE source_chunk_id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    edges_by_chunk: dict[int, list[Edge]] = {}
    for r in edge_rows:
        edges_by_chunk.setdefault(int(r["source_chunk_id"]), []).append(
            Edge(id=r["id"], source_chunk_id=r["source_chunk_id"],
                 target_chunk_id=r["target_chunk_id"], target_hint=r["target_hint"],
                 edge_type=r["edge_type"], label=r["label"], weight=r["weight"])
        )

    out: list[ChunkResult] = []
    for rank_idx, (cid, score) in enumerate(scored, start=1):
        c = chunks.get(cid)
        if c is None: continue
        out.append(ChunkResult(
            chunk=c, source=sources_by_id[c.source_id], score=score, rank=rank_idx,
            metadata=meta_by_chunk.get(cid, {}),
            edges=tuple(edges_by_chunk.get(cid, ())),
        ))
    return out
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/format.py tests/integration/retrieve/test_format.py
git commit -m "feat(retrieve): ChunkResult hydration joining chunk+source+meta+edges"
```

---

## Task 7: Query rewriter (stub + test, disabled by default)

**Files:**
- Create: `contextd/retrieve/rewrite.py`
- Test: `tests/integration/retrieve/test_rewrite.py`

Per PRD D-30, rewrite is off by default in v0.1 but the code path exists and is tested.

- [ ] **Step 1: Test**

```python
# tests/integration/retrieve/test_rewrite.py
import pytest
from contextd.retrieve.rewrite import rewrite_query

pytestmark = pytest.mark.integration


async def test_rewrite_deduplicates_and_caps(monkeypatch):
    class FakeMessages:
        def create(self, **kw):
            class R: content = [type("b", (), {"text": '{"sub_queries": ["a", "a", "b", "c", "d", "e", "f"]}'})()]
            return R()
    class FakeClient: messages = FakeMessages()
    monkeypatch.setattr("contextd.retrieve.rewrite._anthropic_client", lambda: FakeClient())
    out = await rewrite_query(query="orig", model="m", timeout_ms=3000)
    assert out.original == "orig"
    assert out.sub_queries[:1] == ["a"]
    assert len(set(["orig", *out.sub_queries])) <= 6


async def test_rewrite_failure_returns_empty_subqueries(monkeypatch):
    class FakeMessages:
        def create(self, **kw): raise ConnectionError("down")
    class FakeClient: messages = FakeMessages()
    monkeypatch.setattr("contextd.retrieve.rewrite._anthropic_client", lambda: FakeClient())
    out = await rewrite_query(query="orig", model="m", timeout_ms=3000)
    assert out.sub_queries == []
    assert out.rewriter_used is None
```

- [ ] **Step 2: Run — expect fail. Step 3: Implement**

```python
# contextd/retrieve/rewrite.py
from __future__ import annotations
import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from anthropic import Anthropic

_SYS = (
    "You are a query-expansion assistant. Given one user query, produce 3-5 "
    "alternative phrasings covering the semantic territory they might want. "
    'Respond ONLY with JSON: {"sub_queries": ["...", "..."]}'
)


@dataclass(frozen=True)
class RewrittenQueries:
    original: str
    sub_queries: list[str]
    rewriter_used: str | None


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def rewrite_query(*, query: str, model: str, timeout_ms: int) -> RewrittenQueries:
    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(
                _anthropic_client().messages.create,
                model=model, max_tokens=400, temperature=0.4,
                system=_SYS,
                messages=[{"role": "user", "content": query}],
            ),
            timeout=timeout_ms / 1000.0,
        )
        data = json.loads(res.content[0].text.strip())
        subs = [s.strip() for s in data.get("sub_queries", []) if isinstance(s, str) and s.strip()]
        # Dedup preserving order; drop any equal to original
        seen: set[str] = {query}
        uniq: list[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s); uniq.append(s)
        # Hard cap 5 (PRD §15.3); original + 5 = 6 total
        return RewrittenQueries(original=query, sub_queries=uniq[:5], rewriter_used=model)
    except Exception:
        return RewrittenQueries(original=query, sub_queries=[], rewriter_used=None)
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/retrieve/rewrite.py tests/integration/retrieve/test_rewrite.py
git commit -m "feat(retrieve): Haiku query rewriter (disabled-by-default per D-30)"
```

---

## Task 8: Pipeline assembly

**Files:**
- Create: `contextd/retrieve/pipeline.py`
- Test: `tests/integration/retrieve/test_pipeline_end_to_end.py`
- Test: `tests/integration/retrieve/test_graceful_degradation.py`

- [ ] **Step 1: Tests (end-to-end + graceful degradation)**

```python
# tests/integration/retrieve/test_pipeline_end_to_end.py
from datetime import datetime, timezone
import numpy as np
import pytest
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed_corpus(corpus="personal"):
    conn = open_db(corpus)
    insert_corpus(conn, name=corpus, embed_model="t", embed_dim=4,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus=corpus, source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=0, status="active", title="A")
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=5, content="negation handling clinical")
    c2 = insert_chunk(conn, source_id=sid, ordinal=1, token_count=5, content="transformer architecture overview")
    conn.commit()
    vs = VectorStore.open(corpus=corpus, embed_dim=4, model_name="t")
    vs.upsert([c1, c2], np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32))
    return c1, c2


async def test_retrieve_returns_rrf_ordered_without_rerank(monkeypatch, tmp_contextd_home):
    c1, c2 = _seed_corpus()
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr("contextd.ingest.embedder.default_embedder", lambda: StubEmb())
    req = build_request(query="negation", corpus="personal", limit=2, rerank=False, rewrite=False)
    results, trace = await retrieve(req)
    assert results[0].chunk.id == c1   # top-ranked: matches both dense and sparse
    assert trace.reranker_used is None


async def test_retrieve_returns_at_most_limit(monkeypatch, tmp_contextd_home):
    c1, c2 = _seed_corpus()
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr("contextd.ingest.embedder.default_embedder", lambda: StubEmb())
    req = build_request(query="architecture", corpus="personal", limit=1, rerank=False)
    results, _ = await retrieve(req)
    assert len(results) == 1
```

```python
# tests/integration/retrieve/test_graceful_degradation.py
import pytest
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request
from contextd.retrieve.rerank import RerankUnavailable

pytestmark = pytest.mark.integration

async def test_rerank_failure_falls_through_to_rrf(monkeypatch, tmp_contextd_home):
    # Seed two chunks; force rerank to raise.
    from tests.integration.retrieve.test_pipeline_end_to_end import _seed_corpus  # type: ignore
    import numpy as np
    _seed_corpus()
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr("contextd.ingest.embedder.default_embedder", lambda: StubEmb())

    async def bad_rerank(**kw): raise RerankUnavailable("down")
    monkeypatch.setattr("contextd.retrieve.pipeline.rerank", bad_rerank)

    req = build_request(query="negation", corpus="personal", limit=2, rerank=True)
    results, trace = await retrieve(req)
    assert len(results) == 2
    assert trace.reranker_used is None  # noted as disabled on failure
```

- [ ] **Step 2: Run both — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/retrieve/pipeline.py
from __future__ import annotations
import asyncio
import time
from contextd.config import get_settings
from contextd.logging_ import get_logger
from contextd.retrieve.dense import dense_search
from contextd.retrieve.format import hydrate_results
from contextd.retrieve.fusion import reciprocal_rank_fusion
from contextd.retrieve.preprocess import QueryRequest
from contextd.retrieve.rerank import RerankUnavailable, rerank
from contextd.retrieve.rewrite import rewrite_query
from contextd.retrieve.sparse import sparse_search
from contextd.storage.models import ChunkResult, QueryTrace

log = get_logger(__name__)


async def retrieve(req: QueryRequest) -> tuple[list[ChunkResult], QueryTrace]:
    s = get_settings()
    t0 = time.perf_counter()

    # 2. rewrite (optional)
    queries: list[str] = [req.query]
    rewriter_used: str | None = None
    if req.rewrite:
        rw = await rewrite_query(query=req.query, model=s.rewriter_model, timeout_ms=s.retrieval_rewrite_timeout_ms)
        queries.extend(rw.sub_queries)
        rewriter_used = rw.rewriter_used

    # 3+4. dense + sparse in parallel per query
    tasks = []
    for q in queries:
        tasks.append(dense_search(query=q, corpus=req.corpus, k=s.retrieval_dense_top_k))
        tasks.append(sparse_search(query=q, corpus=req.corpus, k=s.retrieval_sparse_top_k))
    outs = await asyncio.gather(*tasks)

    per_query: list[tuple[list[tuple[int, float]], list[tuple[int, float]]]] = []
    dense_count = sparse_count = 0
    for i in range(0, len(outs), 2):
        dense, sparse = outs[i], outs[i + 1]
        dense_count += len(dense); sparse_count += len(sparse)
        per_query.append((dense, sparse))

    # 5. fuse
    fused = reciprocal_rank_fusion(per_query, k=s.retrieval_rrf_k, top_n=s.retrieval_rerank_top_k)

    # 6. rerank (optional)
    reranker_used: str | None = None
    if req.rerank and fused:
        # Hydrate raw text for candidates (only content, not full ChunkResult, to keep reranker cheap)
        from contextd.storage.db import fetch_chunks_by_ids, open_db
        chunk_map = {c.id: c for c in fetch_chunks_by_ids(open_db(req.corpus), [cid for cid, _ in fused])}
        ordered_candidates = [(cid, chunk_map[cid].content) for cid, _ in fused if cid in chunk_map]
        try:
            reranked = await rerank(
                query=req.query, candidates=ordered_candidates,
                model=s.reranker_model, timeout_ms=s.retrieval_rerank_timeout_ms,
            )
            # Ties broken by RRF order: reranked is primary; missing IDs get 0 and fall through to RRF.
            rrf_index = {cid: idx for idx, (cid, _) in enumerate(fused)}
            rerank_ids = {cid for cid, _ in reranked}
            remainder = [(cid, s_) for cid, s_ in fused if cid not in rerank_ids]
            fused = reranked + remainder
            reranker_used = s.reranker_model
        except RerankUnavailable as e:
            log.warning("rerank.unavailable", error=str(e), trace_id=req.trace_id)

    # 7. format
    top = fused[: req.limit]
    results = hydrate_results(corpus=req.corpus, scored=[(cid, float(score)) for cid, score in top])

    latency_ms = int((time.perf_counter() - t0) * 1000)
    trace = QueryTrace(
        trace_id=req.trace_id, latency_ms=latency_ms,
        dense_candidates=dense_count, sparse_candidates=sparse_count,
        reranker_used=reranker_used, rewriter_used=rewriter_used,
    )
    # Audit log (hashed query, not raw)
    import hashlib
    from contextd.storage.db import open_db
    qh = hashlib.sha256(req.query.encode()).hexdigest()[:16]
    conn = open_db(req.corpus)
    conn.execute(
        "INSERT INTO audit_log(occurred_at, actor, action, target, details_json) "
        "VALUES (datetime('now'), 'retrieve', 'query', ?, ?)",
        (f"query#{qh}", f'{{"trace_id":"{req.trace_id}","latency_ms":{latency_ms}}}'),
    )
    conn.commit()
    return results, trace
```

- [ ] **Step 4: Run tests — expect pass. Commit.**

```bash
git add contextd/retrieve/pipeline.py tests/integration/retrieve/
git commit -m "feat(retrieve): end-to-end pipeline with RRF + optional rerank + trace"
```

---

## Task 9: `contextd query` CLI

**Files:**
- Create: `contextd/cli/commands/query.py`
- Modify: `contextd/cli/main.py` — register new command
- Test: `tests/integration/cli/test_query.py`

- [ ] **Step 1: Test**

```python
# tests/integration/cli/test_query.py
from datetime import datetime, timezone
import json
import numpy as np
import pytest
from typer.testing import CliRunner
from contextd.cli.main import app
from contextd.storage.db import insert_chunk, insert_corpus, insert_source, open_db
from contextd.storage.vectors import VectorStore

pytestmark = pytest.mark.integration


def _seed(monkeypatch):
    conn = open_db("personal")
    insert_corpus(conn, name="personal", embed_model="t", embed_dim=4,
                  created_at=datetime.now(timezone.utc), schema_version=1)
    sid = insert_source(conn, corpus="personal", source_type="pdf", path="/a.pdf",
                         content_hash="sha256:x", ingested_at=datetime.now(timezone.utc),
                         chunk_count=0, status="active", title="A")
    c1 = insert_chunk(conn, source_id=sid, ordinal=0, token_count=5, content="negation handling clinical")
    conn.commit()
    vs = VectorStore.open(corpus="personal", embed_dim=4, model_name="t")
    vs.upsert([c1], np.array([[1,0,0,0]], dtype=np.float32))
    class StubEmb:
        model_name = "t"; dim = 4
        def embed(self, texts): return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)
    monkeypatch.setattr("contextd.ingest.embedder.default_embedder", lambda: StubEmb())


def test_query_json_output_shape(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    r = CliRunner().invoke(app, ["query", "negation", "--corpus", "personal", "--limit", "1", "--no-rerank", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert "results" in data and "trace" in data
    assert data["results"][0]["chunk"]["content"] == "negation handling clinical"


def test_query_rich_output_includes_source_path(tmp_contextd_home, monkeypatch):
    _seed(monkeypatch)
    r = CliRunner().invoke(app, ["query", "negation", "--corpus", "personal", "--limit", "1", "--no-rerank"])
    assert r.exit_code == 0
    assert "/a.pdf" in r.stdout
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/cli/commands/query.py
from __future__ import annotations
import asyncio
import json
from dataclasses import asdict
import typer
from rich.console import Console
from rich.table import Table
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request

console = Console()


def query(
    query: str = typer.Argument(...),
    corpus: str = typer.Option("personal", "--corpus"),
    limit: int = typer.Option(10, "--limit"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    rewrite: bool = typer.Option(False, "--rewrite/--no-rewrite"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    req = build_request(query=query, corpus=corpus, limit=limit, rerank=rerank, rewrite=rewrite)
    results, trace = asyncio.run(retrieve(req))

    if as_json:
        payload = {
            "query": {"original": req.query, "corpus": req.corpus},
            "results": [_result_to_dict(r) for r in results],
            "trace": asdict(trace),
        }
        console.print(json.dumps(payload, default=str))
        return

    table = Table(title=f"Top {len(results)} for: {query}")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("section")
    table.add_column("content", overflow="fold")
    for r in results:
        table.add_row(str(r.rank), f"{r.score:.3f}", r.source.path, r.chunk.section_label or "-", r.chunk.content[:200])
    console.print(table)
    console.print(f"[dim]trace={trace.trace_id} latency={trace.latency_ms}ms "
                  f"reranker={trace.reranker_used or 'off'}[/dim]")


def _result_to_dict(r) -> dict:
    return {
        "chunk": {
            "id": r.chunk.id, "content": r.chunk.content, "ordinal": r.chunk.ordinal,
            "section_label": r.chunk.section_label, "scope": r.chunk.scope, "role": r.chunk.role,
            "token_count": r.chunk.token_count,
        },
        "source": {"id": r.source.id, "path": r.source.path, "type": r.source.source_type, "title": r.source.title},
        "score": r.score, "rank": r.rank, "metadata": r.metadata,
        "edges": [{"type": e.edge_type, "target_chunk_id": e.target_chunk_id, "target_hint": e.target_hint} for e in r.edges],
    }
```

```python
# contextd/cli/main.py — append
from contextd.cli.commands import query as query_cmd
app.command(name="query", help="Retrieve chunks matching a query.")(query_cmd.query)
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/cli/commands/query.py contextd/cli/main.py tests/integration/cli/test_query.py
git commit -m "feat(cli): contextd query with rich + --json output"
```

---

## Task 10: Eval harness v0 (10 queries)

**Files:**
- Create: `contextd/eval/__init__.py`
- Create: `contextd/eval/seed_queries.json`
- Create: `contextd/eval/harness.py`
- Test: `tests/integration/eval/test_harness.py`

- [ ] **Step 1: Write `seed_queries.json` with 10 queries** — each entry has `{"query": str, "corpus": str, "expected_chunk_ids": [int], "tags": [str]}`. For now, use simple queries whose ground-truth IDs are populated by the test fixture setup. Full 30-query set lands in Phase 5.

```json
[
  {"query": "negation handling", "corpus": "personal", "expected_keywords": ["negation"], "tags": ["paraphrase"]},
  {"query": "transformer architecture", "corpus": "personal", "expected_keywords": ["transformer"], "tags": ["direct"]}
]
```

(Engineer fills remaining 8 from fixture content.)

- [ ] **Step 2: Implement harness**

```python
# contextd/eval/harness.py
from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request


@dataclass(frozen=True)
class EvalResult:
    recall_at_5: float
    mrr: float
    n_queries: int


async def run_eval(seed_path: Path, corpus: str, k: int = 5) -> EvalResult:
    queries = json.loads(seed_path.read_text())
    hits = mrr_sum = 0
    for q in queries:
        req = build_request(query=q["query"], corpus=corpus, limit=k, rerank=False, rewrite=False)
        results, _ = await retrieve(req)
        kw = [kw.lower() for kw in q.get("expected_keywords", [])]
        positions = [i for i, r in enumerate(results, start=1) if any(k in r.chunk.content.lower() for k in kw)]
        if positions:
            hits += 1
            mrr_sum += 1.0 / positions[0]
    n = len(queries) or 1
    return EvalResult(recall_at_5=hits / n, mrr=mrr_sum / n, n_queries=n)
```

- [ ] **Step 3: Test**

```python
# tests/integration/eval/test_harness.py
from pathlib import Path
import asyncio
import pytest
from contextd.eval.harness import run_eval

pytestmark = pytest.mark.integration

async def test_eval_runs_on_seed_file(tmp_contextd_home, monkeypatch):
    # Minimal seed corpus from query CLI test helper
    from tests.integration.cli.test_query import _seed; _seed(monkeypatch)
    seed = Path(__file__).resolve().parents[3] / "contextd" / "eval" / "seed_queries.json"
    result = await run_eval(seed, corpus="personal", k=5)
    assert result.n_queries >= 2
    assert 0.0 <= result.recall_at_5 <= 1.0
```

- [ ] **Step 4: Run — expect pass. Commit.**

```bash
git add contextd/eval/ tests/integration/eval/
git commit -m "feat(eval): v0 harness + 10-query seed (30-query gate in Phase 5)"
```

---

## Phase 3 Exit Gate Checklist

- [ ] `uv run ruff check .` clean, `uv run mypy contextd/` clean
- [ ] `uv run pytest -q` green (incl. graceful-degrade test)
- [ ] `contextd query "negation" --corpus personal --no-rerank` returns in < 2s
- [ ] `contextd query "negation" --corpus personal --rerank` returns in < 4s (with `ANTHROPIC_API_KEY` set)
- [ ] Unsetting `ANTHROPIC_API_KEY` and running with `--rerank` still returns RRF-ordered results + trace shows `reranker_used=null`
- [ ] `uv run python -m contextd.eval.harness contextd/eval/seed_queries.json personal` reports `recall_at_5 >= 0.60`
- [ ] `contextd query "X" --json` output validates against `ChunkResult` schema (trace, score, source all present)

Commit all work; move to Phase 4.
