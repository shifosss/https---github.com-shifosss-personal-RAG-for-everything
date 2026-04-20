# Phase 2 — Ingestion Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`.

**Goal:** Ship three Must-have adapters (PDF, Claude export, git repo) behind a common `Adapter` protocol, plus the orchestration pipeline and `contextd ingest` CLI, so Phase 3 has real chunks + embeddings to retrieve over.

**Architecture:** Each adapter is a pure producer — it takes a filesystem path, yields `SourceCandidate` objects, and given a candidate, yields `Chunk` + `Edge` objects. The pipeline (`contextd/ingest/pipeline.py`) owns hashing, idempotency (skip on unchanged `content_hash`), embedding (via BGE-M3), and all storage writes. The embedder loads lazily on first use (2 GB model; pre-download via `hf download BAAI/bge-m3` if bandwidth-limited). Per PRD §14, adapters must never mutate source files.

**Tech Stack:** pymupdf4llm 0.0.17 + pymupdf 1.25.1 (PDF), tree-sitter 0.23.2 + grammar wheels (git), pygit2 1.16 (git), markdown-it-py 3.0 (Phase 5 stub), sentence-transformers 3.3.1 + FlagEmbedding 1.3.4 (embedder), typer 0.13 + rich 13.9 (CLI).

**Prereqs:** Phase 1 complete, storage schema applied, `CONTEXTD_HOME` resolves. Test fixtures needed (add during Task 2): 3 tiny public-domain arXiv PDFs (≤ 2 MB each), one sanitized Claude export JSON (≤ 20 conversations), one small git repo (check in as a tarball at `tests/fixtures/git/tiny-repo.tar.gz` with 5–10 Python files + one README).

**Exit gate (PRD §16.4):**
- `contextd ingest tests/fixtures/pdfs/` ingests all fixture PDFs
- `contextd ingest tests/fixtures/claude/export.json` creates one source per conversation, turn-granular chunks
- `contextd ingest tests/fixtures/git/tiny-repo/` creates function-scoped chunks with `scope` populated
- Re-running the same command is a no-op (hash idempotency)
- PRD §14.2.11 / §14.3.11 / §14.4.10 quality bars pass
- `uv run pytest tests/integration/ingest/ -q` green

---

## File Structure

Create:
- `contextd/ingest/__init__.py`
- `contextd/ingest/protocol.py` — `Adapter` Protocol, `SourceCandidate`, `IngestResult`
- `contextd/ingest/registry.py` — name → class
- `contextd/ingest/embedder.py` — BGE-M3 wrapper
- `contextd/ingest/pipeline.py` — orchestration
- `contextd/ingest/adapters/__init__.py`
- `contextd/ingest/adapters/pdf.py`
- `contextd/ingest/adapters/claude_export.py`
- `contextd/ingest/adapters/git_repo.py`
- `contextd/cli/__init__.py`, `contextd/cli/main.py`
- `contextd/cli/commands/__init__.py`, `contextd/cli/commands/ingest.py`
- `tests/fixtures/pdfs/`, `tests/fixtures/claude/export.json`, `tests/fixtures/git/tiny-repo.tar.gz`
- `tests/unit/ingest/test_protocol.py`
- `tests/integration/ingest/test_pdf.py`, `test_claude.py`, `test_git.py`, `test_pipeline_idempotency.py`
- `tests/integration/cli/test_ingest.py`

Modify:
- `pyproject.toml` — add tree-sitter grammar dependencies (see Task 7)

---

## Task 1: `Adapter` protocol + `SourceCandidate` DTO

**Files:**
- Create: `contextd/ingest/protocol.py`
- Test: `tests/unit/ingest/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/ingest/test_protocol.py
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest
from contextd.ingest.protocol import Adapter, SourceCandidate, ChunkDraft, EdgeDraft

def test_source_candidate_is_frozen():
    sc = SourceCandidate(path=Path("/a.pdf"), source_type="pdf", canonical_id="/a.pdf", content_hash="sha256:x")
    with pytest.raises(FrozenInstanceError):
        sc.path = Path("/b.pdf")  # type: ignore[misc]

def test_chunk_draft_defaults():
    c = ChunkDraft(ordinal=0, content="hi", token_count=1)
    assert c.section_label is None

def test_adapter_is_runtime_checkable():
    class StubPDF:
        source_type: str = "pdf"
        def can_handle(self, path: Path) -> bool: return True
        def sources(self, path: Path): yield from ()
        def parse(self, source: SourceCandidate): yield from ()
        def metadata(self, source: SourceCandidate): return {}
        def edges(self, chunks): yield from ()
    assert isinstance(StubPDF(), Adapter)
```

- [ ] **Step 2: Run test — expect fail.**

- [ ] **Step 3: Implement `contextd/ingest/protocol.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable
from contextd.storage.models import EdgeType, Role, SourceType


@dataclass(frozen=True)
class SourceCandidate:
    path: Path                     # canonical path; for multi-source files, path#fragment
    source_type: SourceType
    canonical_id: str              # equal to str(path) for single-source files; "path#frag" otherwise
    content_hash: str              # "sha256:..." computed over canonical bytes
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
    target_ordinal: int | None = None   # resolved within the same source
    target_hint: str | None = None       # unresolved (e.g., wikilink text)
    label: str | None = None
    weight: float | None = None


@runtime_checkable
class Adapter(Protocol):
    source_type: str

    def can_handle(self, path: Path) -> bool: ...
    def sources(self, path: Path) -> Iterable[SourceCandidate]: ...
    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]: ...
    def metadata(self, source: SourceCandidate) -> dict[str, str]: ...
    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]: ...
```

- [ ] **Step 4: Run test — expect pass. Commit.**

```bash
git add contextd/ingest/protocol.py tests/unit/ingest/
git commit -m "feat(ingest): Adapter Protocol, SourceCandidate, ChunkDraft, EdgeDraft"
```

---

## Task 2: BGE-M3 embedder wrapper

**Files:**
- Create: `contextd/ingest/embedder.py`
- Test: `tests/integration/ingest/test_embedder.py` (slow; mark `@pytest.mark.slow`)

- [ ] **Step 1: Test (requires model download — mark slow)**

```python
# tests/integration/ingest/test_embedder.py
import numpy as np
import pytest
from contextd.ingest.embedder import Embedder

pytestmark = [pytest.mark.integration, pytest.mark.slow]

def test_embed_returns_1024_dim_unit_vectors():
    e = Embedder.load(model="BAAI/bge-m3", device="cpu")
    vecs = e.embed(["hello world", "negation handling in clinical NLP"])
    assert vecs.shape == (2, 1024)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

def test_embed_empty_batch_returns_empty():
    e = Embedder.load(model="BAAI/bge-m3", device="cpu")
    vecs = e.embed([])
    assert vecs.shape == (0, 1024)
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/ingest/embedder.py
from __future__ import annotations
from functools import lru_cache
from typing import Self
import numpy as np


class Embedder:
    def __init__(self, model_name: str, device: str, model) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model
        self._dim = 1024  # BGE-M3

    @property
    def model_name(self) -> str: return self._model_name
    @property
    def dim(self) -> int: return self._dim

    @classmethod
    def load(cls, *, model: str = "BAAI/bge-m3", device: str = "cpu") -> Self:
        from FlagEmbedding import BGEM3FlagModel
        m = BGEM3FlagModel(model, use_fp16=False, device=device)
        return cls(model, device, m)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = self._model.encode(texts, batch_size=16, max_length=8192)["dense_vecs"]
        arr = np.asarray(out, dtype=np.float32)
        # BGE-M3 outputs are already L2-normalized; reassert defensively.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


@lru_cache(maxsize=1)
def default_embedder() -> Embedder:
    from contextd.config import get_settings
    s = get_settings()
    return Embedder.load(model=s.embedding_model, device=s.embedding_device)
```

- [ ] **Step 4: Run — expect pass** (~30s first time; ~2s subsequent).

- [ ] **Step 5: Commit.**

```bash
git add contextd/ingest/embedder.py tests/integration/ingest/test_embedder.py
git commit -m "feat(ingest): BGE-M3 embedder with lazy load, 1024-dim output"
```

---

## Task 3: Ingestion pipeline (orchestration)

**Files:**
- Create: `contextd/ingest/pipeline.py`
- Create: `contextd/ingest/registry.py`
- Test: `tests/integration/ingest/test_pipeline_idempotency.py`

- [ ] **Step 1: Test with a mock adapter to isolate the pipeline**

```python
# tests/integration/ingest/test_pipeline_idempotency.py
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pytest
from contextd.ingest.pipeline import IngestionPipeline, IngestReport
from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate
from contextd.ingest.embedder import Embedder

pytestmark = pytest.mark.integration


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        self._model_name = "fake-4d"
        self._device = "cpu"
        self._model = None
        self._dim = 4

    def embed(self, texts):
        return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32) if texts else np.zeros((0, 4), np.float32)


class TwoChunkAdapter:
    source_type = "pdf"
    def can_handle(self, p): return True
    def sources(self, p):
        yield SourceCandidate(path=Path("/tmp/f.pdf"), source_type="pdf",
                              canonical_id="/tmp/f.pdf", content_hash="sha256:deadbeef",
                              title="F", source_mtime=datetime(2026, 4, 1, tzinfo=timezone.utc))
    def parse(self, source):
        yield ChunkDraft(ordinal=0, content="intro text", token_count=2, section_label="introduction")
        yield ChunkDraft(ordinal=1, content="method text", token_count=2, section_label="methods")
    def metadata(self, source): return {"pdf_authors_list": "Fu, Smith"}
    def edges(self, chunks): yield from ()


def test_ingest_writes_source_chunks_embeddings(tmp_contextd_home):
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunkAdapter()])
    report = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    assert report.sources_ingested == 1
    assert report.chunks_written == 2
    assert report.sources_skipped == 0

def test_ingest_is_idempotent_on_unchanged_hash(tmp_contextd_home):
    pipe = IngestionPipeline(embedder=FakeEmbedder(), adapters=[TwoChunkAdapter()])
    r1 = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    r2 = pipe.ingest(path=Path("/tmp/f.pdf"), corpus="personal")
    assert r1.sources_ingested == 1 and r1.sources_skipped == 0
    assert r2.sources_ingested == 0 and r2.sources_skipped == 1
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement `contextd/ingest/registry.py`**

```python
from __future__ import annotations
from typing import Iterable
from contextd.ingest.protocol import Adapter

_REGISTRY: dict[str, Adapter] = {}


def register(adapter: Adapter) -> None:
    _REGISTRY[adapter.source_type] = adapter


def get(source_type: str) -> Adapter:
    if source_type not in _REGISTRY:
        raise KeyError(f"no adapter registered for source_type={source_type!r}")
    return _REGISTRY[source_type]


def all_adapters() -> Iterable[Adapter]:
    return _REGISTRY.values()
```

- [ ] **Step 4: Implement `contextd/ingest/pipeline.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from contextd.config import get_settings
from contextd.ingest.embedder import Embedder, default_embedder
from contextd.ingest.protocol import Adapter, ChunkDraft, EdgeDraft, SourceCandidate
from contextd.logging_ import get_logger
from contextd.storage.db import (
    insert_chunk, insert_corpus, insert_source, open_db, get_source_by_path,
)
from contextd.storage.models import Source
from contextd.storage.vectors import VectorStore

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestReport:
    sources_ingested: int = 0
    sources_skipped: int = 0
    sources_failed: int = 0
    chunks_written: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)


class IngestionPipeline:
    def __init__(self, *, embedder: Embedder | None = None, adapters: Iterable[Adapter] | None = None) -> None:
        self._embedder = embedder or default_embedder()
        if adapters is not None:
            self._adapters = list(adapters)
        else:
            # Lazy import to avoid circulars
            from contextd.ingest.adapters import load_default_adapters
            self._adapters = list(load_default_adapters())

    def _select_adapter(self, path: Path, source_type: str | None) -> Adapter:
        if source_type:
            for a in self._adapters:
                if a.source_type == source_type: return a
            raise ValueError(f"no adapter with source_type={source_type!r}")
        for a in self._adapters:
            if a.can_handle(path): return a
        raise ValueError(f"no adapter handles path={path}")

    def ingest(self, *, path: Path, corpus: str, source_type: str | None = None, force: bool = False) -> IngestReport:
        adapter = self._select_adapter(path, source_type)
        conn = open_db(corpus)
        insert_corpus(
            conn, name=corpus,
            embed_model=self._embedder.model_name,
            embed_dim=self._embedder.dim,
            created_at=datetime.now(timezone.utc),
            schema_version=get_settings().schema_version,
        )
        vs = VectorStore.open(corpus=corpus, embed_dim=self._embedder.dim, model_name=self._embedder.model_name)

        ingested = skipped = failed = total_chunks = 0
        errors: list[str] = []

        for candidate in adapter.sources(path):
            existing = get_source_by_path(conn, corpus=corpus, path=str(candidate.path))
            if existing and existing.content_hash == candidate.content_hash and not force:
                skipped += 1
                continue
            try:
                n = self._write_source(conn, vs, adapter, candidate, corpus)
                ingested += 1
                total_chunks += n
            except Exception as e:
                failed += 1
                errors.append(f"{candidate.path}: {e!r}")
                log.error("ingest.source_failed", path=str(candidate.path), error=repr(e))

        # Audit log
        conn.execute(
            "INSERT INTO audit_log(occurred_at, actor, action, target, details_json) VALUES (?, 'cli', 'ingest', ?, ?)",
            (datetime.now(timezone.utc).isoformat(), str(path), f'{{"ingested":{ingested},"skipped":{skipped}}}'),
        )
        conn.commit()
        return IngestReport(
            sources_ingested=ingested, sources_skipped=skipped, sources_failed=failed,
            chunks_written=total_chunks, errors=tuple(errors),
        )

    def _write_source(self, conn, vs: VectorStore, adapter: Adapter, candidate: SourceCandidate, corpus: str) -> int:
        now = datetime.now(timezone.utc)
        # 1) parse chunks + metadata
        chunks: list[ChunkDraft] = list(adapter.parse(candidate))
        meta = adapter.metadata(candidate)
        # 2) insert source
        source_id = insert_source(
            conn, corpus=corpus, source_type=candidate.source_type, path=str(candidate.path),
            content_hash=candidate.content_hash, ingested_at=now, chunk_count=len(chunks),
            status="active", title=candidate.title, source_mtime=candidate.source_mtime,
        )
        # 3) insert source_meta
        for k, v in meta.items():
            conn.execute("INSERT INTO source_meta(source_id, key, value) VALUES (?, ?, ?)", (source_id, k, v))
        # 4) insert chunks
        ordinal_to_id: dict[int, int] = {}
        for ch in chunks:
            cid = insert_chunk(
                conn, source_id=source_id, ordinal=ch.ordinal, token_count=ch.token_count,
                content=ch.content, offset_start=ch.offset_start, offset_end=ch.offset_end,
                section_label=ch.section_label, scope=ch.scope, role=ch.role,
                chunk_timestamp=ch.chunk_timestamp,
            )
            ordinal_to_id[ch.ordinal] = cid
            for k, v in ch.metadata.items():
                conn.execute("INSERT INTO chunk_meta(chunk_id, key, value) VALUES (?, ?, ?)", (cid, k, v))
        # 5) embed + upsert
        if chunks:
            vecs = self._embedder.embed([c.content for c in chunks])
            vs.upsert([ordinal_to_id[c.ordinal] for c in chunks], vecs)
        # 6) edges
        for e in adapter.edges(chunks):
            tgt = ordinal_to_id.get(e.target_ordinal) if e.target_ordinal is not None else None
            conn.execute(
                "INSERT INTO edge(source_chunk_id, target_chunk_id, target_hint, edge_type, label, weight) VALUES (?, ?, ?, ?, ?, ?)",
                (ordinal_to_id[e.source_ordinal], tgt, e.target_hint, e.edge_type, e.label, e.weight),
            )
        conn.commit()
        return len(chunks)
```

- [ ] **Step 5: Run test — expect pass. Commit.**

```bash
git add contextd/ingest/
git commit -m "feat(ingest): orchestration pipeline with hash idempotency"
```

---

## Task 4: PDF adapter

**Files:**
- Create: `contextd/ingest/adapters/pdf.py`
- Create: `contextd/ingest/adapters/__init__.py`
- Test: `tests/integration/ingest/test_pdf.py`
- Fixtures: `tests/fixtures/pdfs/{sample-a,sample-b,sample-c}.pdf` (small public-domain arXiv papers — download manually and commit, ≤ 2 MB each)

Per PRD §14.2: primary parser `pymupdf4llm.to_markdown`; fallback `pypdf`; target 512 tokens, max 1024, split on paragraph then sentence; section-aware (classify headings by keyword map); title from first-page layout.

- [ ] **Step 1: Write the test**

```python
# tests/integration/ingest/test_pdf.py
from pathlib import Path
import pytest
from contextd.ingest.adapters.pdf import PDFAdapter

pytestmark = pytest.mark.integration

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "pdfs"

def test_handles_pdf_extension():
    a = PDFAdapter()
    assert a.can_handle(FIX / "sample-a.pdf")
    assert not a.can_handle(FIX / "nope.txt")

def test_sources_one_per_pdf_file():
    a = PDFAdapter()
    candidates = list(a.sources(FIX))
    # 3 fixtures in directory
    assert len({c.path for c in candidates}) >= 3
    for c in candidates:
        assert c.source_type == "pdf"
        assert c.content_hash.startswith("sha256:")

def test_parses_into_section_labeled_chunks():
    a = PDFAdapter()
    [cand] = [c for c in a.sources(FIX) if c.path.name == "sample-a.pdf"]
    chunks = list(a.parse(cand))
    assert chunks, "at least one chunk expected"
    labels = {c.section_label for c in chunks if c.section_label}
    assert labels & {"abstract", "introduction", "methods", "results", "discussion", "conclusion", "other"}
    for c in chunks:
        assert c.token_count <= 1024
        assert c.metadata.get("pdf_page") is not None

def test_metadata_extracts_title():
    a = PDFAdapter()
    [cand] = [c for c in a.sources(FIX) if c.path.name == "sample-a.pdf"]
    meta = a.metadata(cand)
    # title comes from first-page layout; require either title presence or pdf_authors_list
    assert cand.title or "pdf_authors_list" in meta
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement** (core logic; engineer finishes the section-classification heuristics)

```python
# contextd/ingest/adapters/pdf.py
from __future__ import annotations
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import pymupdf4llm  # markdown extraction
import pymupdf
from tokenizers import Tokenizer
from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate

_SECTION_PATTERNS = {
    "abstract": re.compile(r"^#+\s*abstract\b", re.I | re.M),
    "introduction": re.compile(r"^#+\s*(introduction|1\s+introduction)\b", re.I | re.M),
    "methods": re.compile(r"^#+\s*(methods?|materials? and methods?|approach)\b", re.I | re.M),
    "results": re.compile(r"^#+\s*(results?|experiments?|evaluation)\b", re.I | re.M),
    "discussion": re.compile(r"^#+\s*discussion\b", re.I | re.M),
    "conclusion": re.compile(r"^#+\s*(conclusions?|summary)\b", re.I | re.M),
    "references": re.compile(r"^#+\s*(references|bibliography)\b", re.I | re.M),
}

_TARGET_TOKENS = 512
_MAX_TOKENS = 1024
_MIN_FILE_BYTES = 4 * 1024  # PRD §14.2: skip < 4KB
_MAX_FILE_BYTES = 500 * 1024 * 1024


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


class PDFAdapter:
    source_type: str = "pdf"

    def __init__(self) -> None:
        # BGE-M3 tokenizer for accurate counting
        self._tok = Tokenizer.from_pretrained("BAAI/bge-m3")

    def can_handle(self, path: Path) -> bool:
        return path.is_dir() or path.suffix.lower() == ".pdf"

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        files: list[Path] = []
        if path.is_dir():
            files = sorted(path.rglob("*.pdf"))
        elif path.suffix.lower() == ".pdf":
            files = [path]
        for f in files:
            size = f.stat().st_size
            if size < _MIN_FILE_BYTES or size > _MAX_FILE_BYTES:
                continue
            try:
                title = self._title(f)
            except Exception:
                title = None
            yield SourceCandidate(
                path=f, source_type="pdf", canonical_id=str(f),
                content_hash=_sha256_file(f), title=title,
                source_mtime=datetime.fromtimestamp(f.stat().st_mtime, timezone.utc),
            )

    def _title(self, f: Path) -> str | None:
        doc = pymupdf.open(str(f))
        try:
            page0 = doc[0]
            blocks = page0.get_text("blocks")
            if not blocks: return None
            # largest-area block with most characters = title heuristic
            blocks.sort(key=lambda b: -(len(b[4])))
            return (blocks[0][4].strip().splitlines() or [None])[0]
        finally:
            doc.close()

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        # Primary: pymupdf4llm markdown with page markers
        try:
            md_pages = pymupdf4llm.to_markdown(str(source.path), page_chunks=True)  # list[dict]
        except Exception:
            yield from self._fallback_pypdf(source)
            return

        section = "other"
        ordinal = 0
        for page in md_pages:
            page_num = page.get("metadata", {}).get("page", 1)
            text = page.get("text", "")
            # Update section based on headings found
            for label, pat in _SECTION_PATTERNS.items():
                if pat.search(text):
                    section = label
            if section == "references":
                # PRD: references excluded from retrieval by default
                continue
            for piece in self._split_by_budget(text):
                yield ChunkDraft(
                    ordinal=ordinal, content=piece,
                    token_count=self._count_tokens(piece),
                    section_label=section,
                    metadata={"pdf_page": str(page_num)},
                )
                ordinal += 1

    def _fallback_pypdf(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        import pypdf
        reader = pypdf.PdfReader(str(source.path))
        ordinal = 0
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip(): continue
            for piece in self._split_by_budget(text):
                yield ChunkDraft(
                    ordinal=ordinal, content=piece,
                    token_count=self._count_tokens(piece),
                    section_label="other",
                    metadata={"pdf_page": str(i)},
                )
                ordinal += 1

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        # First-page authors extraction (heuristic)
        doc = pymupdf.open(str(source.path))
        try:
            page0_text = doc[0].get_text()
        finally:
            doc.close()
        meta: dict[str, str] = {}
        # arXiv id pattern
        m = re.search(r"arXiv:(\d{4}\.\d{4,5})", page0_text)
        if m: meta["arxiv_id"] = m.group(1)
        # DOI pattern
        m = re.search(r"\b10\.\d{4,9}/\S+\b", page0_text)
        if m: meta["doi"] = m.group(0).rstrip(".)")
        # Author line: first non-empty line after title, split on comma
        lines = [ln.strip() for ln in page0_text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            authors = lines[1]
            if 4 <= len(authors) <= 300 and "," in authors:
                meta["pdf_authors_list"] = authors
        return meta

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        return iter(())  # v0.1: pdf_cites deferred

    def _count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def _split_by_budget(self, text: str) -> Iterable[str]:
        # Paragraph-first split, then sentence fallback, capping at _MAX_TOKENS.
        paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        buf: list[str] = []
        buf_tok = 0
        for para in paragraphs:
            t = self._count_tokens(para)
            if t > _MAX_TOKENS:
                # Sentence split
                for sent in re.split(r"(?<=[.!?])\s+", para):
                    st = self._count_tokens(sent)
                    if buf_tok + st > _TARGET_TOKENS and buf:
                        yield "\n".join(buf).strip()
                        buf, buf_tok = [], 0
                    buf.append(sent); buf_tok += st
            else:
                if buf_tok + t > _TARGET_TOKENS and buf:
                    yield "\n".join(buf).strip()
                    buf, buf_tok = [], 0
                buf.append(para); buf_tok += t
        if buf:
            yield "\n".join(buf).strip()
```

- [ ] **Step 4: Wire into registry `contextd/ingest/adapters/__init__.py`**

```python
from __future__ import annotations
from contextd.ingest.protocol import Adapter
from contextd.ingest.adapters.pdf import PDFAdapter


def load_default_adapters() -> list[Adapter]:
    from contextd.ingest.adapters.claude_export import ClaudeExportAdapter
    from contextd.ingest.adapters.git_repo import GitRepoAdapter
    return [PDFAdapter(), ClaudeExportAdapter(), GitRepoAdapter()]
```

- [ ] **Step 5: Run test — expect pass. Commit.**

```bash
git add contextd/ingest/adapters/ tests/integration/ingest/test_pdf.py tests/fixtures/pdfs/
git commit -m "feat(ingest): PDF adapter with section-aware chunking (pymupdf4llm primary, pypdf fallback)"
```

---

## Task 5: Claude export adapter

**Files:**
- Create: `contextd/ingest/adapters/claude_export.py`
- Test: `tests/integration/ingest/test_claude.py`
- Fixture: `tests/fixtures/claude/export.json` — JSON matching Anthropic's Claude.ai export schema, ≥ 3 conversations, ≥ 20 messages total, sanitized.

Per PRD §14.3: schema `conversations[].chat_messages[]{uuid, text, sender, created_at}`; one source per conversation (path `<export>#conversations/<uuid>`); one chunk per message; `conversation_next`/`conversation_prev` edges as a linked list.

- [ ] **Step 1: Test**

```python
# tests/integration/ingest/test_claude.py
from pathlib import Path
import pytest
from contextd.ingest.adapters.claude_export import ClaudeExportAdapter

pytestmark = pytest.mark.integration
FIX = Path(__file__).resolve().parents[2] / "fixtures" / "claude" / "export.json"

def test_sources_one_per_conversation():
    a = ClaudeExportAdapter()
    cands = list(a.sources(FIX))
    assert len(cands) >= 3
    # path uses #conversations/<uuid> fragment
    assert all("#conversations/" in c.canonical_id for c in cands)

def test_parse_message_count_matches_source_meta():
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    meta = a.metadata(cand)
    assert chunks
    assert int(meta["message_count"]) == len(chunks)

def test_roles_are_user_or_assistant():
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    assert {c.role for c in chunks} <= {"user", "assistant"}

def test_edges_form_linked_list():
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    edges = list(a.edges(chunks))
    # N messages → 2*(N-1) edges (next + prev)
    assert len(edges) == 2 * (len(chunks) - 1)
    assert {e.edge_type for e in edges} == {"conversation_next", "conversation_prev"}
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/ingest/adapters/claude_export.py
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer
from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate


class ClaudeExportAdapter:
    source_type: str = "claude_export"

    def __init__(self) -> None:
        self._tok = Tokenizer.from_pretrained("BAAI/bge-m3")

    def can_handle(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".json"

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        if not path.is_file():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        # Accept top-level list or {"conversations": [...]}
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        for conv in conversations:
            uuid = conv.get("uuid") or conv.get("id")
            if not uuid: continue
            title = conv.get("name") or conv.get("title") or ""
            canonical = f"{path}#conversations/{uuid}"
            # Content hash over the canonical JSON of the single conversation
            canon = json.dumps(conv, sort_keys=True, ensure_ascii=False)
            h = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
            mtime = None
            if ts := conv.get("updated_at") or conv.get("created_at"):
                mtime = _parse_iso(ts)
            yield SourceCandidate(
                path=Path(canonical), source_type="claude_export",
                canonical_id=canonical, content_hash=h, title=title or None,
                source_mtime=mtime, metadata={"uuid": uuid},
            )

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        file_path_str, _, frag = source.canonical_id.partition("#conversations/")
        data = json.loads(Path(file_path_str).read_text(encoding="utf-8"))
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        conv = next((c for c in conversations if (c.get("uuid") or c.get("id")) == frag), None)
        if conv is None: return
        for i, msg in enumerate(conv.get("chat_messages", []) or conv.get("messages", [])):
            text = (msg.get("text") or msg.get("content") or "").strip()
            if not text: continue
            sender = msg.get("sender") or "user"
            role = "assistant" if sender in ("assistant", "claude") else "user"
            ts_raw = msg.get("created_at")
            yield ChunkDraft(
                ordinal=i, content=text,
                token_count=len(self._tok.encode(text, add_special_tokens=False).ids),
                role=role,
                chunk_timestamp=_parse_iso(ts_raw) if ts_raw else None,
                metadata={"message_id": msg.get("uuid") or msg.get("id") or str(i)},
            )

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        file_path_str, _, frag = source.canonical_id.partition("#conversations/")
        data = json.loads(Path(file_path_str).read_text(encoding="utf-8"))
        conversations = data if isinstance(data, list) else data.get("conversations", [])
        conv = next((c for c in conversations if (c.get("uuid") or c.get("id")) == frag), None)
        if conv is None: return {}
        messages = conv.get("chat_messages", []) or conv.get("messages", [])
        meta: dict[str, str] = {"message_count": str(len([m for m in messages if (m.get("text") or m.get("content"))]))}
        if ts := conv.get("created_at"): meta["created_at"] = ts
        if ts := conv.get("updated_at"): meta["updated_at"] = ts
        if url := conv.get("url"): meta["conversation_url"] = url
        return meta

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        for i in range(len(chunks) - 1):
            yield EdgeDraft(source_ordinal=i, target_ordinal=i + 1, edge_type="conversation_next")
            yield EdgeDraft(source_ordinal=i + 1, target_ordinal=i, edge_type="conversation_prev")


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None
```

- [ ] **Step 4: Run test — expect pass. Commit.**

```bash
git add contextd/ingest/adapters/claude_export.py tests/integration/ingest/test_claude.py tests/fixtures/claude/
git commit -m "feat(ingest): Claude export adapter with turn chunking + conversation edges"
```

---

## Task 6: Git repo adapter

**Files:**
- Create: `contextd/ingest/adapters/git_repo.py`
- Test: `tests/integration/ingest/test_git.py`
- Fixture: `tests/fixtures/git/tiny-repo.tar.gz` containing `.git/`, ~5 `.py` files, a `README.md`, a `.gitignore` excluding `__pycache__/`
- Modify: `pyproject.toml` to add `tree-sitter-python==0.23.6`, `tree-sitter-typescript==0.23.2`, `tree-sitter-javascript==0.23.1`, `tree-sitter-rust==0.23.2`, `tree-sitter-go==0.23.4`, `tree-sitter-java==0.23.5`

Per PRD §14.4: one SOURCE per repo; enumerate via `git ls-files`; skip binaries, >1MB, `.gitignore`'d; tree-sitter-supported → per top-level declaration; whole-file fallback (<1024 tok) else 512/64-overlap sliding window; YAML/TOML/JSON whole-file.

- [ ] **Step 1: Test**

```python
# tests/integration/ingest/test_git.py
import tarfile
from pathlib import Path
import pytest
from contextd.ingest.adapters.git_repo import GitRepoAdapter

pytestmark = pytest.mark.integration


@pytest.fixture
def tiny_repo(tmp_path):
    archive = Path(__file__).resolve().parents[2] / "fixtures" / "git" / "tiny-repo.tar.gz"
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path)
    return tmp_path / "tiny-repo"

def test_one_source_per_repo(tiny_repo):
    a = GitRepoAdapter()
    cands = list(a.sources(tiny_repo))
    assert len(cands) == 1
    assert cands[0].source_type == "git_repo"

def test_chunks_have_scope_for_python_functions(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    chunks = list(a.parse(cand))
    py_chunks = [c for c in chunks if c.metadata.get("language") == "python"]
    scopes = {c.scope for c in py_chunks if c.scope}
    assert scopes, "expected at least one function/class scope"

def test_gitignored_paths_are_skipped(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    chunks = list(a.parse(cand))
    assert not any("__pycache__" in c.metadata.get("file_path", "") for c in chunks)

def test_source_metadata_has_commit_hash(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    meta = a.metadata(cand)
    assert meta.get("repo_head_commit")
    assert meta.get("repo_branch")
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/ingest/adapters/git_repo.py
from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import pygit2
from tokenizers import Tokenizer
from tree_sitter import Language, Parser
from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate

_TARGET_TOKENS = 512
_OVERLAP_TOKENS = 64
_MAX_FILE_BYTES = 1 * 1024 * 1024

_LANG_BY_EXT: dict[str, str] = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".json": "json",
}

_DECL_QUERY_BY_LANG: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @name) @decl
        (class_definition name: (identifier) @name) @decl
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @decl
        (class_declaration name: (type_identifier) @name) @decl
        (interface_declaration name: (type_identifier) @name) @decl
        (method_definition name: (property_identifier) @name) @decl
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @decl
        (class_declaration name: (identifier) @name) @decl
        (method_definition name: (property_identifier) @name) @decl
    """,
    "rust": """
        (function_item name: (identifier) @name) @decl
        (struct_item name: (type_identifier) @name) @decl
        (enum_item name: (type_identifier) @name) @decl
        (trait_item name: (type_identifier) @name) @decl
        (impl_item type: (type_identifier) @name) @decl
    """,
    "go": """
        (function_declaration name: (identifier) @name) @decl
        (method_declaration name: (field_identifier) @name) @decl
        (type_declaration (type_spec name: (type_identifier) @name)) @decl
    """,
    "java": """
        (method_declaration name: (identifier) @name) @decl
        (class_declaration name: (identifier) @name) @decl
        (interface_declaration name: (identifier) @name) @decl
    """,
}


def _load_lang(name: str) -> Language | None:
    try:
        mod = __import__(f"tree_sitter_{name}")
        return Language(mod.language())
    except Exception:
        return None


class GitRepoAdapter:
    source_type: str = "git_repo"

    def __init__(self) -> None:
        self._tok = Tokenizer.from_pretrained("BAAI/bge-m3")
        self._parsers: dict[str, Parser] = {}
        for lang in _DECL_QUERY_BY_LANG:
            if (L := _load_lang(lang)):
                p = Parser()
                p.language = L
                self._parsers[lang] = p

    def can_handle(self, path: Path) -> bool:
        return path.is_dir() and (path / ".git").exists()

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        if not self.can_handle(path): return
        repo = pygit2.Repository(str(path))
        head_commit = str(repo.head.target) if not repo.head_is_unborn else ""
        try:
            branch = repo.head.shorthand
        except pygit2.GitError:
            branch = "(detached)"
        h = hashlib.sha256((str(path) + head_commit).encode()).hexdigest()
        yield SourceCandidate(
            path=path, source_type="git_repo", canonical_id=str(path),
            content_hash="sha256:" + h,
            title=path.name,
            source_mtime=datetime.now(timezone.utc),
            metadata={"repo_head_commit": head_commit, "repo_branch": branch},
        )

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        repo = pygit2.Repository(str(source.path))
        head_commit = str(repo.head.target) if not repo.head_is_unborn else ""
        ordinal = 0
        # Use `git ls-files` equivalent: iterate HEAD tree
        if repo.head_is_unborn: return
        tree = repo.head.peel(pygit2.Commit).tree
        for file_path, blob_bytes in _walk_tree(tree, repo, prefix=""):
            if len(blob_bytes) > _MAX_FILE_BYTES: continue
            if _is_binary(blob_bytes): continue
            ext = Path(file_path).suffix.lower()
            lang = _LANG_BY_EXT.get(ext, "text")
            text = blob_bytes.decode("utf-8", errors="replace")
            base_meta = {"file_path": file_path, "language": lang, "commit_hash": head_commit}
            if lang in self._parsers:
                yield from self._parse_with_tree_sitter(text, lang, base_meta, ordinal_start=ordinal)
                ordinal += sum(1 for _ in self._parse_with_tree_sitter(text, lang, base_meta, 0))
            else:
                for piece in self._split_text(text):
                    yield ChunkDraft(
                        ordinal=ordinal, content=piece,
                        token_count=self._count(piece),
                        scope="", metadata=base_meta,
                    )
                    ordinal += 1

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        return dict(source.metadata)

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        return iter(())  # v0.1: code_imports deferred

    def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def _split_text(self, text: str) -> Iterable[str]:
        tokens = self._tok.encode(text, add_special_tokens=False).ids
        if len(tokens) <= 1024:
            yield text
            return
        step = _TARGET_TOKENS - _OVERLAP_TOKENS
        for i in range(0, len(tokens), step):
            window = tokens[i : i + _TARGET_TOKENS]
            yield self._tok.decode(window)

    def _parse_with_tree_sitter(
        self, text: str, lang: str, base_meta: dict[str, str], ordinal_start: int
    ) -> Iterable[ChunkDraft]:
        parser = self._parsers[lang]
        source_bytes = text.encode("utf-8")
        tree = parser.parse(source_bytes)
        language = parser.language
        query = language.query(_DECL_QUERY_BY_LANG[lang])

        # Group captures by (start_byte, end_byte) of enclosing @decl match
        decls: list[tuple[int, int, str]] = []  # (start, end, name)
        captures = query.captures(tree.root_node)
        # tree-sitter 0.23: captures = {capture_name: [node, ...]}
        decl_nodes = captures.get("decl", [])
        name_nodes = captures.get("name", [])
        # Pair each @decl node with its first-descendant @name node.
        for decl in decl_nodes:
            name_text = ""
            for n in name_nodes:
                if n.start_byte >= decl.start_byte and n.end_byte <= decl.end_byte:
                    name_text = source_bytes[n.start_byte : n.end_byte].decode("utf-8", "replace")
                    break
            decls.append((decl.start_byte, decl.end_byte, name_text))

        if not decls:
            # Whole-file fallback if nothing matched
            if self._count(text) <= 1024:
                yield ChunkDraft(
                    ordinal=ordinal_start, content=text, token_count=self._count(text),
                    scope="", offset_start=0, offset_end=len(source_bytes),
                    metadata=base_meta,
                )
            else:
                for i, piece in enumerate(self._split_text(text)):
                    yield ChunkDraft(
                        ordinal=ordinal_start + i, content=piece,
                        token_count=self._count(piece), scope="", metadata=base_meta,
                    )
            return

        decls.sort()
        # Module-top = bytes before the first declaration
        module_top = source_bytes[: decls[0][0]].decode("utf-8", "replace").rstrip()
        idx = ordinal_start
        if module_top.strip() and self._count(module_top) <= _TARGET_TOKENS:
            yield ChunkDraft(
                ordinal=idx, content=module_top, token_count=self._count(module_top),
                scope="", offset_start=0, offset_end=decls[0][0], metadata=base_meta,
            )
            idx += 1
        for start, end, name in decls:
            body = source_bytes[start:end].decode("utf-8", "replace")
            if self._count(body) <= _MAX_FILE_BYTES:
                yield ChunkDraft(
                    ordinal=idx, content=body, token_count=self._count(body),
                    scope=name, offset_start=start, offset_end=end, metadata=base_meta,
                )
                idx += 1
            else:
                for j, piece in enumerate(self._split_text(body)):
                    yield ChunkDraft(
                        ordinal=idx, content=piece, token_count=self._count(piece),
                        scope=name,
                        metadata={**base_meta, "split_of": f"{name}#{j}"},
                    )
                    idx += 1


def _walk_tree(tree, repo, prefix: str) -> Iterable[tuple[str, bytes]]:
    for entry in tree:
        name = entry.name
        full = f"{prefix}{name}"
        if entry.type_str == "tree":
            yield from _walk_tree(repo[entry.id], repo, prefix=f"{full}/")
        elif entry.type_str == "blob":
            yield full, repo[entry.id].data


def _is_binary(b: bytes) -> bool:
    return b"\x00" in b[:8192]
```

- [ ] **Step 4: Run test — expect pass. Commit.**

```bash
git add contextd/ingest/adapters/git_repo.py tests/integration/ingest/test_git.py tests/fixtures/git/ pyproject.toml uv.lock
git commit -m "feat(ingest): git adapter with tree-sitter scoped chunks + gitignore respect"
```

---

## Task 7: CLI `contextd ingest`

**Files:**
- Create: `contextd/cli/main.py`
- Create: `contextd/cli/commands/ingest.py`
- Test: `tests/integration/cli/test_ingest.py`

- [ ] **Step 1: Test with CLI runner**

```python
# tests/integration/cli/test_ingest.py
from pathlib import Path
import pytest
from typer.testing import CliRunner
from contextd.cli.main import app

pytestmark = pytest.mark.integration
PDFS = Path(__file__).resolve().parents[2] / "fixtures" / "pdfs"


def test_ingest_pdf_directory(tmp_contextd_home):
    r = CliRunner().invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    assert r.exit_code == 0, r.output
    assert "Ingested" in r.output

def test_ingest_idempotent_second_run_skips(tmp_contextd_home):
    runner = CliRunner()
    r1 = runner.invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    r2 = runner.invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert "skipped" in r2.output.lower()

def test_ingest_unknown_path_fails_nonzero(tmp_contextd_home):
    r = CliRunner().invoke(app, ["ingest", "/does/not/exist"])
    assert r.exit_code != 0
```

- [ ] **Step 2: Run — expect fail.**

- [ ] **Step 3: Implement**

```python
# contextd/cli/main.py
from __future__ import annotations
import typer
from contextd.cli.commands import ingest as ingest_cmd

app = typer.Typer(no_args_is_help=True, add_completion=False, help="contextd — local-first personal RAG")
app.command(name="ingest", help="Ingest a path into a corpus.")(ingest_cmd.ingest)
```

```python
# contextd/cli/commands/ingest.py
from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from contextd.ingest.pipeline import IngestionPipeline

console = Console()


def ingest(
    path: Path = typer.Argument(..., exists=True, resolve_path=True),
    corpus: str = typer.Option("personal", "--corpus"),
    source_type: str | None = typer.Option(None, "--type"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    pipe = IngestionPipeline()
    report = pipe.ingest(path=path, corpus=corpus, source_type=source_type, force=force)
    console.print(
        f"[green]Ingested[/green] {report.sources_ingested} sources, "
        f"{report.chunks_written} chunks; "
        f"{report.sources_skipped} skipped, {report.sources_failed} failed."
    )
    if report.errors:
        for e in report.errors[:5]:
            console.print(f"[red]error:[/red] {e}")
    if report.sources_failed and report.sources_ingested == 0:
        raise typer.Exit(code=1)
```

- [ ] **Step 4: Run test — expect pass. Commit.**

```bash
git add contextd/cli/ tests/integration/cli/
git commit -m "feat(cli): contextd ingest subcommand with rich output"
```

---

## Phase 2 Exit Gate Checklist

- [ ] `uv run ruff check .` clean
- [ ] `uv run mypy contextd/` clean
- [ ] `uv run pytest -q` green (incl. `-m slow` for embedder at least once locally)
- [ ] CLI smoke: `uv run contextd ingest tests/fixtures/pdfs/ --corpus personal` produces 3+ sources
- [ ] CLI smoke: `uv run contextd ingest tests/fixtures/claude/export.json --corpus personal` produces N-conversation sources with `conversation_next`/`conversation_prev` edges
- [ ] CLI smoke: extract and ingest `tests/fixtures/git/tiny-repo.tar.gz` — scope fields populated for Python functions
- [ ] Re-running each command logs "skipped" for every source (hash idempotency)
- [ ] PRD §14.2.11 quality bar: random 20-chunk sample from 3 PDFs — ≥ 90% correct section labels
- [ ] PRD §14.3.11 quality bar: 50-message sample → 50 chunks, 49 next + 49 prev edges
- [ ] PRD §14.4.10 quality bar: ≥ 90% of top-level Python declarations have their own chunk with correct `scope`

Commit all work; move to Phase 3.
