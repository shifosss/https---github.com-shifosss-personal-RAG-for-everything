"""PDF ingestion adapter.

Primary extraction: pymupdf4llm (markdown with page chunks).
Fallback: pypdf plain-text extraction.

Section labelling uses heading-pattern matching on the markdown output.
Tokenisation uses the BGE-M3 tokenizer (lazy-loaded via cached_property so
adapter construction is cheap in tests — deviation from spec's eager __init__
load; functionally identical, saves ~0.4 s per cold-start).
"""

from __future__ import annotations

import functools
import hashlib
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pymupdf  # type: ignore[import-untyped]
import pymupdf4llm  # type: ignore[import-untyped]
from tokenizers import Tokenizer  # type: ignore[import-untyped]

from contextd.ingest.protocol import ChunkDraft, EdgeDraft, SourceCandidate

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from contextd.storage.models import SourceType

# ---------------------------------------------------------------------------
# Section classification regexes (spec §14.3 — do not change without PRD bump)
# ---------------------------------------------------------------------------
_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
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
_MIN_FILE_BYTES = 4 * 1024  # PRD §14.2: skip files smaller than 4 KB
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB upper guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PDFAdapter:
    source_type: SourceType = "pdf"

    # ------------------------------------------------------------------
    # Tokenizer: lazy-loaded once per instance via cached_property.
    # Deviation from spec (which loads eagerly in __init__): avoids the
    # ~0.4 s HF tokenizer load during adapter registry construction and
    # keeps test cold-starts fast.  Semantics are identical; the tokenizer
    # is initialised on first call to _count_tokens / _split_by_budget.
    # ------------------------------------------------------------------
    @functools.cached_property
    def _tok(self) -> Tokenizer:
        return Tokenizer.from_pretrained("BAAI/bge-m3")

    # ------------------------------------------------------------------
    # Protocol: can_handle
    # ------------------------------------------------------------------

    def can_handle(self, path: Path) -> bool:
        return path.is_dir() or path.suffix.lower() == ".pdf"

    # ------------------------------------------------------------------
    # Protocol: sources
    # ------------------------------------------------------------------

    def sources(self, path: Path) -> Iterable[SourceCandidate]:
        files: list[Path]
        if path.is_dir():
            files = sorted(path.rglob("*.pdf"))
        elif path.suffix.lower() == ".pdf":
            files = [path]
        else:
            files = []

        for f in files:
            size = f.stat().st_size
            if size < _MIN_FILE_BYTES or size > _MAX_FILE_BYTES:
                continue
            try:
                title = self._title(f)
            except Exception:
                title = None
            yield SourceCandidate(
                path=f,
                source_type="pdf",
                canonical_id=str(f),
                content_hash=_sha256_file(f),
                title=title,
                source_mtime=datetime.fromtimestamp(f.stat().st_mtime, UTC),
            )

    # ------------------------------------------------------------------
    # Protocol: parse
    # ------------------------------------------------------------------

    def parse(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        try:
            md_pages: list[dict] = pymupdf4llm.to_markdown(str(source.path), page_chunks=True)
        except Exception:
            yield from self._fallback_pypdf(source)
            return

        section = "other"
        ordinal = 0

        for page in md_pages:
            page_num: int = page.get("metadata", {}).get("page", 1)
            text: str = page.get("text", "")

            # Update running section label based on headings in this page.
            # First matching section heading wins; prior section carries across
            # page boundaries when no heading appears on the page.
            # Fix: last-match-wins was silently dropping Discussion/Conclusion
            # body text on pages that also contained a References heading.
            for label, pat in _SECTION_PATTERNS.items():
                if pat.search(text):
                    section = label
                    break

            # PRD: references section excluded from retrieval by default.
            if section == "references":
                continue

            for piece in self._split_by_budget(text):
                yield ChunkDraft(
                    ordinal=ordinal,
                    content=piece,
                    token_count=self._count_tokens(piece),
                    section_label=section,
                    metadata={"pdf_page": str(page_num)},
                )
                ordinal += 1

    # ------------------------------------------------------------------
    # Protocol: metadata
    # ------------------------------------------------------------------

    def metadata(self, source: SourceCandidate) -> dict[str, str]:
        doc = pymupdf.open(str(source.path))
        try:
            page0_text: str = doc[0].get_text()
        finally:
            doc.close()

        meta: dict[str, str] = {}

        # arXiv ID
        m = re.search(r"arXiv:(\d{4}\.\d{4,5})", page0_text)
        if m:
            meta["arxiv_id"] = m.group(1)

        # DOI
        m = re.search(r"\b10\.\d{4,9}/\S+\b", page0_text)
        if m:
            meta["doi"] = m.group(0).rstrip(".)")

        # Author line heuristic: second non-empty line that looks like names.
        lines = [ln.strip() for ln in page0_text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            authors = lines[1]
            if 4 <= len(authors) <= 300 and "," in authors:
                meta["pdf_authors_list"] = authors

        return meta

    # ------------------------------------------------------------------
    # Protocol: edges
    # ------------------------------------------------------------------

    def edges(self, chunks: list[ChunkDraft]) -> Iterable[EdgeDraft]:
        # v0.1: pdf_cites edge extraction deferred to v0.2
        return iter(())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _title(self, f: Path) -> str | None:
        doc = pymupdf.open(str(f))
        try:
            page0 = doc[0]
            blocks = page0.get_text("blocks")
            if not blocks:
                return None
            # Largest-by-character-count block on page 0 = title heuristic.
            blocks.sort(key=lambda b: -len(b[4]))
            first_line = (blocks[0][4].strip().splitlines() or [None])[0]
            return first_line
        finally:
            doc.close()

    def _count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def _split_by_budget(self, text: str) -> Iterable[str]:
        """Paragraph-first split, sentence fallback, capped at _MAX_TOKENS."""
        paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        buf: list[str] = []
        buf_tok = 0

        for para in paragraphs:
            t = self._count_tokens(para)
            if t > _MAX_TOKENS:
                # Para too large — split by sentence.
                # Flush any pending buffer first so prior content is not mixed
                # with sentences from this oversized paragraph.
                if buf:
                    yield "\n".join(buf).strip()
                    buf, buf_tok = [], 0
                for sent in re.split(r"(?<=[.!?])\s+", para):
                    st = self._count_tokens(sent)
                    if buf_tok + st > _TARGET_TOKENS and buf:
                        yield "\n".join(buf).strip()
                        buf, buf_tok = [], 0
                    buf.append(sent)
                    buf_tok += st
            else:
                if buf_tok + t > _TARGET_TOKENS and buf:
                    yield "\n".join(buf).strip()
                    buf, buf_tok = [], 0
                buf.append(para)
                buf_tok += t

        if buf:
            yield "\n".join(buf).strip()

    def _fallback_pypdf(self, source: SourceCandidate) -> Iterable[ChunkDraft]:
        """Plain-text fallback when pymupdf4llm fails."""
        import pypdf

        reader = pypdf.PdfReader(str(source.path))
        ordinal = 0
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for piece in self._split_by_budget(text):
                yield ChunkDraft(
                    ordinal=ordinal,
                    content=piece,
                    token_count=self._count_tokens(piece),
                    section_label="other",
                    metadata={"pdf_page": str(i)},
                )
                ordinal += 1
