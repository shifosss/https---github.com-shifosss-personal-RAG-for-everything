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
    assert labels & {
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "other",
    }
    for c in chunks:
        assert c.token_count <= 1024
        assert c.metadata.get("pdf_page") is not None


def test_metadata_extracts_title():
    a = PDFAdapter()
    [cand] = [c for c in a.sources(FIX) if c.path.name == "sample-a.pdf"]
    meta = a.metadata(cand)
    assert cand.title or "pdf_authors_list" in meta
