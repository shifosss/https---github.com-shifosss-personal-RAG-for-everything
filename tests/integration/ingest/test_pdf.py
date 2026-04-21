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


def test_parse_emits_all_non_references_sections_on_multi_heading_page():
    """Regression: prior first-match-wins logic was last-match-wins and dropped
    Discussion/Conclusion content on pages that also contained References.
    """
    a = PDFAdapter()
    [cand] = [c for c in a.sources(FIX) if c.path.name == "sample-a.pdf"]
    chunks = list(a.parse(cand))
    labels = {c.section_label for c in chunks if c.section_label}
    # sample-a.pdf has Discussion and Conclusion on page 4 alongside References
    assert (
        "discussion" in labels or "conclusion" in labels
    ), f"Expected discussion or conclusion section; got {labels}"


def test_split_by_budget_flushes_buffer_before_sentence_mode():
    """Regression: huge paragraphs caused prior buf content to mix with split sentences."""
    a = PDFAdapter()
    prior = "Short intro paragraph. " * 10  # ~20 tokens
    huge = "The core claim is that local retrieval is feasible. " * 200  # >1024 tokens
    text = prior + "\n\n" + huge
    chunks = list(a._split_by_budget(text))
    # With the fix, the first chunk should contain only the prior paragraph text
    # (no huge-paragraph sentences).
    assert chunks[0].strip().startswith("Short intro paragraph."), chunks[0][:100]
    assert (
        "The core claim" not in chunks[0]
    ), f"buffer not flushed; first chunk leaked huge para: {chunks[0][:200]}"
