"""
Generate synthetic PDF fixtures for PDF adapter integration tests.

Run with: uv run python tests/fixtures/_generate_pdfs.py

Uses pymupdf (fitz) to create minimal PDFs with proper heading/body structure.
pymupdf4llm detects bold+larger-font blocks as markdown headings (# H1).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent / "pdfs"
OUT.mkdir(parents=True, exist_ok=True)

# Font constants
FONT = "helv"  # Helvetica — always available in pymupdf
TITLE_SIZE = 18.0
HEAD_SIZE = 14.0
BODY_SIZE = 11.0
LINE_H = 16  # line height for body
HEAD_H = 22  # line height for headings


def _add_heading(page: pymupdf.Page, y: float, text: str) -> float:
    """Insert a bold-looking heading using a larger font. Returns updated y."""
    page.insert_text(
        (50, y),
        text,
        fontname=FONT,
        fontsize=HEAD_SIZE,
        color=(0, 0, 0),
    )
    return y + HEAD_H


def _add_body(page: pymupdf.Page, y: float, text: str, max_width: int = 490) -> float:
    """Insert body text, wrapping long lines. Returns updated y."""
    words = text.split()
    line: list[str] = []
    lines: list[str] = []
    for word in words:
        trial = " ".join(line + [word])
        # crude character-width estimate: ~5.5px per char at 11pt
        if len(trial) * 5.5 > max_width and line:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        page.insert_text((50, y), ln, fontname=FONT, fontsize=BODY_SIZE, color=(0, 0, 0))
        y += LINE_H
    return y + 4  # small paragraph gap


LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
    "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
    "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum."
)

LOREM2 = (
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque "
    "laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi "
    "architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas "
    "sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione "
    "voluptatem sequi nesciunt."
)

LOREM3 = (
    "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium "
    "voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint "
    "occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt "
    "mollitia animi, id est laborum et dolorum fuga."
)


# ---------------------------------------------------------------------------
# sample-a.pdf  — 4-page arXiv lookalike
# ---------------------------------------------------------------------------
def make_sample_a(path: Path) -> None:
    doc = pymupdf.open()

    # Page 1: Title, authors, Abstract
    p1 = doc.new_page(width=595, height=842)
    y: float = 60
    p1.insert_text(
        (50, y),
        "A Lightweight Approach to Local RAG Retrieval",
        fontname=FONT,
        fontsize=TITLE_SIZE,
        color=(0, 0, 0),
    )
    y += 28
    p1.insert_text(
        (50, y),
        "Alex Zhang, Jane Smith, Bob Chen",
        fontname=FONT,
        fontsize=BODY_SIZE,
        color=(0, 0, 0),
    )
    y += 24
    y = _add_heading(p1, y, "Abstract")
    y = _add_body(p1, y, LOREM)
    y = _add_body(p1, y, LOREM2)

    # Page 2: Introduction
    p2 = doc.new_page(width=595, height=842)
    y = 60.0
    y = _add_heading(p2, y, "Introduction")
    y = _add_body(p2, y, LOREM)
    y = _add_body(p2, y, LOREM2)
    y = _add_body(p2, y, LOREM3)

    # Page 3: Methods + Results
    p3 = doc.new_page(width=595, height=842)
    y = 60.0
    y = _add_heading(p3, y, "Methods")
    y = _add_body(p3, y, LOREM)
    y = _add_body(p3, y, LOREM2)
    y = _add_heading(p3, y, "Results")
    y = _add_body(p3, y, LOREM3)

    # Page 4: Discussion + Conclusion + References
    p4 = doc.new_page(width=595, height=842)
    y = 60.0
    y = _add_heading(p4, y, "Discussion")
    y = _add_body(p4, y, LOREM)
    y = _add_heading(p4, y, "Conclusion")
    y = _add_body(p4, y, LOREM2)
    y = _add_heading(p4, y, "References")
    for i in range(1, 6):
        y = _add_body(p4, y, f"[{i}] Author et al. Some Journal 202{i}. DOI: 10.1000/xyz{i:03d}.")

    doc.save(str(path), deflate=True)
    doc.close()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# sample-b.pdf  — 2-section paper
# ---------------------------------------------------------------------------
def make_sample_b(path: Path) -> None:
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    y: float = 60
    p1.insert_text(
        (50, y),
        "Dense Retrieval at Scale: A Survey",
        fontname=FONT,
        fontsize=TITLE_SIZE,
        color=(0, 0, 0),
    )
    y += 28
    p1.insert_text(
        (50, y),
        "Maria Lopez, James Wu",
        fontname=FONT,
        fontsize=BODY_SIZE,
        color=(0, 0, 0),
    )
    y += 24
    y = _add_heading(p1, y, "Abstract")
    y = _add_body(p1, y, LOREM)
    y = _add_body(p1, y, LOREM2)

    p2 = doc.new_page(width=595, height=842)
    y = 60.0
    y = _add_heading(p2, y, "Introduction")
    y = _add_body(p2, y, LOREM)
    y = _add_body(p2, y, LOREM2)
    y = _add_body(p2, y, LOREM3)

    doc.save(str(path), deflate=True)
    doc.close()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# sample-c.pdf  — 2-section paper
# ---------------------------------------------------------------------------
def make_sample_c(path: Path) -> None:
    doc = pymupdf.open()

    p1 = doc.new_page(width=595, height=842)
    y: float = 60
    p1.insert_text(
        (50, y),
        "Hybrid Sparse-Dense Retrieval for Personal Knowledge Bases",
        fontname=FONT,
        fontsize=TITLE_SIZE,
        color=(0, 0, 0),
    )
    y += 28
    p1.insert_text(
        (50, y),
        "Sarah Kim, David Park, Lisa Chen",
        fontname=FONT,
        fontsize=BODY_SIZE,
        color=(0, 0, 0),
    )
    y += 24
    y = _add_heading(p1, y, "Abstract")
    y = _add_body(p1, y, LOREM2)
    y = _add_body(p1, y, LOREM3)

    p2 = doc.new_page(width=595, height=842)
    y = 60.0
    y = _add_heading(p2, y, "Introduction")
    y = _add_body(p2, y, LOREM)
    y = _add_body(p2, y, LOREM2)
    y = _add_body(p2, y, LOREM3)

    doc.save(str(path), deflate=True)
    doc.close()
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    make_sample_a(OUT / "sample-a.pdf")
    make_sample_b(OUT / "sample-b.pdf")
    make_sample_c(OUT / "sample-c.pdf")
    print("Done.")
