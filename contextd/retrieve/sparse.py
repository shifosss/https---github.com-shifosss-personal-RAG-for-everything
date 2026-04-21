"""Sparse (BM25/FTS5) retrieval via SQLite full-text search.

PRD refs: §15 (sparse retrieval), §13.2 (SQLite FTS5).

SQLite's ``bm25()`` function returns *negative* scores (more negative = more
relevant), so we negate them before returning so that higher scores mean better
matches — consistent with the dense and fusion layers.
"""

from __future__ import annotations

import asyncio
import re

from contextd.storage.db import open_db


def _sanitize_fts(q: str) -> str:
    """Convert a raw query string to a safe FTS5 MATCH expression.

    Extracts word tokens and joins them with OR so that documents containing
    any query term are returned. Returns an empty string when no tokens are
    found (e.g. whitespace-only input), which the caller treats as a no-op.
    """
    tokens = re.findall(r"\w+", q)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else ""


async def sparse_search(
    *,
    query: str,
    corpus: str,
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k (chunk_id, bm25_score) pairs via FTS5.

    Scores are negated ``bm25()`` values so that higher = more relevant,
    matching the convention used by ``dense_search``.

    Args:
        query: Raw query string; empty / whitespace returns ``[]``.
        corpus: Corpus name — determines which SQLite DB to query.
        k: Maximum number of results to return.

    Returns:
        List of ``(chunk_id, score)`` sorted by score descending.
    """
    fts_q = _sanitize_fts(query)
    if not fts_q:
        return []

    def _run() -> list[tuple[int, float]]:
        conn = open_db(corpus)
        rows = conn.execute(
            "SELECT rowid, bm25(chunk_fts) AS score FROM chunk_fts "
            "WHERE chunk_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (fts_q, k),
        ).fetchall()
        # bm25() returns negative values; negate for ascending similarity.
        return [(int(r["rowid"]), -float(r["score"])) for r in rows]

    return await asyncio.to_thread(_run)
