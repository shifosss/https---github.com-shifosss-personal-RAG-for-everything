"""Reciprocal Rank Fusion (RRF) for combining dense and sparse result lists.

PRD refs: §15.5 (RRF formula: score = Σ 1 / (k + rank)).

Rank is 1-based. The constant ``k=60`` is the standard Cormack et al. (2009)
default and is left configurable for experimentation.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    per_query: list[tuple[list[tuple[int, float]], list[tuple[int, float]]]],
    *,
    k: int = 60,
    top_n: int = 50,
) -> list[tuple[int, float]]:
    """Merge dense + sparse ranked lists from one or more queries via RRF.

    For each query, iterates both the dense and sparse lists, assigning each
    candidate chunk ``1 / (k + rank)`` (1-based rank) and accumulating into a
    shared score dictionary.  Supports multi-query fusion: pass multiple
    ``(dense, sparse)`` pairs and scores are summed across queries.

    Args:
        per_query: Sequence of ``(dense_hits, sparse_hits)`` pairs, where each
            hits list is ordered best-first and contains ``(chunk_id, score)``
            tuples.  Scores are not used — only rank position matters.
        k: RRF smoothing constant (default 60, per Cormack et al. 2009).
        top_n: Maximum number of results to return.

    Returns:
        List of ``(chunk_id, rrf_score)`` sorted by score descending,
        truncated to ``top_n`` entries.
    """
    acc: dict[int, float] = defaultdict(float)
    for dense, sparse in per_query:
        for rank_idx, (cid, _) in enumerate(dense):
            acc[cid] += 1.0 / (k + rank_idx + 1)
        for rank_idx, (cid, _) in enumerate(sparse):
            acc[cid] += 1.0 / (k + rank_idx + 1)
    return sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_n]
