"""Unit tests for reciprocal_rank_fusion."""

from __future__ import annotations

from contextd.retrieve.fusion import reciprocal_rank_fusion


def test_rrf_merges_two_lists() -> None:
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(3, 5.0), (2, 4.0), (4, 1.0)]
    out = reciprocal_rank_fusion([(dense, sparse)], k=60, top_n=4)
    ids = [c for c, _ in out]
    assert ids[:2] == [2, 3] or ids[:2] == [3, 2]
    assert 4 in ids
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_single_list_preserves_order() -> None:
    dense = [(10, 0.9), (20, 0.8), (30, 0.1)]
    sparse: list[tuple[int, float]] = []
    out = reciprocal_rank_fusion([(dense, sparse)], k=60, top_n=3)
    assert [c for c, _ in out] == [10, 20, 30]


def test_rrf_multiple_queries_accumulate() -> None:
    q1: tuple[list[tuple[int, float]], list[tuple[int, float]]] = ([(1, 0.0)], [])
    q2: tuple[list[tuple[int, float]], list[tuple[int, float]]] = (
        [(1, 0.0), (2, 0.0)],
        [],
    )
    out = reciprocal_rank_fusion([q1, q2], k=60, top_n=2)
    ids = [c for c, _ in out]
    assert ids[0] == 1
