"""Integration tests for the full eval runner.

Stubs ``contextd.eval.run.retrieve`` with a canned result list so the
scoring logic (Recall@5, Recall@10, MRR, per-tag, judge aggregation,
gate pass/fail) is exercised without touching storage or the embedder.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import contextd.eval.run as run_mod
from contextd.eval.run import EvalReport, run
from contextd.storage.models import Chunk, ChunkResult, QueryTrace, Source

pytestmark = pytest.mark.integration


def _chunk(cid: int, content: str) -> Chunk:
    return Chunk(id=cid, source_id=1, ordinal=0, content=content, token_count=1)


def _source(source_type: str = "pdf") -> Source:
    return Source(
        id=1,
        corpus="eval",
        source_type=source_type,  # type: ignore[arg-type]
        path="/tmp/x",
        content_hash="sha256:x",
        ingested_at=datetime.now(UTC),
        chunk_count=1,
        status="active",
    )


def _result(rank: int, content: str, source_type: str = "pdf") -> ChunkResult:
    return ChunkResult(
        chunk=_chunk(rank, content),
        source=_source(source_type),
        score=1.0 / rank,
        rank=rank,
        metadata={},
        edges=(),
    )


def _trace() -> QueryTrace:
    return QueryTrace(
        trace_id="01",
        latency_ms=1,
        dense_candidates=0,
        sparse_candidates=0,
        reranker_used=None,
        rewriter_used=None,
    )


async def _stub_retrieve_allhits(_req: Any) -> tuple[list[ChunkResult], QueryTrace]:
    return [_result(1, "this contains NEGEX and dataclass and slugify SHA256_HEX")], _trace()


async def _stub_retrieve_nohits(_req: Any) -> tuple[list[ChunkResult], QueryTrace]:
    return [_result(1, "completely unrelated lorem ipsum content")], _trace()


@pytest.fixture
def seed_file(tmp_path: Path) -> Path:
    queries = [
        {
            "id": "q1",
            "query": "NegEx negation",
            "corpus": "eval",
            "expected_keywords": ["negex"],
            "expected_source_types": ["pdf"],
            "tags": ["direct"],
        },
        {
            "id": "q2",
            "query": "dataclass frozen",
            "corpus": "eval",
            "expected_keywords": ["dataclass"],
            "tags": ["paraphrase"],
        },
        {
            "id": "q3",
            "query": "slugify",
            "corpus": "eval",
            "expected_keywords": ["slugify"],
            "tags": ["code_identifier"],
        },
    ]
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(queries))
    return p


async def test_run_scores_all_hits_no_judge(
    seed_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "retrieve", _stub_retrieve_allhits)
    report = await run(seed_file, corpus="eval", rerank=False, judge=False)
    assert isinstance(report, EvalReport)
    assert report.n_queries == 3
    assert report.recall_at_5 == 1.0
    assert report.recall_at_10 == 1.0
    assert report.mrr == 1.0
    assert report.judge_mean is None
    assert report.judge_n == 0
    assert set(report.per_tag.keys()) == {"direct", "paraphrase", "code_identifier"}
    assert report.gate_passed is True


async def test_run_scores_no_hits(seed_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_mod, "retrieve", _stub_retrieve_nohits)
    report = await run(seed_file, corpus="eval", rerank=False, judge=False)
    assert report.recall_at_5 == 0.0
    assert report.recall_at_10 == 0.0
    assert report.mrr == 0.0
    assert report.gate_passed is False


async def test_run_judge_none_does_not_fail_gate(
    seed_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "retrieve", _stub_retrieve_allhits)

    async def fake_judge(**_: Any) -> int | None:
        return None

    monkeypatch.setattr(run_mod, "judge_result", fake_judge)
    report = await run(seed_file, corpus="eval", rerank=False, judge=True)
    assert report.judge_mean is None
    assert report.judge_n == 0
    assert report.gate_passed is True


async def test_run_judge_mean_below_gate_fails(
    seed_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_mod, "retrieve", _stub_retrieve_allhits)

    async def fake_judge(**_: Any) -> int | None:
        return 3  # below 6.5 threshold

    monkeypatch.setattr(run_mod, "judge_result", fake_judge)
    report = await run(seed_file, corpus="eval", rerank=False, judge=True)
    assert report.judge_mean == 3.0
    assert report.gate_passed is False


async def test_run_filters_by_source_type(seed_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def stub_wrong_type(_req: Any) -> tuple[list[ChunkResult], QueryTrace]:
        # NEGEX is in the content but source_type is git_repo, not pdf — q1 expects pdf.
        return [_result(1, "NEGEX content", source_type="git_repo")], _trace()

    monkeypatch.setattr(run_mod, "retrieve", stub_wrong_type)
    report = await run(seed_file, corpus="eval", rerank=False, judge=False)
    # q1 should MISS (wrong source_type); q2 and q3 also miss (content mismatch).
    assert report.recall_at_5 == 0.0
