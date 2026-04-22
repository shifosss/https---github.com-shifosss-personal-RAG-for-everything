"""Full eval runner — Recall@k, MRR, per-tag breakdown, LLM-judge aggregate.

PRD ref: §16.7 Phase 5 ship gate:
    Recall@5 ≥ 0.80, Recall@10 ≥ 0.90, MRR ≥ 0.60, judge ≥ 6.5.

Exit code 0 if every gate passes, 1 otherwise. Run via:

    uv run python -m contextd.eval.run contextd/eval/seed_queries.json \\
        --corpus eval [--no-rerank] [--no-judge]
"""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from contextd.eval.judge import judge_result
from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request

if TYPE_CHECKING:
    from pathlib import Path

    from contextd.storage.models import ChunkResult

_GATE_R5 = 0.80
_GATE_R10 = 0.90
_GATE_MRR = 0.60
_GATE_JUDGE = 6.5


@dataclass(frozen=True)
class EvalReport:
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    judge_mean: float | None
    judge_n: int
    per_tag: dict[str, dict[str, float]]
    gate_passed: bool


def _match(r: ChunkResult, keywords: list[str], allowed_types: list[str]) -> bool:
    if allowed_types and r.source.source_type not in allowed_types:
        return False
    content = r.chunk.content.lower()
    return any(kw in content for kw in keywords)


async def _score_one(
    q: dict[str, object],
    corpus: str,
    rerank: bool,
    judge: bool,
) -> tuple[int | None, int | None]:
    """Return (first_hit_position_1_based_or_None, judge_score_or_None)."""
    req = build_request(
        query=str(q["query"]),
        corpus=str(q.get("corpus") or corpus),
        limit=10,
        rerank=rerank,
    )
    results, _ = await retrieve(req)
    keywords = [str(k).lower() for k in q.get("expected_keywords") or []]
    allowed_types = [str(t) for t in q.get("expected_source_types") or []]
    pos: int | None = None
    for i, r in enumerate(results, start=1):
        if _match(r, keywords, allowed_types):
            pos = i
            break

    judge_score: int | None = None
    if judge and results:
        top_text = "\n---\n".join(r.chunk.content for r in results[:5])
        judge_score = await judge_result(query=str(q["query"]), result_text=top_text)
    return pos, judge_score


async def run(
    seed_path: Path,
    corpus: str,
    *,
    rerank: bool = True,
    judge: bool = True,
) -> EvalReport:
    queries: list[dict[str, object]] = json.loads(seed_path.read_text())
    n = len(queries) or 1

    r5 = r10 = 0
    mrr_sum = 0.0
    judge_scores: list[int] = []
    per_tag_hits: dict[str, list[int]] = {}

    for q in queries:
        pos, judge_score = await _score_one(q, corpus, rerank, judge)
        if pos is not None and pos <= 5:
            r5 += 1
        if pos is not None and pos <= 10:
            r10 += 1
        if pos is not None:
            mrr_sum += 1.0 / pos
        if judge_score is not None:
            judge_scores.append(judge_score)
        tags = q.get("tags") or []
        for tag in tags:
            per_tag_hits.setdefault(str(tag), []).append(1 if pos is not None and pos <= 5 else 0)

    per_tag = {
        tag: {"recall_at_5": sum(xs) / len(xs), "n": float(len(xs))}
        for tag, xs in per_tag_hits.items()
    }
    judge_mean = statistics.mean(judge_scores) if judge_scores else None

    recall_at_5 = r5 / n
    recall_at_10 = r10 / n
    mrr = mrr_sum / n
    gate_passed = (
        recall_at_5 >= _GATE_R5
        and recall_at_10 >= _GATE_R10
        and mrr >= _GATE_MRR
        and (judge_mean is None or judge_mean >= _GATE_JUDGE)
    )

    return EvalReport(
        n_queries=n,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        judge_mean=judge_mean,
        judge_n=len(judge_scores),
        per_tag=per_tag,
        gate_passed=gate_passed,
    )


def main() -> None:
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="contextd.eval.run")
    ap.add_argument("seed", type=Path)
    ap.add_argument("--corpus", default="personal")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    report = asyncio.run(
        run(
            args.seed,
            args.corpus,
            rerank=not args.no_rerank,
            judge=not args.no_judge,
        )
    )
    print(json.dumps(asdict(report), indent=2))
    sys.exit(0 if report.gate_passed else 1)


if __name__ == "__main__":
    main()
