"""Eval harness v0 — recall@k and MRR over a seed query file.

PRD refs: §13.8.3 (coverage targets), §16 Phase 5 (30-query gate).
The real 30-query set and Recall@5 ≥ 0.80 ship gate land in Phase 5.
This v0 harness validates the pipeline is exercisable and the metric
logic is correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class EvalResult:
    recall_at_5: float
    mrr: float
    n_queries: int


async def run_eval(seed_path: Path, corpus: str, k: int = 5) -> EvalResult:
    """Run keyword-hit eval over all queries in *seed_path*.

    Args:
        seed_path: Path to a JSON file containing a list of query dicts.
                   Each dict must have ``query`` and ``expected_keywords``
                   keys; ``corpus`` and ``tags`` are ignored (corpus comes
                   from the *corpus* arg).
        corpus: Corpus name to run retrieval against.
        k: Number of results to retrieve per query (Recall@k, MRR@k).

    Returns:
        ``EvalResult`` with recall_at_5, mrr, and n_queries.
    """
    queries: list[dict] = json.loads(seed_path.read_text())
    hits = 0
    mrr_sum = 0.0
    for q in queries:
        req = build_request(
            query=q["query"],
            corpus=corpus,
            limit=k,
            rerank=False,
            rewrite=False,
        )
        results, _ = await retrieve(req)
        kw = [kw.lower() for kw in q.get("expected_keywords", [])]
        positions = [
            i
            for i, r in enumerate(results, start=1)
            if any(k in r.chunk.content.lower() for k in kw)
        ]
        if positions:
            hits += 1
            mrr_sum += 1.0 / positions[0]
    n = len(queries) or 1
    return EvalResult(recall_at_5=hits / n, mrr=mrr_sum / n, n_queries=n)
