"""Integration tests for the eval harness v0."""

from __future__ import annotations

from pathlib import Path

import pytest

from contextd.eval.harness import run_eval

pytestmark = pytest.mark.integration


async def test_eval_runs_on_seed_file(
    tmp_contextd_home: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.integration.cli.test_query import _seed

    _seed(monkeypatch)
    seed = Path(__file__).resolve().parents[3] / "contextd" / "eval" / "seed_queries.json"
    result = await run_eval(seed, corpus="personal", k=5)
    assert result.n_queries >= 2
    assert 0.0 <= result.recall_at_5 <= 1.0
