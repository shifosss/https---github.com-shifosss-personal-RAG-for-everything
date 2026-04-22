"""eval subcommand — run the retrieval eval harness + print a JSON report.

Exits non-zero if the PRD §16.7 ship gate is not met.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import typer

from contextd.eval.run import run


def eval_(
    seed: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),  # noqa: B008
    corpus: str = typer.Option("personal", "--corpus"),  # noqa: B008
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),  # noqa: B008
    judge: bool = typer.Option(True, "--judge/--no-judge"),  # noqa: B008
) -> None:
    """Run the retrieval eval harness and enforce the ship gate."""
    report = asyncio.run(run(seed, corpus, rerank=rerank, judge=judge))
    typer.echo(json.dumps(asdict(report), indent=2, default=str))
    if not report.gate_passed:
        raise typer.Exit(code=1)
