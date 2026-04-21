"""query subcommand — retrieve chunks matching a natural-language query.

PRD refs: §15 (retrieval pipeline), §16 Phase 3.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from contextd.retrieve.pipeline import retrieve
from contextd.retrieve.preprocess import build_request

if TYPE_CHECKING:
    from contextd.storage.models import ChunkResult

console = Console()


def query(
    query: str = typer.Argument(...),  # noqa: B008
    corpus: str = typer.Option("personal", "--corpus"),  # noqa: B008
    limit: int = typer.Option(10, "--limit"),  # noqa: B008
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),  # noqa: B008
    rewrite: bool = typer.Option(False, "--rewrite/--no-rewrite"),  # noqa: B008
    as_json: bool = typer.Option(False, "--json"),  # noqa: B008
) -> None:
    """Retrieve chunks matching a query."""
    req = build_request(
        query=query,
        corpus=corpus,
        limit=limit,
        rerank=rerank,
        rewrite=rewrite,
    )
    results, trace = asyncio.run(retrieve(req))

    if as_json:
        payload = {
            "query": {"original": req.query, "corpus": req.corpus},
            "results": [_result_to_dict(r) for r in results],
            "trace": asdict(trace),
        }
        console.print(json.dumps(payload, default=str))
        return

    table = Table(title=f"Top {len(results)} for: {query}")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("section")
    table.add_column("content", overflow="fold")
    for r in results:
        table.add_row(
            str(r.rank),
            f"{r.score:.3f}",
            r.source.path,
            r.chunk.section_label or "-",
            r.chunk.content[:200],
        )
    console.print(table)
    console.print(
        f"[dim]trace={trace.trace_id} latency={trace.latency_ms}ms "
        f"reranker={trace.reranker_used or 'off'}[/dim]"
    )


def _result_to_dict(r: ChunkResult) -> dict[str, object]:
    return {
        "chunk": {
            "id": r.chunk.id,
            "source_id": r.chunk.source_id,
            "content": r.chunk.content,
            "ordinal": r.chunk.ordinal,
            "section_label": r.chunk.section_label,
            "scope": r.chunk.scope,
            "role": r.chunk.role,
            "token_count": r.chunk.token_count,
        },
        "source": {
            "id": r.source.id,
            "path": r.source.path,
            "type": r.source.source_type,
            "title": r.source.title,
        },
        "score": r.score,
        "rank": r.rank,
        "metadata": r.metadata,
        "edges": [
            {
                "type": e.edge_type,
                "target_chunk_id": e.target_chunk_id,
                "target_hint": e.target_hint,
            }
            for e in r.edges
        ],
    }
