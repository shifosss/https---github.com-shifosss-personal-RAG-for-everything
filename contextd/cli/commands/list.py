from __future__ import annotations

import json as _json

import typer
from rich.console import Console
from rich.table import Table

from contextd.storage.db import open_db

console = Console()


def list_(
    corpus: str = typer.Option("personal", "--corpus"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    conn = open_db(corpus)
    rows = conn.execute(
        "SELECT id, source_type, path, title, ingested_at, chunk_count FROM source "
        "WHERE status='active' AND corpus=? ORDER BY ingested_at DESC",
        (corpus,),
    ).fetchall()
    if as_json:
        typer.echo(_json.dumps([dict(r) for r in rows], default=str))
        return
    t = Table(title=f"Sources in {corpus}")
    for h in ("id", "type", "path", "title", "ingested_at", "chunks"):
        t.add_column(h)
    for r in rows:
        t.add_row(
            *(
                str(r[c])
                for c in ("id", "source_type", "path", "title", "ingested_at", "chunk_count")
            )
        )
    console.print(t)
