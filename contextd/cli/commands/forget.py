from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from contextd.storage.db import open_db
from contextd.storage.vectors import VectorStore

console = Console()


def forget(
    path: str = typer.Argument(...),
    corpus: str = typer.Option("personal", "--corpus"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    resolved = str(Path(path).expanduser().resolve())
    conn = open_db(corpus)
    row = conn.execute(
        "SELECT id FROM source WHERE corpus = ? AND path = ?", (corpus, resolved)
    ).fetchone()
    if row is None:
        console.print(f"[red]no source at {resolved} in corpus {corpus}[/red]")
        raise typer.Exit(1)
    sid = row["id"]
    chunk_ids = [r["id"] for r in conn.execute("SELECT id FROM chunk WHERE source_id = ?", (sid,))]
    if dry_run:
        console.print(f"would delete source {sid} + {len(chunk_ids)} chunks + vectors")
        return
    if not yes:
        typer.confirm(f"Delete source {resolved} ({len(chunk_ids)} chunks)?", abort=True)
    conn.execute("DELETE FROM source WHERE id = ?", (sid,))
    conn.execute(
        "INSERT INTO audit_log(occurred_at, actor, action, target, details_json) "
        "VALUES (?, 'cli', 'forget', ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            resolved,
            f'{{"source_id":{sid},"chunks":{len(chunk_ids)}}}',
        ),
    )
    conn.commit()
    corp = conn.execute(
        "SELECT embed_dim, embed_model FROM corpus WHERE name = ?", (corpus,)
    ).fetchone()
    if chunk_ids and corp:
        vs = VectorStore.open(
            corpus=corpus,
            embed_dim=corp["embed_dim"],
            model_name=corp["embed_model"],
        )
        vs.delete(chunk_ids)
    console.print(f"[green]deleted[/green] source {sid} and {len(chunk_ids)} chunks")
