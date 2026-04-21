from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer
from rich.console import Console

from contextd.ingest.pipeline import IngestionPipeline

console = Console()


def ingest(
    path: Path = typer.Argument(..., exists=True, resolve_path=True),  # noqa: B008
    corpus: str = typer.Option("personal", "--corpus"),  # noqa: B008
    source_type: str | None = typer.Option(None, "--type"),  # noqa: B008
    force: bool = typer.Option(False, "--force"),  # noqa: B008
) -> None:
    pipe = IngestionPipeline()
    report = pipe.ingest(path=path, corpus=corpus, source_type=source_type, force=force)
    console.print(
        f"[green]Ingested[/green] {report.sources_ingested} sources, "
        f"{report.chunks_written} chunks; "
        f"{report.sources_skipped} skipped, {report.sources_failed} failed."
    )
    if report.errors:
        for e in report.errors[:5]:
            console.print(f"[red]error:[/red] {e}")
    if report.sources_failed and report.sources_ingested == 0:
        raise typer.Exit(code=1)
