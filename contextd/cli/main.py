from __future__ import annotations

import typer

from contextd.cli.commands import ingest as ingest_cmd

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="contextd — local-first personal RAG",
)


@app.callback()
def _root() -> None:
    """contextd — local-first personal RAG."""


app.command(name="ingest", help="Ingest a path into a corpus.")(ingest_cmd.ingest)
