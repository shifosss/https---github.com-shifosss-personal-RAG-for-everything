from __future__ import annotations

import typer

from contextd.cli.commands import ingest as ingest_cmd
from contextd.cli.commands import query as query_cmd

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="contextd — local-first personal RAG",
)


@app.callback()
def _root() -> None:
    """contextd — local-first personal RAG."""


app.command(name="ingest", help="Ingest a path into a corpus.")(ingest_cmd.ingest)
app.command(name="query", help="Retrieve chunks matching a query.")(query_cmd.query)

from contextd.cli.commands import eval as eval_cmd  # noqa: E402
from contextd.cli.commands import forget as forget_cmd  # noqa: E402
from contextd.cli.commands import list as list_cmd  # noqa: E402
from contextd.cli.commands import serve as serve_cmd  # noqa: E402
from contextd.cli.commands import status as status_cmd  # noqa: E402
from contextd.cli.commands import version as version_cmd  # noqa: E402
from contextd.cli.commands.config import config_app  # noqa: E402

app.command(name="serve", help="Start the MCP server (stdio) + HTTP backend.")(serve_cmd.serve)
app.command(name="list", help="List sources in a corpus.")(list_cmd.list_)
app.command(name="forget", help="Delete a source + cascade.")(forget_cmd.forget)
app.command(name="status", help="Print config and runtime status.")(status_cmd.status)
app.command(name="version", help="Print contextd version.")(version_cmd.version)
app.command(name="eval", help="Run the retrieval eval harness.")(eval_cmd.eval_)
app.add_typer(config_app, name="config")
