from __future__ import annotations

import typer

from contextd.mcp.server_runner import run_http, run_mcp_stdio


def serve(
    mcp_only: bool = typer.Option(False, "--mcp-only"),
    http_only: bool = typer.Option(False, "--http-only"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8787, "--port"),
) -> None:
    if mcp_only and http_only:
        raise typer.BadParameter("--mcp-only and --http-only are mutually exclusive")
    if http_only:
        run_http(host, port)
        return
    if mcp_only:
        run_mcp_stdio()
        return
    import multiprocessing

    p = multiprocessing.Process(
        target=run_http,
        kwargs={"host": host, "port": port},
        daemon=True,
    )
    p.start()
    try:
        run_mcp_stdio()
    finally:
        p.terminate()
