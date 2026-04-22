from __future__ import annotations

import json as _json
import os

import typer
from rich.console import Console

from contextd import __version__
from contextd.config import get_settings

console = Console()


def status(as_json: bool = typer.Option(False, "--json")) -> None:
    s = get_settings()
    payload = {
        "version": __version__,
        "data_root": str(s.data_root),
        "default_corpus": s.default_corpus,
        "reranker": {
            "provider": s.reranker_provider,
            "model": s.reranker_model,
            "api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        },
        "network_default": "offline",
    }
    if as_json:
        typer.echo(_json.dumps(payload, default=str))
        return
    for k, v in payload.items():
        console.print(f"[bold]{k}[/bold]: {v}")
