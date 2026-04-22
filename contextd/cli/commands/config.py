from __future__ import annotations

import typer
from rich.console import Console

from contextd.config import get_settings

console = Console()
config_app = typer.Typer(help="Config introspection.")


@config_app.command("show")
def show() -> None:
    s = get_settings()
    for k, v in s.model_dump().items():
        console.print(f"{k} = {v}")


@config_app.command("path")
def path() -> None:
    console.print(str(get_settings().data_root))
