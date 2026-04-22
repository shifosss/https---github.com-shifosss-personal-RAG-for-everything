from __future__ import annotations

import typer

from contextd import __version__


def version() -> None:
    typer.echo(f"contextd {__version__}")
