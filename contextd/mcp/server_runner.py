from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import uvicorn

from contextd.config import get_settings
from contextd.mcp.api import create_app


def run_http(host: str | None = None, port: int | None = None) -> None:
    s = get_settings()
    uvicorn.run(
        create_app(),
        host=host or s.mcp_host,
        port=port or s.mcp_port,
        log_level="info",
        access_log=False,
    )


def run_mcp_stdio() -> None:
    """Start the TS MCP server as a child process piped to our stdio."""
    root = Path(__file__).resolve().parents[2] / "mcp-server"
    dist = root / "dist" / "index.js"
    if not dist.exists():
        raise RuntimeError(
            f"MCP server build missing at {dist}. Run 'cd mcp-server && pnpm install && pnpm run build' first."
        )
    node = os.environ.get("CONTEXTD_NODE_BIN", "node")
    proc = subprocess.Popen(
        [node, str(dist)],
        stdin=0,
        stdout=1,
        stderr=2,
        env=os.environ.copy(),
    )

    def _forward(sig: int, _frame: object) -> None:
        proc.send_signal(sig)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)
    raise SystemExit(proc.wait())
