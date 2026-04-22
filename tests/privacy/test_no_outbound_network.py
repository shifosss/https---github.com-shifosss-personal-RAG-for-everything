"""Enforce: ingest + query complete with every non-loopback socket blocked.

The PRD §2 design principle is "local-first, no telemetry." This test
pins that behavior: with ``socket.socket.connect`` replaced by a guard
that raises ``OutboundBlocked`` on any non-loopback IP, a full ingest
+ query cycle must still succeed. If any subsystem reaches out to the
network by default, the guard trips and this test fails.

Embedding model download is sidestepped by stubbing ``default_embedder``
at both consuming import sites (see conftest).
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from contextd.cli.main import app

pytestmark = pytest.mark.privacy

_LOOPBACK_V4 = ipaddress.IPv4Network("127.0.0.0/8")
_LOOPBACK_V6 = ipaddress.ip_address("::1")


class OutboundBlocked(AssertionError):
    """Raised when something tries to connect to a non-loopback destination."""


def _is_loopback(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in _LOOPBACK_V4
    return ip == _LOOPBACK_V6


@pytest.fixture
def block_outbound(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        # AF_UNIX addresses are strings (paths) or bytes — local IPC, always allowed.
        if not isinstance(address, tuple):
            return real_connect(self, address)
        host = address[0]
        if not _is_loopback(str(host)):
            raise OutboundBlocked(f"outbound connect to {host} blocked")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def test_ingest_and_query_no_outbound(
    tmp_contextd_home: Path,
    stub_embedder: object,
    stub_tokenizer: None,
    block_outbound: None,
) -> None:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "pdfs"

    runner = CliRunner()
    ingest_result = runner.invoke(
        app,
        ["ingest", str(fixtures), "--corpus", "personal"],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    query_result = runner.invoke(
        app,
        [
            "query",
            "abstract",
            "--corpus",
            "personal",
            "--limit",
            "1",
            "--no-rerank",
            "--no-rewrite",
            "--json",
        ],
    )
    assert query_result.exit_code == 0, query_result.output
