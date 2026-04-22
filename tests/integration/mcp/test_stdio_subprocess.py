import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_mcp_stdio_lists_tools(tmp_contextd_home):
    root = Path(__file__).resolve().parents[3] / "mcp-server"
    assert (root / "dist" / "index.js").exists(), "run pnpm build in mcp-server first"

    proc = subprocess.Popen(
        ["node", str(root / "dist" / "index.js")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(req) + "\n").encode())
        proc.stdin.flush()
        assert proc.stdout is not None
        line = proc.stdout.readline().decode()
        data = json.loads(line)
        names = {t["name"] for t in data["result"]["tools"]}
        assert {
            "search_corpus",
            "fetch_chunk",
            "expand_context",
            "get_edges",
            "list_sources",
            "get_source",
            "list_corpora",
        } <= names
    finally:
        proc.terminate()
        proc.wait(timeout=5)
