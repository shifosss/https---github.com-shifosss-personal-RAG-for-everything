from __future__ import annotations

import json as _json

import pytest
from typer.testing import CliRunner

from contextd.cli.main import app

pytestmark = pytest.mark.integration


def test_status_json_shape(tmp_contextd_home):
    r = CliRunner().invoke(app, ["status", "--json"])
    assert r.exit_code == 0, r.output
    data = _json.loads(r.stdout)
    assert "version" in data
    assert "data_root" in data
    assert data["reranker"]["model"]


def test_version_prints_version(tmp_contextd_home):
    r = CliRunner().invoke(app, ["version"])
    assert r.exit_code == 0
    assert "contextd" in r.output
