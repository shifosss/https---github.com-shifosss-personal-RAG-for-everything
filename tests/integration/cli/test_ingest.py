from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from contextd.cli.main import app

pytestmark = pytest.mark.integration

PDFS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "pdfs"


class FakeEmbedder:
    """Stub embedder — avoids 2GB BGE-M3 download in CI."""

    model_name = "fake-4d"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 4), dtype=np.float32)
        return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace default_embedder so CLI tests don't load the real model."""
    from contextd.ingest import pipeline as pipe_mod

    monkeypatch.setattr(pipe_mod, "default_embedder", lambda: FakeEmbedder())


def test_ingest_pdf_directory(tmp_contextd_home: Path) -> None:
    r = CliRunner().invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    assert r.exit_code == 0, r.output
    assert "Ingested" in r.output


def test_ingest_idempotent_second_run_skips(tmp_contextd_home: Path) -> None:
    runner = CliRunner()
    r1 = runner.invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    r2 = runner.invoke(app, ["ingest", str(PDFS), "--corpus", "personal"])
    assert r1.exit_code == 0 and r2.exit_code == 0
    assert "skipped" in r2.output.lower()


def test_ingest_unknown_path_fails_nonzero(tmp_contextd_home: Path) -> None:
    r = CliRunner().invoke(app, ["ingest", "/does/not/exist"])
    assert r.exit_code != 0
