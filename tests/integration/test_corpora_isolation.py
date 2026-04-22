from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from contextd.cli.main import app

pytestmark = pytest.mark.integration

# Fixtures live at tests/fixtures/pdfs/ — one level above the integration/ dir.
FIXTURE_PDF = Path(__file__).resolve().parent.parent / "fixtures" / "pdfs"


class FakeEmbedder:
    """Stub embedder — avoids 2 GB BGE-M3 download in CI."""

    model_name = "fake-4d"
    dim = 4

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 4), dtype=np.float32)
        return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype(np.float32)


@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextd.ingest import pipeline as pipe_mod

    monkeypatch.setattr(pipe_mod, "default_embedder", lambda: FakeEmbedder())


def test_corpora_are_isolated(tmp_contextd_home: Path) -> None:
    runner = CliRunner()

    # Ingest fixtures into the *research* corpus.
    r1 = runner.invoke(app, ["ingest", str(FIXTURE_PDF), "--corpus", "research"])
    assert r1.exit_code == 0, r1.output

    # The *personal* corpus must remain empty.
    r2 = runner.invoke(app, ["list", "--corpus", "personal", "--json"])
    assert r2.exit_code == 0, r2.output
    personal_sources = json.loads(r2.stdout)
    assert personal_sources == [], "personal corpus should be empty"

    # The *research* corpus must contain at least one source.
    r3 = runner.invoke(app, ["list", "--corpus", "research", "--json"])
    assert r3.exit_code == 0, r3.output
    research_sources = json.loads(r3.stdout)
    assert len(research_sources) >= 1, "research corpus should have fixture PDFs"

    # On-disk layout: each corpus has its own chunks.db.
    research_db = tmp_contextd_home / "corpora" / "research" / "chunks.db"
    assert research_db.exists(), f"expected {research_db}"


def test_list_on_missing_corpus_is_empty_not_error(tmp_contextd_home: Path) -> None:
    """open_db auto-creates the corpus dir but no sources exist → empty list."""
    r = CliRunner().invoke(app, ["list", "--corpus", "does_not_exist", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout) == []
