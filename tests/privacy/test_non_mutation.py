"""Enforce: ingestion never mutates the source tree.

PRD §2 design principle: contextd reads source files and writes only
into its own data home. This test sha256-hashes every file in a
temporary copy of the fixtures tree before ingest, runs the full
ingest pipeline, then re-hashes. Any diff (added, removed, or changed
file) fails the test.

We copy the fixtures to a scratch dir rather than hashing the checked-in
tree directly, so the assertion is crisp and does not depend on repo
cleanliness.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from contextd.cli.main import app

pytestmark = pytest.mark.privacy


def _tree_sha256(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@pytest.mark.parametrize(
    "fixture_subdir",
    ["pdfs", "claude"],
    ids=["pdf_adapter", "claude_export_adapter"],
)
def test_ingest_does_not_mutate_source(
    tmp_contextd_home: Path,
    stub_embedder: object,
    stub_tokenizer: None,
    tmp_path: Path,
    fixture_subdir: str,
) -> None:
    src = Path(__file__).resolve().parents[1] / "fixtures" / fixture_subdir
    scratch = tmp_path / fixture_subdir
    shutil.copytree(src, scratch)

    pre = _tree_sha256(scratch)
    assert pre, f"fixture tree {src} is empty — broken precondition"

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", str(scratch), "--corpus", "personal"])
    assert result.exit_code == 0, result.output

    post = _tree_sha256(scratch)
    added = set(post) - set(pre)
    removed = set(pre) - set(post)
    changed = {k for k in pre if k in post and pre[k] != post[k]}

    assert not added, f"ingest added files: {added}"
    assert not removed, f"ingest removed files: {removed}"
    assert not changed, f"ingest modified files: {changed}"
