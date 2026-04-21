import tarfile
from pathlib import Path

import pytest

from contextd.ingest.adapters.git_repo import GitRepoAdapter

pytestmark = pytest.mark.integration


@pytest.fixture
def tiny_repo(tmp_path):
    archive = Path(__file__).resolve().parents[2] / "fixtures" / "git" / "tiny-repo.tar.gz"
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path)
    return tmp_path / "tiny-repo"


def test_one_source_per_repo(tiny_repo):
    a = GitRepoAdapter()
    cands = list(a.sources(tiny_repo))
    assert len(cands) == 1
    assert cands[0].source_type == "git_repo"


def test_chunks_have_scope_for_python_functions(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    chunks = list(a.parse(cand))
    py_chunks = [c for c in chunks if c.metadata.get("language") == "python"]
    scopes = {c.scope for c in py_chunks if c.scope}
    assert scopes, "expected at least one function/class scope"


def test_gitignored_paths_are_skipped(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    chunks = list(a.parse(cand))
    assert not any("__pycache__" in c.metadata.get("file_path", "") for c in chunks)


def test_source_metadata_has_commit_hash(tiny_repo):
    a = GitRepoAdapter()
    cand = next(iter(a.sources(tiny_repo)))
    meta = a.metadata(cand)
    assert meta.get("repo_head_commit")
    assert meta.get("repo_branch")
