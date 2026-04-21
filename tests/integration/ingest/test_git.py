import tarfile
from pathlib import Path

import pytest

from contextd.ingest.adapters.git_repo import GitRepoAdapter

pytestmark = pytest.mark.integration


@pytest.fixture
def tiny_repo(tmp_path):
    archive = Path(__file__).resolve().parents[2] / "fixtures" / "git" / "tiny-repo.tar.gz"
    with tarfile.open(archive) as tar:
        tar.extractall(tmp_path, filter="data")
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


def test_tsx_ext_uses_tsx_grammar():
    from contextd.ingest.adapters.git_repo import _LANG_BY_EXT, _LANG_ENTRY_OVERRIDE

    assert _LANG_BY_EXT[".tsx"] == "tsx"
    assert _LANG_ENTRY_OVERRIDE["tsx"] == "language_tsx"
    assert _LANG_BY_EXT[".ts"] == "typescript"
    assert _LANG_ENTRY_OVERRIDE["typescript"] == "language_typescript"


def test_name_nodes_sorted_for_stable_decl_pairing(tiny_repo):
    """Regression guard: pairing loop now sorts name_nodes by start_byte."""
    import inspect

    from contextd.ingest.adapters import git_repo

    src = inspect.getsource(git_repo)
    assert (
        "sorted(captures.get" in src or "sort(key=lambda" in src or ".sort(" in src
    ), "expected explicit sort of name_nodes by start_byte"
