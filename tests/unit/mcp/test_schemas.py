import pytest

from contextd.mcp.schemas import SearchFilters, SearchRequest


def test_search_request_defaults() -> None:
    r = SearchRequest(query="hello")
    assert r.corpus == "personal"
    assert r.limit == 10
    assert r.rerank is True
    assert r.rewrite is False  # D-30


def test_search_request_clamps_limit() -> None:
    r = SearchRequest(query="h", limit=500)
    assert r.limit == 100


def test_search_request_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        SearchRequest(query="")


def test_filters_parse_source_types() -> None:
    f = SearchFilters(source_types=["pdf", "git_repo"])
    assert "pdf" in f.source_types
