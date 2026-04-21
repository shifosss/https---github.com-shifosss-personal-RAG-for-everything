import pytest

from contextd.retrieve.preprocess import build_request


def test_nfc_normalizes():
    req = build_request(query="café", corpus="personal")  # NFC of "cafe\u0301"
    assert len(req.query) == 4


def test_length_cap_truncates_at_2000():
    req = build_request(query="x" * 5000, corpus="personal")
    assert len(req.query) == 2000


def test_trace_id_is_ulid_like():
    req = build_request(query="x", corpus="personal")
    assert len(req.trace_id) == 26  # ULID canonical length


def test_filter_defaults_empty():
    req = build_request(query="x", corpus="personal")
    assert req.filters.source_types == ()
    assert req.filters.metadata == {}


def test_invalid_corpus_raises():
    with pytest.raises(ValueError):
        build_request(query="x", corpus="")
