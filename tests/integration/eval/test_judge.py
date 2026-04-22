"""Integration tests for the LLM-as-judge scorer.

The real Anthropic client is replaced by a stub so the test suite
can run without ``ANTHROPIC_API_KEY`` and without network access.
"""

from __future__ import annotations

from typing import Any

import pytest

import contextd.eval.judge as judge_mod
from contextd.eval.judge import judge_result

pytestmark = pytest.mark.integration


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessagesOK:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def create(self, **_: Any) -> _FakeResponse:
        return _FakeResponse(self._payload)


class _FakeMessagesRaises:
    def create(self, **_: Any) -> _FakeResponse:
        raise ConnectionError("api down")


class _FakeClient:
    def __init__(self, messages: Any) -> None:
        self.messages = messages


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    judge_mod._anthropic_client.cache_clear()


async def test_judge_returns_0_to_10_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_FakeMessagesOK('{"score": 7, "rationale": "on topic"}'))
    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: client)
    score = await judge_result(query="q", result_text="ctx")
    assert score == 7


async def test_judge_clamps_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_FakeMessagesOK('{"score": 42}'))
    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: client)
    score = await judge_result(query="q", result_text="ctx")
    assert score == 10


async def test_judge_returns_none_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_FakeMessagesRaises())
    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: client)
    score = await judge_result(query="q", result_text="ctx")
    assert score is None


async def test_judge_returns_none_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_FakeMessagesOK("not json at all"))
    monkeypatch.setattr(judge_mod, "_anthropic_client", lambda: client)
    score = await judge_result(query="q", result_text="ctx")
    assert score is None
