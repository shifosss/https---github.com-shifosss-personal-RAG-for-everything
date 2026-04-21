import pytest

from contextd.retrieve.rerank import RerankUnavailable, rerank

pytestmark = pytest.mark.integration


async def test_rerank_returns_ordered_ids_from_anthropic(monkeypatch):
    calls: list[dict] = []

    class FakeMessages:
        def create(self, *, model, max_tokens, temperature, system, messages, response_format=None):
            calls.append({"model": model, "messages": messages})

            class R:
                content = [
                    type("b", (), {"text": '[{"id": 2, "score": 10}, {"id": 1, "score": 5}]'})()
                ]

            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())

    chunks = [(1, "chunk one"), (2, "chunk two")]
    out = await rerank(query="q", candidates=chunks, model="claude-haiku-4-5", timeout_ms=5000)
    assert [c for c, _ in out] == [2, 1]


async def test_rerank_unavailable_raises_typed_error(monkeypatch):
    class FakeClient:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                raise ConnectionError("api down")

    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())
    with pytest.raises(RerankUnavailable):
        await rerank(query="q", candidates=[(1, "t")], model="m", timeout_ms=5000)


async def test_rerank_timeout_raises(monkeypatch):
    class SlowMessages:
        def create(self, **kw):
            import time

            time.sleep(2.0)

            class R:
                content = [type("b", (), {"text": "[]"})()]

            return R()

    class FakeClient:
        messages = SlowMessages()

    monkeypatch.setattr("contextd.retrieve.rerank._anthropic_client", lambda: FakeClient())
    with pytest.raises(RerankUnavailable):
        await rerank(query="q", candidates=[(1, "t")], model="m", timeout_ms=500)
