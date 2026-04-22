import pytest

from contextd.retrieve.rewrite import rewrite_query

pytestmark = pytest.mark.integration


async def test_rewrite_deduplicates_and_caps(monkeypatch):
    class FakeMessages:
        def create(self, **kw):
            class R:
                content = [
                    type(
                        "b",
                        (),
                        {"text": '{"sub_queries": ["a", "a", "b", "c", "d", "e", "f"]}'},
                    )()
                ]

            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("contextd.retrieve.rewrite._anthropic_client", lambda: FakeClient())
    out = await rewrite_query(query="orig", model="m", timeout_ms=3000)
    assert out.original == "orig"
    assert out.sub_queries[:1] == ["a"]
    assert len(set(["orig", *out.sub_queries])) <= 6


async def test_rewrite_parses_markdown_fenced_response(monkeypatch):
    """Regression: Haiku 4.5 wraps JSON in ```json ... ``` fences."""
    fenced = '```json\n{"sub_queries": ["alt one", "alt two"]}\n```'

    class FakeMessages:
        def create(self, **kw):
            class R:
                content = [type("b", (), {"text": fenced})()]

            return R()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("contextd.retrieve.rewrite._anthropic_client", lambda: FakeClient())
    out = await rewrite_query(query="orig", model="m", timeout_ms=3000)
    assert out.sub_queries == ["alt one", "alt two"]
    assert out.rewriter_used == "m"


async def test_rewrite_failure_returns_empty_subqueries(monkeypatch):
    class FakeMessages:
        def create(self, **kw):
            raise ConnectionError("down")

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("contextd.retrieve.rewrite._anthropic_client", lambda: FakeClient())
    out = await rewrite_query(query="orig", model="m", timeout_ms=3000)
    assert out.sub_queries == []
    assert out.rewriter_used is None
