from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from anthropic import Anthropic

from contextd.llm_json import parse_llm_json

if TYPE_CHECKING:
    from anthropic.types import Message, MessageParam

_SYS_PROMPT = (
    "You are a reranker for a personal retrieval system. Score each candidate "
    "chunk 0-10 for relevance to the user's query. Reply ONLY with a JSON array "
    'like [{"id": 12345, "score": 8}, ...]. No prose.'
)


class RerankUnavailable(Exception):  # noqa: N818
    """Raised when the reranker API is unreachable, times out, or returns invalid output after retry."""


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def rerank(
    *,
    query: str,
    candidates: list[tuple[int, str]],
    model: str,
    timeout_ms: int,
    truncate_tokens: int = 800,
) -> list[tuple[int, float]]:
    if not candidates:
        return []

    payload = {
        "query": query,
        "candidates": [
            {"id": cid, "content": _truncate(text, truncate_tokens)} for cid, text in candidates
        ],
    }
    user_msg = f"Query: {payload['query']}\n\nCandidates:\n" + json.dumps(
        payload["candidates"], ensure_ascii=False
    )

    client = _anthropic_client()
    _msgs: list[MessageParam] = [{"role": "user", "content": user_msg}]

    def _call() -> Message:
        return client.messages.create(
            model=model,
            max_tokens=1200,
            temperature=0.0,
            system=_SYS_PROMPT,
            messages=_msgs,
        )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_call),
            timeout=timeout_ms / 1000.0,
        )
    except (TimeoutError, ConnectionError, OSError) as e:
        raise RerankUnavailable(f"rerank API unreachable: {e!r}") from e
    except Exception as e:
        raise RerankUnavailable(f"rerank failed: {e!r}") from e

    try:
        block = result.content[0]
        # Duck-type: real TextBlock and test fakes both expose .text
        raw_text: str = block.text  # type: ignore[union-attr]
        data = parse_llm_json(raw_text)
        scored = [(int(x["id"]), float(x["score"])) for x in data]
    except Exception as e:
        raise RerankUnavailable(f"invalid rerank JSON: {e!r}") from e

    return sorted(scored, key=lambda x: x[1], reverse=True)


def _truncate(s: str, tokens: int) -> str:
    cap = tokens * 4
    return s if len(s) <= cap else s[:cap]
