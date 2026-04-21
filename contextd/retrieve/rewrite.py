from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from anthropic import Anthropic

if TYPE_CHECKING:
    from anthropic.types import Message, MessageParam

_SYS = (
    "You are a query-expansion assistant. Given one user query, produce 3-5 "
    "alternative phrasings covering the semantic territory they might want. "
    'Respond ONLY with JSON: {"sub_queries": ["...", "..."]}'
)


@dataclass(frozen=True)
class RewrittenQueries:
    original: str
    sub_queries: list[str]
    rewriter_used: str | None


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def rewrite_query(*, query: str, model: str, timeout_ms: int) -> RewrittenQueries:
    """Expand *query* into sub-queries via LLM.

    Disabled-by-default: callers gate on config before calling this function.
    On any failure (network, timeout, bad JSON) returns empty sub_queries so
    the pipeline degrades gracefully (D-30).
    """
    client = _anthropic_client()
    _msgs: list[MessageParam] = [{"role": "user", "content": query}]

    def _call() -> Message:
        return client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.4,
            system=_SYS,
            messages=_msgs,
        )

    try:
        res = await asyncio.wait_for(
            asyncio.to_thread(_call),
            timeout=timeout_ms / 1000.0,
        )
        block = res.content[0]
        raw_text: str = block.text  # type: ignore[union-attr]
        data = json.loads(raw_text.strip())
        subs = [s.strip() for s in data.get("sub_queries", []) if isinstance(s, str) and s.strip()]
        seen: set[str] = {query}
        uniq: list[str] = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return RewrittenQueries(original=query, sub_queries=uniq[:5], rewriter_used=model)
    except Exception:
        return RewrittenQueries(original=query, sub_queries=[], rewriter_used=None)
