"""LLM-as-judge scorer for retrieval quality.

PRD ref: §16.7 Phase 5 — aggregate judge score ≥ 6.5 is part of the
ship gate. Skips silently (returns None) when the API is unavailable,
so eval still runs on machines without ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from anthropic import Anthropic

from contextd.llm_json import parse_llm_json

_log = logging.getLogger("contextd.eval.judge")

_SYS = (
    "You are a retrieval-quality judge. Score how well the retrieved text "
    "answers the query. Scores 0-10: 10 = directly answers, 7-9 strongly "
    "on-topic, 4-6 tangential, 1-3 keyword overlap only, 0 irrelevant. "
    'Reply ONLY with JSON: {"score": <int>, "rationale": "<short>"}'
)

_MODEL = "claude-haiku-4-5"
_MAX_TOKENS = 200
_TIMEOUT_S = 10.0
_CONTENT_BUDGET = 2000


@lru_cache(maxsize=1)
def _anthropic_client() -> Anthropic:
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


async def judge_result(*, query: str, result_text: str) -> int | None:
    """Score a retrieval result on a 0-10 scale via Claude Haiku.

    Returns ``None`` on any failure (missing API key, network error,
    malformed response, timeout). Never raises — the eval loop is
    expected to tolerate missing scores and aggregate over what it has.
    """
    client = _anthropic_client()

    def _call() -> object:
        return client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
            system=_SYS,
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRetrieved:\n{result_text[:_CONTENT_BUDGET]}",
                }
            ],
        )

    try:
        res = await asyncio.wait_for(asyncio.to_thread(_call), timeout=_TIMEOUT_S)
        blocks = getattr(res, "content", None) or []
        # Duck-type: we only care about blocks exposing .text. Covers TextBlock in
        # real Anthropic responses and lets tests substitute lighter stubs.
        first = next((b for b in blocks if hasattr(b, "text")), None)
        if first is None:
            return None
        data = parse_llm_json(str(first.text))
        score = int(data.get("score", 0))
    except Exception:
        _log.debug("judge_result: scoring skipped", exc_info=True)
        return None
    return max(0, min(10, score))
