"""Tolerant JSON parser for LLM responses.

Haiku 4.5 (and occasionally other Claude models) wraps JSON responses in
``` ... ``` markdown fences despite "reply ONLY with JSON" system prompts.
Raw ``json.loads`` fails on the leading backtick with ``Expecting value:
line 1 column 1 (char 0)``. This helper strips the fence then delegates
to ``json.loads``, raising the same ``JSONDecodeError`` on any failure so
callers keep their existing graceful-degradation paths.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(
    r"\A\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)


def parse_llm_json(text: str) -> Any:
    """Parse JSON emitted by an LLM, tolerating markdown code fences.

    Raises ``json.JSONDecodeError`` on any parse failure (including empty
    input or malformed content inside a fence).
    """
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        stripped = m.group(1).strip()
    return json.loads(stripped)
