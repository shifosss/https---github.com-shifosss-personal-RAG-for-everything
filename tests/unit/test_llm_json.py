"""Unit tests for the LLM JSON parsing helper.

Background: Haiku 4.5 (and other Claude models) sometimes wrap JSON in
```json ... ``` markdown fences despite "reply ONLY with JSON" system
prompts. Raw ``json.loads`` chokes on the leading backtick. This helper
strips the fence before parsing.
"""

from __future__ import annotations

import json

import pytest

from contextd.llm_json import parse_llm_json


def test_plain_json_array_parses() -> None:
    assert parse_llm_json('[{"id": 1, "score": 9}]') == [{"id": 1, "score": 9}]


def test_plain_json_object_parses() -> None:
    assert parse_llm_json('{"score": 7}') == {"score": 7}


def test_fenced_json_array_with_language_tag_parses() -> None:
    payload = '```json\n[{"id": 1, "score": 9}]\n```'
    assert parse_llm_json(payload) == [{"id": 1, "score": 9}]


def test_fenced_json_object_with_language_tag_parses() -> None:
    payload = '```json\n{"score": 7, "rationale": "on topic"}\n```'
    assert parse_llm_json(payload) == {"score": 7, "rationale": "on topic"}


def test_fenced_json_without_language_tag_parses() -> None:
    payload = "```\n[1, 2, 3]\n```"
    assert parse_llm_json(payload) == [1, 2, 3]


def test_fenced_json_with_leading_and_trailing_whitespace_parses() -> None:
    payload = "   \n```json\n[42]\n```  \n"
    assert parse_llm_json(payload) == [42]


def test_fenced_json_case_insensitive_language_tag() -> None:
    payload = "```JSON\n[1]\n```"
    assert parse_llm_json(payload) == [1]


def test_empty_string_raises_json_decode_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("")


def test_invalid_json_raises_json_decode_error() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("not json at all")


def test_fenced_but_malformed_inner_json_raises() -> None:
    # The fence-strip succeeds, but the inner content isn't JSON.
    # Should still propagate JSONDecodeError so callers can handle it.
    payload = "```json\nnot really json\n```"
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json(payload)
