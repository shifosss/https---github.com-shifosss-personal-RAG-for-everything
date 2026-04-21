"""Integration tests for ClaudeExportAdapter.

Fixture: tests/fixtures/claude/export.json
  - 3 conversations
  - 8 messages per conversation = 24 total non-empty messages
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contextd.ingest.adapters.claude_export import ClaudeExportAdapter

pytestmark = pytest.mark.integration

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "claude" / "export.json"


def test_sources_one_per_conversation() -> None:
    a = ClaudeExportAdapter()
    cands = list(a.sources(FIX))
    assert len(cands) >= 3
    assert all("#conversations/" in c.canonical_id for c in cands)


def test_parse_message_count_matches_source_meta() -> None:
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    meta = a.metadata(cand)
    assert chunks
    assert int(meta["message_count"]) == len(chunks)


def test_roles_are_user_or_assistant() -> None:
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    assert {c.role for c in chunks} <= {"user", "assistant"}


def test_edges_form_linked_list() -> None:
    a = ClaudeExportAdapter()
    cand = next(iter(a.sources(FIX)))
    chunks = list(a.parse(cand))
    edges = list(a.edges(chunks))
    assert len(edges) == 2 * (len(chunks) - 1)
    assert {e.edge_type for e in edges} == {"conversation_next", "conversation_prev"}
