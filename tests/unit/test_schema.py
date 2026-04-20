from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from contextd.storage.schema import apply_schema

if TYPE_CHECKING:
    from pathlib import Path


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
    return {r[0] for r in rows}


def test_apply_schema_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    apply_schema(conn)
    names = _tables(conn)
    for t in (
        "corpus",
        "source",
        "chunk",
        "chunk_fts",
        "edge",
        "chunk_meta",
        "source_meta",
        "audit_log",
    ):
        assert t in names, f"missing table: {t}"


def test_apply_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    apply_schema(conn)
    apply_schema(conn)  # must not raise
    assert "chunk" in _tables(conn)


def test_fts_trigger_populates_on_insert(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    apply_schema(conn)
    conn.execute(
        "INSERT INTO corpus VALUES ('personal', NULL, 'BAAI/bge-m3', 1024, "
        "'2026-04-20T00:00:00', 1)"
    )
    conn.execute(
        "INSERT INTO source(corpus, source_type, path, content_hash, ingested_at, "
        "chunk_count, status) VALUES "
        "('personal', 'pdf', '/a.pdf', 'sha256:x', '2026-04-20T00:00:00', 1, 'active')"
    )
    sid = conn.execute("SELECT id FROM source").fetchone()[0]
    conn.execute(
        "INSERT INTO chunk(source_id, ordinal, token_count, content) "
        "VALUES (?, 0, 2, 'negation handling')",
        (sid,),
    )
    conn.commit()
    hits = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'negation'").fetchall()
    assert len(hits) == 1
