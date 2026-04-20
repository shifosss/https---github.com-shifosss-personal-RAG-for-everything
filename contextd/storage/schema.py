"""SQLite schema DDL for contextd storage layer.

Authoritative DDL mirrors ``docs/plans/00-master-spec.md`` section "SQLite DDL".
Each element of :data:`DDL_STATEMENTS` is a single statement so it can be
executed individually — SQLite's ``Connection.execute`` rejects multi-statement
strings. Triggers keep the external-content FTS5 table ``chunk_fts`` in sync
with ``chunk`` on INSERT / DELETE / UPDATE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
)

DDL_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS corpus (
        name            TEXT PRIMARY KEY,
        root_path       TEXT,
        embed_model     TEXT NOT NULL,
        embed_dim       INTEGER NOT NULL,
        created_at      TEXT NOT NULL,
        schema_version  INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS source (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        corpus          TEXT NOT NULL REFERENCES corpus(name) ON DELETE CASCADE,
        source_type     TEXT NOT NULL,
        path            TEXT NOT NULL,
        content_hash    TEXT NOT NULL,
        title           TEXT,
        ingested_at     TEXT NOT NULL,
        source_mtime    TEXT,
        chunk_count     INTEGER NOT NULL DEFAULT 0,
        status          TEXT NOT NULL DEFAULT 'active',
        UNIQUE (corpus, path)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_source_type ON source(source_type)",
    "CREATE INDEX IF NOT EXISTS idx_source_hash ON source(content_hash)",
    """CREATE TABLE IF NOT EXISTS chunk (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id       INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
        ordinal         INTEGER NOT NULL,
        offset_start    INTEGER,
        offset_end      INTEGER,
        token_count     INTEGER NOT NULL,
        content         TEXT NOT NULL,
        section_label   TEXT,
        scope           TEXT,
        role            TEXT,
        chunk_timestamp TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_chunk_source_ordinal ON chunk(source_id, ordinal)",
    "CREATE INDEX IF NOT EXISTS idx_chunk_timestamp ON chunk(chunk_timestamp)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        content,
        content='chunk',
        content_rowid='id',
        tokenize='unicode61'
    )""",
    """CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunk BEGIN
        INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES ('delete', old.id, old.content);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunk_au AFTER UPDATE ON chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, content) VALUES ('delete', old.id, old.content);
        INSERT INTO chunk_fts(rowid, content) VALUES (new.id, new.content);
    END""",
    """CREATE TABLE IF NOT EXISTS edge (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        source_chunk_id   INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
        target_chunk_id   INTEGER REFERENCES chunk(id) ON DELETE CASCADE,
        target_hint       TEXT,
        edge_type         TEXT NOT NULL,
        label             TEXT,
        weight            REAL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_edge_src ON edge(source_chunk_id, edge_type)",
    "CREATE INDEX IF NOT EXISTS idx_edge_tgt ON edge(target_chunk_id, edge_type)",
    "CREATE INDEX IF NOT EXISTS idx_edge_hint ON edge(target_hint)",
    """CREATE TABLE IF NOT EXISTS chunk_meta (
        chunk_id    INTEGER NOT NULL REFERENCES chunk(id) ON DELETE CASCADE,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        PRIMARY KEY (chunk_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS source_meta (
        source_id   INTEGER NOT NULL REFERENCES source(id) ON DELETE CASCADE,
        key         TEXT NOT NULL,
        value       TEXT NOT NULL,
        PRIMARY KEY (source_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS audit_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_at   TEXT NOT NULL,
        actor         TEXT NOT NULL,
        action        TEXT NOT NULL,
        target        TEXT NOT NULL,
        details_json  TEXT NOT NULL DEFAULT '{}'
    )""",
)


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply PRAGMAs and DDL to ``conn``, committing at the end.

    Idempotent: every DDL uses ``IF NOT EXISTS`` so a second call is a no-op.
    """
    for pragma in PRAGMAS:
        conn.execute(pragma)
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)
    conn.commit()
