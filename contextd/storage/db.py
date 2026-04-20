from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from contextd.config import get_settings
from contextd.storage.models import Chunk, Source, SourceStatus, SourceType
from contextd.storage.schema import apply_schema

if TYPE_CHECKING:
    from pathlib import Path


def _corpus_db_path(corpus: str) -> Path:
    settings = get_settings()
    p = settings.data_root / "corpora" / corpus
    p.mkdir(parents=True, exist_ok=True)
    return p / "chunks.db"


def open_db(corpus: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(_corpus_db_path(corpus)), isolation_level=None)
    conn.row_factory = sqlite3.Row
    apply_schema(conn)
    return conn


def insert_corpus(
    conn: sqlite3.Connection,
    *,
    name: str,
    embed_model: str,
    embed_dim: int,
    created_at: datetime,
    schema_version: int,
    root_path: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO corpus(name, root_path, embed_model, embed_dim, "
        "created_at, schema_version) VALUES (?, ?, ?, ?, ?, ?)",
        (name, root_path, embed_model, embed_dim, created_at.isoformat(), schema_version),
    )


def insert_source(
    conn: sqlite3.Connection,
    *,
    corpus: str,
    source_type: SourceType,
    path: str,
    content_hash: str,
    ingested_at: datetime,
    chunk_count: int,
    status: SourceStatus,
    title: str | None = None,
    source_mtime: datetime | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO source(corpus, source_type, path, content_hash, title, "
        "ingested_at, source_mtime, chunk_count, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            corpus,
            source_type,
            path,
            content_hash,
            title,
            ingested_at.isoformat(),
            source_mtime.isoformat() if source_mtime else None,
            chunk_count,
            status,
        ),
    )
    return int(cur.lastrowid or 0)


def get_source_by_path(
    conn: sqlite3.Connection,
    *,
    corpus: str,
    path: str,
) -> Source | None:
    row = conn.execute(
        "SELECT * FROM source WHERE corpus = ? AND path = ?",
        (corpus, path),
    ).fetchone()
    return row_to_source(row) if row else None


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    source_id: int,
    ordinal: int,
    token_count: int,
    content: str,
    offset_start: int | None = None,
    offset_end: int | None = None,
    section_label: str | None = None,
    scope: str | None = None,
    role: str | None = None,
    chunk_timestamp: datetime | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO chunk(source_id, ordinal, offset_start, offset_end, token_count, "
        "content, section_label, scope, role, chunk_timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            ordinal,
            offset_start,
            offset_end,
            token_count,
            content,
            section_label,
            scope,
            role,
            chunk_timestamp.isoformat() if chunk_timestamp else None,
        ),
    )
    return int(cur.lastrowid or 0)


def fetch_chunks_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[Chunk]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM chunk WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def row_to_source(r: sqlite3.Row) -> Source:
    return Source(
        id=r["id"],
        corpus=r["corpus"],
        source_type=r["source_type"],
        path=r["path"],
        content_hash=r["content_hash"],
        title=r["title"],
        ingested_at=datetime.fromisoformat(r["ingested_at"]),
        source_mtime=(datetime.fromisoformat(r["source_mtime"]) if r["source_mtime"] else None),
        chunk_count=r["chunk_count"],
        status=r["status"],
    )


def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        id=r["id"],
        source_id=r["source_id"],
        ordinal=r["ordinal"],
        offset_start=r["offset_start"],
        offset_end=r["offset_end"],
        token_count=r["token_count"],
        content=r["content"],
        section_label=r["section_label"],
        scope=r["scope"],
        role=r["role"],
        chunk_timestamp=(
            datetime.fromisoformat(r["chunk_timestamp"]) if r["chunk_timestamp"] else None
        ),
    )
