from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from contextd.config import get_settings
from contextd.ingest.embedder import Embedder, default_embedder
from contextd.logging_ import get_logger
from contextd.storage.db import (
    get_source_by_path,
    insert_chunk,
    insert_corpus,
    insert_source,
    open_db,
)
from contextd.storage.vectors import VectorStore

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable
    from pathlib import Path

    from contextd.ingest.protocol import Adapter, ChunkDraft, SourceCandidate

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestReport:
    sources_ingested: int = 0
    sources_skipped: int = 0
    sources_failed: int = 0
    chunks_written: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        adapters: Iterable[Adapter] | None = None,
    ) -> None:
        self._embedder = embedder or default_embedder()
        if adapters is not None:
            self._adapters = list(adapters)
        else:
            from contextd.ingest.adapters import load_default_adapters

            self._adapters = list(load_default_adapters())

    def _select_adapter(self, path: Path, source_type: str | None) -> Adapter:
        if source_type:
            for a in self._adapters:
                if a.source_type == source_type:
                    return a
            raise ValueError(f"no adapter with source_type={source_type!r}")
        for a in self._adapters:
            if a.can_handle(path):
                return a
        raise ValueError(f"no adapter handles path={path}")

    def ingest(
        self,
        *,
        path: Path,
        corpus: str,
        source_type: str | None = None,
        force: bool = False,
    ) -> IngestReport:
        adapter = self._select_adapter(path, source_type)
        conn = open_db(corpus)
        insert_corpus(
            conn,
            name=corpus,
            embed_model=self._embedder.model_name,
            embed_dim=self._embedder.dim,
            created_at=datetime.now(UTC),
            schema_version=get_settings().schema_version,
        )
        vs = VectorStore.open(
            corpus=corpus,
            embed_dim=self._embedder.dim,
            model_name=self._embedder.model_name,
        )

        ingested = skipped = failed = total_chunks = 0
        errors: list[str] = []

        for candidate in adapter.sources(path):
            existing = get_source_by_path(conn, corpus=corpus, path=str(candidate.path))
            if existing and existing.content_hash == candidate.content_hash and not force:
                skipped += 1
                continue
            try:
                n = self._write_source(conn, vs, adapter, candidate, corpus)
                ingested += 1
                total_chunks += n
            except Exception as e:
                failed += 1
                errors.append(f"{candidate.path}: {e!r}")
                log.error("ingest.source_failed", path=str(candidate.path), error=repr(e))

        conn.execute(
            "INSERT INTO audit_log(occurred_at, actor, action, target, details_json)"
            " VALUES (?, 'cli', 'ingest', ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                str(path),
                f'{{"ingested":{ingested},"skipped":{skipped}}}',
            ),
        )
        conn.commit()
        return IngestReport(
            sources_ingested=ingested,
            sources_skipped=skipped,
            sources_failed=failed,
            chunks_written=total_chunks,
            errors=tuple(errors),
        )

    def _write_source(
        self,
        conn: sqlite3.Connection,
        vs: VectorStore,
        adapter: Adapter,
        candidate: SourceCandidate,
        corpus: str,
    ) -> int:
        now = datetime.now(UTC)
        chunks: list[ChunkDraft] = list(adapter.parse(candidate))
        meta = adapter.metadata(candidate)
        conn.execute("BEGIN")
        try:
            source_id = insert_source(
                conn,
                corpus=corpus,
                source_type=candidate.source_type,
                path=str(candidate.path),
                content_hash=candidate.content_hash,
                ingested_at=now,
                chunk_count=len(chunks),
                status="active",
                title=candidate.title,
                source_mtime=candidate.source_mtime,
            )
            for k, v in meta.items():
                conn.execute(
                    "INSERT INTO source_meta(source_id, key, value) VALUES (?, ?, ?)",
                    (source_id, k, v),
                )
            ordinal_to_id: dict[int, int] = {}
            for ch in chunks:
                cid = insert_chunk(
                    conn,
                    source_id=source_id,
                    ordinal=ch.ordinal,
                    token_count=ch.token_count,
                    content=ch.content,
                    offset_start=ch.offset_start,
                    offset_end=ch.offset_end,
                    section_label=ch.section_label,
                    scope=ch.scope,
                    role=ch.role,
                    chunk_timestamp=ch.chunk_timestamp,
                )
                ordinal_to_id[ch.ordinal] = cid
                for k, v in ch.metadata.items():
                    conn.execute(
                        "INSERT INTO chunk_meta(chunk_id, key, value) VALUES (?, ?, ?)",
                        (cid, k, v),
                    )
            if chunks:
                vecs = self._embedder.embed([c.content for c in chunks])
                vs.upsert([ordinal_to_id[c.ordinal] for c in chunks], vecs)
            for e in adapter.edges(chunks):
                tgt = ordinal_to_id.get(e.target_ordinal) if e.target_ordinal is not None else None
                conn.execute(
                    "INSERT INTO edge(source_chunk_id, target_chunk_id, target_hint,"
                    " edge_type, label, weight) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        ordinal_to_id[e.source_ordinal],
                        tgt,
                        e.target_hint,
                        e.edge_type,
                        e.label,
                        e.weight,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return len(chunks)
