from __future__ import annotations

from contextd.storage.db import fetch_chunks_by_ids, open_db, row_to_source
from contextd.storage.models import ChunkResult, Edge


def hydrate_results(*, corpus: str, scored: list[tuple[int, float]]) -> list[ChunkResult]:
    """Join chunk IDs + scores with source, chunk_meta, and edge rows.

    Returns ChunkResult list in the same order as ``scored``, skipping any
    chunk ID that no longer exists in the database.
    """
    if not scored:
        return []

    conn = open_db(corpus)
    ids = [cid for cid, _ in scored]
    chunks = {c.id: c for c in fetch_chunks_by_ids(conn, ids)}
    if not chunks:
        return []

    q_placeholders = ",".join("?" for _ in chunks)

    src_rows = conn.execute(
        f"SELECT DISTINCT s.* FROM source s "
        f"JOIN chunk c ON c.source_id = s.id "
        f"WHERE c.id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    sources_by_id = {r["id"]: row_to_source(r) for r in src_rows}

    meta_rows = conn.execute(
        f"SELECT chunk_id, key, value FROM chunk_meta WHERE chunk_id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    meta_by_chunk: dict[int, dict[str, str]] = {}
    for r in meta_rows:
        meta_by_chunk.setdefault(int(r["chunk_id"]), {})[r["key"]] = r["value"]

    edge_rows = conn.execute(
        f"SELECT * FROM edge WHERE source_chunk_id IN ({q_placeholders})",
        list(chunks.keys()),
    ).fetchall()
    edges_by_chunk: dict[int, list[Edge]] = {}
    for r in edge_rows:
        edges_by_chunk.setdefault(int(r["source_chunk_id"]), []).append(
            Edge(
                id=r["id"],
                source_chunk_id=r["source_chunk_id"],
                edge_type=r["edge_type"],
                target_chunk_id=r["target_chunk_id"],
                target_hint=r["target_hint"],
                label=r["label"],
                weight=r["weight"],
            )
        )

    out: list[ChunkResult] = []
    for rank_idx, (cid, score) in enumerate(scored, start=1):
        c = chunks.get(cid)
        if c is None:
            continue
        source = sources_by_id.get(c.source_id)
        if source is None:
            # Chunk row points at a source that was deleted or never written
            # (e.g. a LanceDB orphan vector from a failed ingest that landed
            # before the compensating-delete fix). Skip rather than crash.
            continue
        out.append(
            ChunkResult(
                chunk=c,
                source=source,
                score=score,
                rank=rank_idx,
                metadata=meta_by_chunk.get(cid, {}),
                edges=tuple(edges_by_chunk.get(cid, ())),
            )
        )
    return out
