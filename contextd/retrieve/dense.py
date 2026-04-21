"""Dense (vector) retrieval via LanceDB ANN search.

PRD refs: §15 (dense retrieval), §13.3 (LanceDB 0.17).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from contextd.ingest.embedder import default_embedder
from contextd.storage.vectors import VectorStore

if TYPE_CHECKING:
    from contextd.ingest.embedder import Embedder


async def dense_search(
    *,
    query: str,
    corpus: str,
    k: int,
    embedder: Embedder | None = None,
) -> list[tuple[int, float]]:
    """Return top-k (chunk_id, similarity) pairs via cosine ANN.

    ``VectorStore.ann_search`` returns cosine *distance* in [0, 2] (lower =
    more similar). This function converts to similarity = 1 - distance so that
    higher scores indicate better matches, consistent with the rest of the
    retrieval pipeline.

    Args:
        query: Raw query string to embed and search.
        corpus: Corpus name — determines which LanceDB table to query.
        k: Number of nearest neighbours to return.
        embedder: Optional embedder override; defaults to ``default_embedder()``.

    Returns:
        List of ``(chunk_id, similarity)`` sorted by similarity descending.
    """
    emb = embedder or default_embedder()
    vec = await asyncio.to_thread(lambda: emb.embed([query])[0])
    vs = VectorStore.open(corpus=corpus, embed_dim=emb.dim, model_name=emb.model_name)
    # ann_search returns (chunk_id, cosine_distance); convert to similarity.
    raw: list[tuple[int, float]] = await asyncio.to_thread(vs.ann_search, vec, k)
    return [(cid, 1.0 - dist) for cid, dist in raw]
