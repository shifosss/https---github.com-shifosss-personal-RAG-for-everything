from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING

from contextd.config import get_settings
from contextd.logging_ import get_logger
from contextd.retrieve.dense import dense_search
from contextd.retrieve.format import hydrate_results
from contextd.retrieve.fusion import reciprocal_rank_fusion
from contextd.retrieve.rerank import RerankUnavailable, rerank
from contextd.retrieve.rewrite import rewrite_query
from contextd.retrieve.sparse import sparse_search
from contextd.storage.db import fetch_chunks_by_ids, open_db
from contextd.storage.models import ChunkResult, QueryTrace

if TYPE_CHECKING:
    from contextd.retrieve.preprocess import QueryRequest

log = get_logger(__name__)


async def retrieve(req: QueryRequest) -> tuple[list[ChunkResult], QueryTrace]:
    s = get_settings()
    t0 = time.perf_counter()

    queries: list[str] = [req.query]
    rewriter_used: str | None = None
    if req.rewrite:
        rw = await rewrite_query(
            query=req.query,
            model=s.rewriter_model,
            timeout_ms=s.retrieval_rewrite_timeout_ms,
        )
        queries.extend(rw.sub_queries)
        rewriter_used = rw.rewriter_used

    tasks = []
    for q in queries:
        tasks.append(dense_search(query=q, corpus=req.corpus, k=s.retrieval_dense_top_k))
        tasks.append(sparse_search(query=q, corpus=req.corpus, k=s.retrieval_sparse_top_k))
    outs = await asyncio.gather(*tasks)

    per_query: list[tuple[list[tuple[int, float]], list[tuple[int, float]]]] = []
    dense_count = 0
    sparse_count = 0
    for i in range(0, len(outs), 2):
        dense, sparse = outs[i], outs[i + 1]
        dense_count += len(dense)
        sparse_count += len(sparse)
        per_query.append((dense, sparse))

    fused = reciprocal_rank_fusion(per_query, k=s.retrieval_rrf_k, top_n=s.retrieval_rerank_top_k)

    reranker_used: str | None = None
    if req.rerank and fused:
        chunk_map = {
            c.id: c for c in fetch_chunks_by_ids(open_db(req.corpus), [cid for cid, _ in fused])
        }
        ordered_candidates = [(cid, chunk_map[cid].content) for cid, _ in fused if cid in chunk_map]
        try:
            reranked = await rerank(
                query=req.query,
                candidates=ordered_candidates,
                model=s.reranker_model,
                timeout_ms=s.retrieval_rerank_timeout_ms,
            )
            rerank_ids = {cid for cid, _ in reranked}
            remainder = [(cid, score) for cid, score in fused if cid not in rerank_ids]
            fused = reranked + remainder
            reranker_used = s.reranker_model
        except RerankUnavailable as e:
            log.warning("rerank.unavailable", error=str(e), trace_id=req.trace_id)

    top = fused[: req.limit]
    results = hydrate_results(
        corpus=req.corpus,
        scored=[(cid, float(score)) for cid, score in top],
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    trace = QueryTrace(
        trace_id=req.trace_id,
        latency_ms=latency_ms,
        dense_candidates=dense_count,
        sparse_candidates=sparse_count,
        reranker_used=reranker_used,
        rewriter_used=rewriter_used,
    )

    qh = hashlib.sha256(req.query.encode()).hexdigest()[:16]
    conn = open_db(req.corpus)
    conn.execute(
        "INSERT INTO audit_log(occurred_at, actor, action, target, details_json) "
        "VALUES (datetime('now'), 'retrieve', 'query', ?, ?)",
        (
            f"query#{qh}",
            f'{{"trace_id":"{req.trace_id}","latency_ms":{latency_ms}}}',
        ),
    )
    conn.commit()
    return results, trace
