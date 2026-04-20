"""Per-corpus LanceDB vector store wrapper.

Stores chunk_id -> embedding vector in a LanceDB table at
``$CONTEXTD_HOME/corpora/<corpus>/vectors.lance``. Cosine metric.

PRD refs: §13.3 (LanceDB 0.17), §15 (dense retrieval).
"""

from __future__ import annotations

from typing import Any, Self

import lancedb  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from contextd.config import get_settings

_TABLE_NAME = "embedding"


class VectorStore:
    """Per-corpus LanceDB table of chunk_id -> float32[embed_dim] vectors.

    Schema: ``chunk_id INT64``, ``vector FLOAT32[embed_dim]``, ``model_name STRING``.
    Metric: cosine. LanceDB 0.17 has no native upsert, so :meth:`upsert` performs
    delete-then-add atomically per call.
    """

    def __init__(
        self,
        db: Any,
        table: Any,
        embed_dim: int,
        model_name: str,
    ) -> None:
        self._db = db
        self._table = table
        self._embed_dim = embed_dim
        self._model_name = model_name

    @classmethod
    def open(cls, *, corpus: str, embed_dim: int, model_name: str) -> Self:
        root = get_settings().data_root / "corpora" / corpus
        root.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(root / "vectors.lance"))
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), embed_dim)),
                pa.field("model_name", pa.string()),
            ]
        )
        if _TABLE_NAME in db.table_names():
            table = db.open_table(_TABLE_NAME)
        else:
            table = db.create_table(_TABLE_NAME, schema=schema, mode="create")
        return cls(db, table, embed_dim, model_name)

    def upsert(self, chunk_ids: list[int], vectors: np.ndarray) -> None:
        """Replace rows for ``chunk_ids`` with the given vectors.

        Raises:
            ValueError: if ``vectors.shape != (len(chunk_ids), embed_dim)``.
        """
        if not chunk_ids:
            return
        expected = (len(chunk_ids), self._embed_dim)
        if vectors.shape != expected:
            raise ValueError(
                f"vectors shape {vectors.shape} != {expected}",
            )
        ids_sql = ", ".join(str(int(i)) for i in chunk_ids)
        self._table.delete(f"chunk_id IN ({ids_sql})")
        vecs_f32 = vectors.astype(np.float32, copy=False)
        records = [
            {
                "chunk_id": int(cid),
                "vector": vecs_f32[i].tolist(),
                "model_name": self._model_name,
            }
            for i, cid in enumerate(chunk_ids)
        ]
        self._table.add(records)

    def delete(self, chunk_ids: list[int]) -> None:
        """Remove rows for the given chunk ids. No-op on empty input."""
        if not chunk_ids:
            return
        ids_sql = ", ".join(str(int(i)) for i in chunk_ids)
        self._table.delete(f"chunk_id IN ({ids_sql})")

    def ann_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return top-k (chunk_id, cosine_distance) for ``query_vec``.

        Returns an empty list when the table has no rows. Cosine distance is
        in ``[0, 2]``; lower is more similar.
        """
        if self._table.count_rows() == 0:
            return []
        query_f32 = query_vec.astype(np.float32, copy=False).tolist()
        result = self._table.search(query_f32).metric("cosine").limit(k).to_list()
        return [(int(r["chunk_id"]), float(r["_distance"])) for r in result]
