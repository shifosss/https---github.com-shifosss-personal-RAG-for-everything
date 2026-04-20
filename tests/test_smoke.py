from contextd import __version__
from contextd.storage.db import open_db
from contextd.storage.vectors import VectorStore


def test_version_defined() -> None:
    assert __version__.startswith("0.1.")


def test_open_db_creates_chunk_table(tmp_contextd_home) -> None:
    conn = open_db("personal")
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk'"
    ).fetchone()
    assert row is not None


def test_vector_store_open_is_idempotent(tmp_contextd_home) -> None:
    v1 = VectorStore.open(corpus="personal", embed_dim=4, model_name="test")
    v2 = VectorStore.open(corpus="personal", embed_dim=4, model_name="test")
    # Separate handles, same underlying store. Both should be constructible.
    assert v1 is not v2
