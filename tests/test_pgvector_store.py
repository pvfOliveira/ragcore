"""Deterministic tests for PgVectorStore: no live Postgres — a fake connection
captures SQL + params. The live round-trip is tests/live/test_pgvector.py."""
import pytest

from ragcore.vectorstores.pgvector_store import PgVectorStore

pytest.importorskip("pgvector")


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class FakeConn:
    """Captures execute() calls; returns queued rows for SELECTs."""
    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows or []

    async def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return FakeCursor(self._rows)


def _store_with(conn):
    store = PgVectorStore(dsn="postgresql://localhost/unused", collection="ragcore")
    store._conn = conn          # bypass _get_conn (no real connection)
    return store


async def test_add_embeddings_creates_table_and_hnsw_index_then_upserts():
    conn = FakeConn()
    store = _store_with(conn)
    await store.add_embeddings("src1", [
        {"id": "c1", "text": "alpha", "embedding": [0.1, 0.2]},
        {"id": "c2", "text": "beta", "embedding": [0.3, 0.4]},
    ])
    sql = " ".join(s for s, _ in conn.calls)
    assert "CREATE TABLE IF NOT EXISTS ragcore" in sql
    assert "vector(2)" in sql                       # dim from first chunk
    assert "USING hnsw" in sql and "vector_cosine_ops" in sql
    inserts = [c for c in conn.calls if c[0].lstrip().startswith("INSERT")]
    assert len(inserts) == 2
    assert "ON CONFLICT (chunk_id) DO UPDATE" in inserts[0][0]
    assert inserts[0][1][1] == "c1" and inserts[0][1][2] == "src1"


async def test_vector_search_returns_ragcore_shape_ordered():
    conn = FakeConn(rows=[("c2", "beta", 0.97), ("c1", "alpha", 0.61)])
    store = _store_with(conn)
    store._dim = 2                                  # table already ensured
    hits = await store.vector_search([0.3, 0.4], k=2)
    assert hits == [
        {"id": "c2", "text": "beta", "score": 0.97},
        {"id": "c1", "text": "alpha", "score": 0.61},
    ]
    select_sql = conn.calls[-1][0]
    assert "<=>" in select_sql and "LIMIT" in select_sql


async def test_empty_add_is_noop():
    conn = FakeConn()
    store = _store_with(conn)
    await store.add_embeddings("src1", [])
    assert conn.calls == []


def test_collection_name_is_validated():
    with pytest.raises(ValueError):
        PgVectorStore(dsn="postgresql://x/y", collection="bad-name; DROP TABLE--")


def test_protocol_conformance():
    from ragcore.vectorstores.base import VectorStore
    assert isinstance(PgVectorStore(dsn="postgresql://x/y"), VectorStore)


def test_factory_dispatch():
    from ragcore.config import StoreConfig
    from ragcore.vectorstores.base import make_vector_store

    class Cfg:
        store = StoreConfig(vector_backend="pgvector",
                            pgvector_dsn="postgresql://localhost/ragcore")

    store = make_vector_store(Cfg())
    assert isinstance(store, PgVectorStore)
    assert store._dsn == "postgresql://localhost/ragcore"
