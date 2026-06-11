"""Deterministic tests for Qdrant hybrid sparse+dense.

Hybrid OFF (default) must be byte-identical to the v3 dense-only adapter:
unnamed vector schema, plain query_points. Hybrid ON builds named
dense+sparse vectors and a server-side RRF fusion query.

HONEST DISTINCTION (spec §3 wave 3): this is learned-sparse (BM42/SPLADE via
fastembed) + dense fused INSIDE Qdrant — distinct from retrieve.py's
client-side RRF over SurrealDB BM25 full-text.
"""
import pytest

pytest.importorskip("qdrant_client")

from ragcore.vectorstores.qdrant_store import QdrantStore


class FakeSparse:
    """Stands in for a fastembed SparseEmbedding (indices/values arrays)."""
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


def _sparse_batch_stub(texts):
    return [FakeSparse(indices=[7, 42], values=[0.5, 1.5]) for _ in texts]


def test_default_off_is_dense_only():
    store = QdrantStore(path="/tmp/q")
    assert store.supports_hybrid is False


def test_hybrid_flag_enables_capability():
    store = QdrantStore(path="/tmp/q", hybrid=True)
    assert store.supports_hybrid is True


async def test_hybrid_add_builds_named_dense_and_sparse_vectors(tmp_path):
    store = QdrantStore(path=str(tmp_path / "q"), collection="hyb", hybrid=True)
    store._encode_sparse_batch = _sparse_batch_stub
    await store.add_embeddings("src1", [
        {"id": "c1", "text": "alpha beta", "embedding": [0.1, 0.2]},
    ])
    client = store._get_client()
    points, _ = client.scroll("hyb", limit=1, with_vectors=True)
    vec = points[0].vector
    assert "dense" in vec and "sparse" in vec
    assert list(vec["sparse"].indices) == [7, 42]


async def test_hybrid_search_fuses_and_returns_ragcore_shape(tmp_path):
    store = QdrantStore(path=str(tmp_path / "q"), collection="hyb", hybrid=True)
    store._encode_sparse_batch = _sparse_batch_stub
    await store.add_embeddings("src1", [
        {"id": "c1", "text": "alpha", "embedding": [1.0, 0.0]},
        {"id": "c2", "text": "beta", "embedding": [0.0, 1.0]},
    ])
    hits = await store.hybrid_search("alpha", [1.0, 0.0], k=2)
    assert {h["id"] for h in hits} == {"c1", "c2"}
    assert set(hits[0]) == {"id", "text", "score"}


async def test_dense_only_schema_unchanged_when_off(tmp_path):
    """Regression: hybrid off keeps the v3 unnamed-vector schema + query path."""
    store = QdrantStore(path=str(tmp_path / "q"), collection="plain", hybrid=False)
    await store.add_embeddings("src1", [
        {"id": "c1", "text": "alpha", "embedding": [1.0, 0.0]},
    ])
    hits = await store.vector_search([1.0, 0.0], k=1)
    assert hits[0]["id"] == "c1"
    client = store._get_client()
    points, _ = client.scroll("plain", limit=1, with_vectors=True)
    assert isinstance(points[0].vector, list)    # unnamed single vector


async def test_schema_mismatch_dense_to_hybrid_raises(tmp_path):
    """Constructing a hybrid=True store over an existing dense-only collection
    must raise ValueError mentioning 'qdrant_hybrid'."""
    db = str(tmp_path / "q")
    # Create dense-only collection then release the file lock
    plain = QdrantStore(path=db, collection="col", hybrid=False)
    await plain.add_embeddings("src1", [
        {"id": "c1", "text": "alpha", "embedding": [1.0, 0.0]},
    ])
    plain.close()
    # Now open the same path+collection as hybrid=True — should fail fast
    hybrid = QdrantStore(path=db, collection="col", hybrid=True)
    hybrid._encode_sparse_batch = _sparse_batch_stub
    with pytest.raises(ValueError, match="qdrant_hybrid"):
        await hybrid.add_embeddings("src1", [
            {"id": "c2", "text": "beta", "embedding": [1.0, 0.0]},
        ])


async def test_schema_mismatch_hybrid_to_dense_raises(tmp_path):
    """Constructing a hybrid=False store over an existing hybrid collection
    must raise ValueError mentioning 'qdrant_hybrid'."""
    db = str(tmp_path / "q")
    # Create hybrid collection then release the file lock
    hybrid = QdrantStore(path=db, collection="col", hybrid=True)
    hybrid._encode_sparse_batch = _sparse_batch_stub
    await hybrid.add_embeddings("src1", [
        {"id": "c1", "text": "alpha", "embedding": [1.0, 0.0]},
    ])
    hybrid.close()
    # Now open the same path+collection as hybrid=False — should fail fast
    plain = QdrantStore(path=db, collection="col", hybrid=False)
    with pytest.raises(ValueError, match="qdrant_hybrid"):
        await plain.add_embeddings("src1", [
            {"id": "c2", "text": "beta", "embedding": [1.0, 0.0]},
        ])


async def test_retrieve_uses_hybrid_when_supported():
    """retrieve.hybrid_search routes text+vec to adapter.hybrid_search when
    supports_hybrid, else the v3 vector_search path."""
    from ragcore.retrieve import hybrid_search

    class Store:
        async def text_search(self, q, k):
            return []

    class HybridVS:
        supports_hybrid = True
        called = None

        async def hybrid_search(self, text, vec, k=10):
            HybridVS.called = (text, vec, k)
            return [{"id": "h1", "text": "t", "score": 1.0}]

    async def embedder(q):
        return [0.5, 0.5]

    out = await hybrid_search(Store(), embedder, "my query", k=3,
                              vector_store=HybridVS())
    assert HybridVS.called == ("my query", [0.5, 0.5], 3)
    assert out and out[0]["id"] == "h1"
