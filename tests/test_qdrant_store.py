import pytest

from ragcore.vectorstores.base import VectorStore


async def test_qdrant_roundtrip(tmp_path):
    pytest.importorskip("qdrant_client")
    from ragcore.vectorstores.qdrant_store import QdrantStore
    store = QdrantStore(path=str(tmp_path / "qd"), collection="t")
    assert isinstance(store, VectorStore)
    chunks = [
        {"id": "a", "text": "reciprocal rank fusion", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "weather report", "embedding": [0.0, 1.0, 0.0]},
    ]
    await store.add_embeddings("src1", chunks)
    res = await store.vector_search([1.0, 0.0, 0.0], k=1)
    assert res and res[0]["id"] == "a"
    assert set(res[0]) >= {"id", "text", "score"}
