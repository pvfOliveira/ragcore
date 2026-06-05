import pytest

pytest.importorskip("faiss")
from ragcore.vectorstores.faiss_store import FaissStore


async def test_faiss_index_and_search(tmp_path):
    s = FaissStore(path=str(tmp_path / "faiss"))
    await s.add_embeddings("src1", [
        {"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "wind", "embedding": [0.0, 1.0, 0.0]},
        {"id": "c", "text": "grid", "embedding": [0.0, 0.0, 1.0]},
    ])
    out = await s.vector_search([0.9, 0.1, 0.0], k=1)
    assert out[0]["id"] == "a"
    assert out[0]["text"] == "solar"
    assert "score" in out[0]


async def test_faiss_persists(tmp_path):
    p = str(tmp_path / "faiss")
    s1 = FaissStore(path=p)
    await s1.add_embeddings("src1", [{"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]}])
    s2 = FaissStore(path=p)  # reopen
    out = await s2.vector_search([1.0, 0.0, 0.0], k=1)
    assert out[0]["id"] == "a"
