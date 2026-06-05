import pytest

pytest.importorskip("chromadb")
from ragcore.vectorstores.chroma_store import ChromaStore


async def test_chroma_index_and_search(tmp_path):
    s = ChromaStore(path=str(tmp_path / "chroma"))
    await s.add_embeddings("src1", [
        {"id": "a", "text": "solar power", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "wind power", "embedding": [0.0, 1.0, 0.0]},
        {"id": "c", "text": "grid", "embedding": [0.0, 0.0, 1.0]},
    ])
    out = await s.vector_search([0.95, 0.05, 0.0], k=1)
    assert out[0]["id"] == "a"
    assert out[0]["text"] == "solar power"
    assert "score" in out[0]


async def test_chroma_persists(tmp_path):
    p = str(tmp_path / "chroma")
    s1 = ChromaStore(path=p)
    await s1.add_embeddings("src1", [{"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]}])
    s2 = ChromaStore(path=p)  # reopen same path
    out = await s2.vector_search([1.0, 0.0, 0.0], k=1)
    assert out[0]["id"] == "a"
