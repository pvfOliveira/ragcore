"""hybrid_search fuses a pluggable VectorStore's vector ranking with the
store's BM25 (text_search) ranking via RRF. Offline fakes only."""
from ragcore.retrieve import hybrid_search


class _FakeVec:
    """Pluggable VectorStore: adapter-shaped results {"id","text","score"}."""

    async def vector_search(self, query, k=10):
        return [
            {"id": "a", "text": "alpha", "score": 0.9},
            {"id": "b", "text": "beta", "score": 0.5},
        ]

    async def add_embeddings(self, *a):
        ...


class _FakeStore:
    """SurrealDB-shaped store: provides BM25 via text_search; its own
    vector_search must NOT be consulted when a vector_store is supplied."""

    async def text_search(self, query, k=10):
        return [{"id": "b", "content": "beta"}, {"id": "c", "content": "gamma"}]

    async def vector_search(self, query, k=10):
        raise AssertionError("store.vector_search must not run when vector_store is given")


async def _embedder(q):
    return [0.1, 0.2]


async def test_hybrid_uses_pluggable_vector_backend():
    out = await hybrid_search(
        _FakeStore(), _embedder, "q", k=3, vector_store=_FakeVec()
    )
    ids = [r["id"] for r in out]
    # vector ranking (a, b) fused with BM25 (b, c); b appears in both -> top.
    assert "a" in ids and "b" in ids and "c" in ids
    assert out[0]["id"] == "b"
    # adapter's {"text"} is bridged to ragcore's {"content"}.
    by_id = {r["id"]: r for r in out}
    assert by_id["a"]["content"] == "alpha"


async def test_hybrid_without_vector_store_uses_store_vector_search():
    class _Store:
        async def text_search(self, query, k=10):
            return [{"id": "b", "content": "beta"}]

        async def vector_search(self, query, k=10):
            return [{"id": "a", "content": "alpha"}]

    out = await hybrid_search(_Store(), _embedder, "q", k=3)
    ids = [r["id"] for r in out]
    assert "a" in ids and "b" in ids
