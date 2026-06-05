import pytest

from ragcore.vectorstores.base import VectorStore, make_vector_store


class _MemStore:
    def __init__(self):
        self.rows = []

    async def add_embeddings(self, source_id, chunks):
        for c in chunks:
            self.rows.append({**c, "source_id": source_id})

    async def vector_search(self, query, k=10):
        import math

        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = [{"id": r["id"], "text": r["text"], "score": cos(query, r["embedding"])} for r in self.rows]
        return sorted(scored, key=lambda r: r["score"], reverse=True)[:k]


async def test_protocol_roundtrip():
    s = _MemStore()
    await s.add_embeddings("src1", [{"id": "a", "text": "alpha", "embedding": [1.0, 0.0]},
                                    {"id": "b", "text": "beta", "embedding": [0.0, 1.0]}])
    out = await s.vector_search([0.9, 0.1], k=1)
    assert out[0]["id"] == "a"
    assert isinstance(s, VectorStore)  # runtime_checkable structural check


def test_factory_unknown_backend_raises():
    from types import SimpleNamespace
    cfg = SimpleNamespace(store=SimpleNamespace(vector_backend="nope"))
    with pytest.raises((ValueError, KeyError)):
        make_vector_store(cfg)
