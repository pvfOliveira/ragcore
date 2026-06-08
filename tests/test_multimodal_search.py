import asyncio

from ragcore.cli import _image_search


def test_image_search_returns_ranked_paths():
    class _FakeImageStore:
        async def search(self, qv, k):
            return [{"id": "image_embedding:cat", "origin": "cat.png",
                     "path": "cat.png", "similarity": 0.91}]
    class _FakeEmb:
        def embed_text(self, t): return [1.0, 0.0]
    out = asyncio.run(_image_search("a cat", k=3, store=_FakeImageStore(), embedder=_FakeEmb()))
    assert out[0]["path"] == "cat.png"
    assert out[0]["similarity"] == 0.91
