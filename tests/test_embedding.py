import pytest

from ragcore import embedding as emb_mod
from ragcore.embedding import generate_embedding, generate_embeddings


class FakeEmbedder:
    async def aembed(self, texts):
        # Deterministic: vector = [len(text), word count]
        return [[float(len(t)), float(len(t.split()))] for t in texts]


@pytest.fixture(autouse=True)
def patch_embedder(monkeypatch):
    monkeypatch.setattr(emb_mod, "_get_embedder", lambda cfg: FakeEmbedder())


async def test_generate_embeddings_batches():
    vecs = await generate_embeddings(["a", "bb", "ccc"], config=None, batch_size=2)
    assert len(vecs) == 3
    assert vecs[2] == [3.0, 1.0]


async def test_generate_embedding_meanpools_long_text():
    long = "word " * 5000
    vec = await generate_embedding(long, config=None, chunk_size=50)
    assert len(vec) == 2  # single pooled vector
