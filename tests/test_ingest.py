import pytest

from ragcore import ingest as ingest_mod
from ragcore.ingest import ingest_source


class FakeStore:
    def __init__(self):
        self.sources, self.embeddings = [], []

    async def create_source(self, title, full_text, origin):
        sid = f"source:{len(self.sources)}"
        self.sources.append({"id": sid, "title": title, "full_text": full_text})
        return sid

    async def add_embeddings(self, source_id, chunks):
        self.embeddings.extend(chunks)


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    async def fake_extract(path_or_url):
        return {"title": "Sample", "content": "alpha beta. " * 300, "origin": path_or_url}
    async def fake_embeddings(texts, config, batch_size=50):
        return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(ingest_mod, "_extract", fake_extract)
    monkeypatch.setattr(ingest_mod, "generate_embeddings", fake_embeddings)


async def test_ingest_creates_source_and_chunks():
    store = FakeStore()
    sid = await ingest_source("doc.txt", store=store, config=None, chunk_size=50)
    assert sid == "source:0"
    assert len(store.embeddings) >= 1
    assert store.embeddings[0]["order"] == 0
    assert "content" in store.embeddings[0] and "embedding" in store.embeddings[0]
