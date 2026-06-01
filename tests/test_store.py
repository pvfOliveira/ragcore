import pytest

from ragcore.config import SurrealConfig
from ragcore.store import Store


@pytest.fixture()
async def store(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    s = Store(cfg)
    await s.init_schema()
    yield s


async def test_create_source_and_search(store):
    src_id = await store.create_source(title="Doc", full_text="full", origin="test")
    assert src_id.startswith("source:")
    await store.add_embeddings(src_id, [
        {"order": 0, "content": "the cat sat on the mat", "embedding": [1.0, 0.0]},
        {"order": 1, "content": "dogs run in the park", "embedding": [0.0, 1.0]},
    ])

    vres = await store.vector_search([1.0, 0.0], k=2)
    assert vres[0]["content"] == "the cat sat on the mat"
    assert vres[0]["similarity"] > vres[1]["similarity"]

    tres = await store.text_search("dogs", k=5)
    assert any("dogs" in r["content"] for r in tres)
