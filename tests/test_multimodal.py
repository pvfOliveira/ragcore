import pytest

from ragcore.multimodal import ClipEmbedder, cross_modal_rank


def test_cross_modal_rank_picks_nearest():
    images = [{"id": "img:cat", "path": "cat.png", "embedding": [1.0, 0.0]},
              {"id": "img:car", "path": "car.png", "embedding": [0.0, 1.0]}]
    out = cross_modal_rank([0.95, 0.05], images, k=1)
    assert out[0]["id"] == "img:cat"
    assert "score" in out[0]


class _FakeClip:
    def encode_text(self, text): return [1.0, 0.0]
    def encode_image(self, path): return [1.0, 0.0]


def test_embedder_uses_injected_model():
    emb = ClipEmbedder(_model=_FakeClip())
    assert emb.embed_text("a cat") == [1.0, 0.0]
    assert emb.embed_image("cat.png") == [1.0, 0.0]


@pytest.fixture()
async def surreal_config(surreal_url):
    from ragcore.config import SurrealConfig
    from ragcore.store import Store
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()
    yield cfg


async def test_image_store_add_and_search(surreal_config):
    from ragcore.multimodal import ImageStore
    store = ImageStore(surreal_config)
    await store.add_image("cat.png", "cat.png", [1.0, 0.0, 0.0])
    await store.add_image("car.png", "car.png", [0.0, 1.0, 0.0])
    hits = await store.search([0.9, 0.1, 0.0], k=1)
    assert hits[0]["path"] == "cat.png"
    assert "similarity" in hits[0]
