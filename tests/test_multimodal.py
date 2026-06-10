import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# ingest_image: VLM caption uses real embedder (not a placeholder)
# ---------------------------------------------------------------------------

_FAKE_CAPTION_EMBEDDING = [0.1, 0.2, 0.3]  # 3-dim stand-in for nomic-embed-text


@pytest.mark.asyncio
async def test_ingest_image_caption_uses_real_embedder(tmp_path, monkeypatch):
    """ingest_image must call generate_embedding for VLM captions.

    When vlm_enabled=True the caption chunk stored via add_embeddings must carry
    the real embedder output (multi-element), never the 1-dim [0.0] placeholder
    that would poison corpus-wide vector_search with a dimension mismatch.
    """
    import ragcore.multimodal as mm_mod
    import ragcore.docai as docai_mod

    # --- Fake config --------------------------------------------------------
    class _MM:
        vlm_enabled = True
        vlm_model = "moondream"
        model = None
        pretrained = None
        device = "cpu"

    class _Cfg:
        multimodal = _MM()
        surreal = MagicMock()

    cfg = _Cfg()

    # --- Fake image file (CLIP will be monkeypatched away) ------------------
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    # --- Monkeypatch VLM caption --------------------------------------------
    monkeypatch.setattr(docai_mod, "_ollama_caption",
                        lambda model, path: "a diagram of system architecture")

    # --- Monkeypatch CLIP embedder so we don't load torch -------------------
    fake_clip_emb = [0.5, 0.5]
    monkeypatch.setattr(mm_mod.ClipEmbedder, "embed_image",
                        lambda self, path: fake_clip_emb)

    # --- Monkeypatch generate_embedding to return known 3-dim vector --------
    async def _fake_generate_embedding(text, config, **kwargs):
        return _FAKE_CAPTION_EMBEDDING

    monkeypatch.setattr("ragcore.multimodal.generate_embedding",
                        _fake_generate_embedding, raising=False)
    # Also patch it in embedding module in case already imported
    import ragcore.embedding as emb_mod
    monkeypatch.setattr(emb_mod, "generate_embedding", _fake_generate_embedding)

    # --- Fake ImageStore (no SurrealDB) -------------------------------------
    fake_image_store = AsyncMock()
    fake_image_store.find_by_origin = AsyncMock(return_value=None)
    fake_image_store.add_image = AsyncMock(return_value="image_embedding:abc123")

    # --- Fake text Store (no SurrealDB) -------------------------------------
    fake_text_store = AsyncMock()
    fake_text_store.create_source = AsyncMock(return_value="source:xyz")
    add_embeddings_calls = []

    async def _capture_add_embeddings(source_id, rows):
        add_embeddings_calls.append((source_id, rows))

    fake_text_store.add_embeddings = _capture_add_embeddings

    monkeypatch.setattr(mm_mod, "ImageStore", lambda cfg: fake_image_store)

    import ragcore.store as store_mod
    monkeypatch.setattr(store_mod, "Store", lambda cfg: fake_text_store)

    # --- Run ----------------------------------------------------------------
    image_id, created = await mm_mod.ingest_image(str(img), cfg)

    assert created is True
    assert len(add_embeddings_calls) == 1, "add_embeddings must be called once for the caption"
    _, rows = add_embeddings_calls[0]
    assert len(rows) == 1
    chunk = rows[0]

    # Dimension check: must NOT be the 1-dim placeholder
    assert chunk["embedding"] != [0.0], "Placeholder [0.0] embedding must not be used"
    assert len(chunk["embedding"]) > 1, (
        f"Caption embedding must be multi-dimensional, got {chunk['embedding']}"
    )
    assert chunk["embedding"] == _FAKE_CAPTION_EMBEDDING, (
        "Caption embedding must come from the real embedder"
    )
    assert chunk["order"] == 0
    assert chunk["content"]  # non-empty caption text


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


async def test_image_store_find_by_origin_and_dedup(surreal_config):
    from ragcore.multimodal import ImageStore
    store = ImageStore(surreal_config)
    assert await store.find_by_origin("cat.png") is None
    id1 = await store.add_image("cat.png", "cat.png", [1.0, 0.0, 0.0])
    found = await store.find_by_origin("cat.png")
    assert found == id1
    # the UNIQUE index must reject a duplicate origin insert
    import pytest as _pytest
    with _pytest.raises(Exception):
        await store.add_image("cat.png", "cat.png", [0.0, 1.0, 0.0])
