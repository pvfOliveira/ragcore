import asyncio

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


async def _seed(store):
    src1 = await store.create_source(title="Doc1", full_text="first", origin="origin-1")
    await store.add_embeddings(src1, [
        {"order": 0, "content": "alpha", "embedding": [1.0, 0.0]},
        {"order": 1, "content": "beta", "embedding": [0.0, 1.0]},
    ])
    # Guarantee a distinct, later creation timestamp so newest-first ordering is
    # deterministic (record ids are random, so they can't tie-break by recency).
    await asyncio.sleep(0.01)
    src2 = await store.create_source(title="Doc2", full_text="second", origin="origin-2")
    await store.add_embeddings(src2, [
        {"order": 0, "content": "gamma", "embedding": [1.0, 0.0]},
    ])
    return src1, src2


async def test_list_sources(store):
    src1, src2 = await _seed(store)
    sources = await store.list_sources()
    assert len(sources) == 2
    # Newest first.
    assert sources[0]["id"] == src2
    assert sources[1]["id"] == src1
    assert sources[0]["origin"] == "origin-2"
    assert sources[1]["origin"] == "origin-1"
    assert sources[0]["title"] == "Doc2"
    assert sources[0]["chunks"] == 1
    assert sources[1]["chunks"] == 2
    assert isinstance(sources[0]["created"], str)


async def test_find_source_id_by_origin(store):
    src1, src2 = await _seed(store)
    assert await store.find_source_id_by_origin("origin-1") == src1
    assert await store.find_source_id_by_origin("origin-2") == src2
    assert await store.find_source_id_by_origin("unknown") is None


async def test_delete_source(store):
    src1, src2 = await _seed(store)
    total_before = sum(s["chunks"] for s in await store.list_sources())
    assert total_before == 3

    assert await store.delete_source(src1) is True

    remaining = await store.list_sources()
    assert len(remaining) == 1
    assert remaining[0]["id"] == src2
    # Deleted source's origin is gone and its chunks cascaded away.
    origins = [s["origin"] for s in remaining]
    assert "origin-1" not in origins
    total_after = sum(s["chunks"] for s in remaining)
    assert total_after == 1
    # The source row itself is gone, so it's no longer findable by origin.
    assert await store.find_source_id_by_origin("origin-1") is None


async def test_delete_source_missing(store):
    await _seed(store)
    assert await store.delete_source("source:doesnotexist") is False


async def test_delete_source_malformed_id(store):
    # A malformed id can't reference a stored source -> False, no exception.
    assert await store.delete_source("not-a-valid-id") is False
