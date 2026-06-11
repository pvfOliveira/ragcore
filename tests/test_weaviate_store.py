"""Deterministic tests for WeaviateStore: a fake client object stands in for
embedded Weaviate (no binary download). Live round-trip: tests/live/test_weaviate.py.

This REPLACES tests/test_weaviate_holdout.py — the RAG v3 holdout claim
("no arm64-mac embedded mode") was outdated: Embedded Weaviate is supported on
macOS via a universal Darwin-all.zip binary, no Docker.
"""
import uuid
from contextlib import contextmanager

import pytest

from ragcore.vectorstores.weaviate_store import WeaviateStore


class FakeObject:
    def __init__(self, props, distance):
        self.properties = props

        class M:  # metadata carrier
            pass
        self.metadata = M()
        self.metadata.distance = distance


class FakeQuery:
    def __init__(self, objects):
        self._objects = objects
        self.calls = []

    def near_vector(self, near_vector, limit, return_metadata=None):
        self.calls.append({"near_vector": near_vector, "limit": limit})

        class R:
            pass
        r = R()
        r.objects = self._objects
        return r


class FakeBatch:
    """Fake batch returned by FakeBatchManager.dynamic()."""

    def __init__(self, collection):
        self._col = collection

    def add_object(self, **kw):
        self._col.inserted.append(kw)


class FakeBatchManager:
    """Fake col.batch with a dynamic() context manager and failed_objects."""

    def __init__(self, collection, failed=None):
        self._col = collection
        self.failed_objects = failed if failed is not None else []

    @contextmanager
    def dynamic(self):
        yield FakeBatch(self._col)


class FakeCollection:
    def __init__(self, objects, batch_failed=None):
        self.query = FakeQuery(objects)
        self.inserted = []
        self.batch = FakeBatchManager(self, failed=batch_failed)


def test_module_imports_without_weaviate_installed():
    # construction must not import weaviate (lazy optional dep)
    store = WeaviateStore(path="/tmp/x", collection="ragcore")
    assert store._client is None


async def test_vector_search_maps_distance_to_score():
    store = WeaviateStore(path="/tmp/x", collection="ragcore")
    objs = [FakeObject({"chunk_id": "c1", "text": "alpha"}, distance=0.1),
            FakeObject({"chunk_id": "c2", "text": "beta"}, distance=0.4)]
    fake = FakeCollection(objs)
    store._collection = fake                      # bypass connect
    hits = await store.vector_search([0.1, 0.2], k=2)
    assert hits == [{"id": "c1", "text": "alpha", "score": pytest.approx(0.9)},
                    {"id": "c2", "text": "beta", "score": pytest.approx(0.6)}]
    assert fake.query.calls[0]["limit"] == 2


def test_protocol_conformance():
    from ragcore.vectorstores.base import VectorStore
    assert isinstance(WeaviateStore(path="/tmp/x"), VectorStore)


def test_factory_dispatch():
    from ragcore.config import StoreConfig
    from ragcore.vectorstores.base import make_vector_store

    class Cfg:
        store = StoreConfig(vector_backend="weaviate", weaviate_path="data/wtest")

    store = make_vector_store(Cfg())
    assert isinstance(store, WeaviateStore)
    assert store._path == "data/wtest"


async def test_add_embeddings_happy_path():
    """add_embeddings inserts objects with uuid5 ids; no raise on empty failed_objects."""
    store = WeaviateStore(path="/tmp/x", collection="ragcore")
    fake = FakeCollection([], batch_failed=[])
    store._collection = fake  # bypass connect

    chunks = [
        {"id": "chunk-1", "text": "hello world", "embedding": [0.1, 0.2]},
        {"id": "chunk-2", "text": "foo bar", "embedding": [0.3, 0.4]},
    ]
    await store.add_embeddings("src-1", chunks)

    assert len(fake.inserted) == 2
    first = fake.inserted[0]
    assert first["uuid"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk-1"))
    assert first["properties"] == {"chunk_id": "chunk-1", "text": "hello world", "source_id": "src-1"}
    assert first["vector"] == [0.1, 0.2]

    second = fake.inserted[1]
    assert second["uuid"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "chunk-2"))
    assert second["properties"]["chunk_id"] == "chunk-2"


async def test_add_embeddings_raises_on_batch_failure():
    """add_embeddings raises RagcoreError when failed_objects is non-empty."""
    from ragcore.errors import RagcoreError

    store = WeaviateStore(path="/tmp/x", collection="ragcore")

    class FailedObj:
        message = "duplicate key"

    fake = FakeCollection([], batch_failed=[FailedObj()])
    store._collection = fake

    chunks = [{"id": "chunk-1", "text": "hello", "embedding": [0.1]}]
    with pytest.raises(RagcoreError, match="1 object"):
        await store.add_embeddings("src-1", chunks)


def test_invalid_collection_name_raises():
    """Collection names with hyphens (or other invalid chars) raise ValueError."""
    with pytest.raises(ValueError):
        WeaviateStore(path="/tmp/x", collection="bad-name")
