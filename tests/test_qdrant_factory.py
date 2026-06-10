import pytest


def test_factory_builds_qdrant(tmp_path):
    pytest.importorskip("qdrant_client")
    from ragcore.config import StoreConfig
    from ragcore.vectorstores.base import make_vector_store
    from ragcore.vectorstores.qdrant_store import QdrantStore

    class Cfg:
        store = StoreConfig(vector_backend="qdrant", qdrant_path=str(tmp_path / "qd"))
    s = make_vector_store(Cfg())
    assert isinstance(s, QdrantStore)
