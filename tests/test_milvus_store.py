"""Tests for MilvusStore (Milvus Lite local-file adapter)."""
from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")
from ragcore.vectorstores.milvus_store import MilvusStore  # noqa: E402


def _milvus_lite_available(tmp_path) -> tuple[bool, str]:
    """Return (available, reason) by probing Milvus Lite with a real file open."""
    try:
        from pymilvus import MilvusClient

        probe_db = str(tmp_path / "_probe.db")
        c = MilvusClient(uri=probe_db)
        c.close()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


@pytest.fixture(autouse=True)
def _require_milvus_lite(tmp_path):
    available, reason = _milvus_lite_available(tmp_path)
    if not available:
        pytest.skip(f"Milvus Lite unavailable on this platform: {reason}")


async def test_milvus_index_and_search(tmp_path):
    s = MilvusStore(uri=str(tmp_path / "milvus.db"), dim=3)
    await s.add_embeddings("src1", [
        {"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "wind", "embedding": [0.0, 1.0, 0.0]},
    ])
    out = await s.vector_search([0.9, 0.1, 0.0], k=1)
    assert out[0]["id"] == "a"
    assert out[0]["text"] == "solar"
    assert "score" in out[0]


async def test_milvus_score_higher_is_better(tmp_path):
    """COSINE distance is converted to a higher-is-better score."""
    s = MilvusStore(uri=str(tmp_path / "milvus.db"), dim=3)
    await s.add_embeddings("src1", [
        {"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "anti", "embedding": [-1.0, 0.0, 0.0]},
    ])
    out = await s.vector_search([1.0, 0.0, 0.0], k=2)
    # "solar" is most similar → highest score
    assert out[0]["id"] == "a"
    assert out[0]["score"] > out[1]["score"]
    # Score for identical vector should be near 1.0 (1 - 0 distance)
    assert out[0]["score"] > 0.9


async def test_milvus_empty_returns_empty(tmp_path):
    """Searching an empty store returns an empty list."""
    s = MilvusStore(uri=str(tmp_path / "milvus.db"), dim=3)
    out = await s.vector_search([1.0, 0.0, 0.0], k=5)
    assert out == []


async def test_milvus_k_limits_results(tmp_path):
    s = MilvusStore(uri=str(tmp_path / "milvus.db"), dim=3)
    await s.add_embeddings("src1", [
        {"id": "a", "text": "solar", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "text": "wind", "embedding": [0.0, 1.0, 0.0]},
        {"id": "c", "text": "grid", "embedding": [0.0, 0.0, 1.0]},
    ])
    out = await s.vector_search([0.5, 0.5, 0.0], k=2)
    assert len(out) == 2
