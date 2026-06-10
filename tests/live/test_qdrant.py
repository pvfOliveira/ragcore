"""Live proof: Qdrant in-process (no Docker) stores real Ollama embeddings and
returns the relevant chunk for a query embedding."""
from __future__ import annotations

import asyncio
import copy

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def test_qdrant_ingest_and_search(surreal_url, tmp_path):
    pytest.importorskip("qdrant_client")
    from ragcore.embedding import generate_embedding
    from ragcore.vectorstores.qdrant_store import QdrantStore
    from ragcore.llmops.registry import RunRegistry

    cfg = copy.deepcopy(load_config())
    store = QdrantStore(path=str(tmp_path / "qd"), collection="live")

    async def go():
        texts = ["Reciprocal rank fusion blends ranked lists.", "Bananas are yellow."]
        chunks = []
        for i, t in enumerate(texts):
            emb = await generate_embedding(t, cfg, chunk_size=cfg.chunking.chunk_size)
            chunks.append({"id": f"c{i}", "text": t, "embedding": emb})
        await store.add_embeddings("s", chunks)
        q = await generate_embedding("how does RRF work?", cfg, chunk_size=cfg.chunking.chunk_size)
        return await store.vector_search(q, k=1)

    res = asyncio.run(go())
    assert res and "rank fusion" in res[0]["text"].lower()

    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_qd", database="ragcore_qd")
    assert asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"qdrant": {"top1_correct": 1.0}}, config_snapshot={"backend": "qdrant"},
        tag="qdrant-v3", gate_passed=True))
