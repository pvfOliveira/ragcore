"""LIVE: real Embedded Weaviate round-trip on this host (macOS arm64).

First run downloads the pinned Darwin-all.zip server binary (network gate).
Records a RunRegistry run tagged weaviate-v4.
"""
from __future__ import annotations

import asyncio
import copy
import shutil
import socket

import pytest

pytestmark = pytest.mark.live


def _network_up(host="github.com", port=443) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_up(), reason="network needed for embedded-binary download")
def test_weaviate_embedded_round_trip(tmp_path, surreal_url):
    pytest.importorskip("weaviate")
    from ragcore.config import SurrealConfig, load_config
    from ragcore.embedding import generate_embedding
    from ragcore.vectorstores.weaviate_store import WeaviateStore

    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_wv", database="ragcore_wv")
    store = WeaviateStore(path=str(tmp_path / "weaviate"), collection="ragcorelive",
                          version=cfg.store.weaviate_version)

    async def _run():
        try:
            texts = {
                "c1": "Weaviate Embedded runs the real server as a child process without Docker.",
                "c2": "Basil grows best with six hours of direct sunlight.",
            }
            chunks = []
            for cid, text in texts.items():
                emb = await generate_embedding(text, cfg, chunk_size=cfg.chunking.chunk_size)
                chunks.append({"id": cid, "text": text, "embedding": emb})
            await store.add_embeddings("live-src", chunks)

            q = await generate_embedding("How does embedded Weaviate run?", cfg,
                                         chunk_size=cfg.chunking.chunk_size)
            hits = await store.vector_search(q, k=1)
            assert hits and hits[0]["id"] == "c1"
            assert set(hits[0]) == {"id", "text", "score"}

            from ragcore.llmops.registry import RunRegistry
            run_id = await RunRegistry(cfg.surreal).record(
                metrics={"weaviate": {"top1_correct": 1.0}},
                config_snapshot={"vector_backend": "weaviate",
                                 "embedded_version": cfg.store.weaviate_version,
                                 "platform": "darwin-arm64, Darwin-all.zip"},
                dataset=None, tag="weaviate-v4", gate_passed=True)
            assert run_id and ":" in run_id
        finally:
            store.close()

    try:
        asyncio.run(_run())
    finally:
        shutil.rmtree(tmp_path / "weaviate", ignore_errors=True)
