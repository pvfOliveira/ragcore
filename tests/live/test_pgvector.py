"""LIVE: real Postgres+pgvector add->search round-trip with real Ollama embeddings.

Requires: local Ollama (live gate), local Postgres with the vector extension
in ragcore_test (Task 1 host setup). Skips cleanly when Postgres is absent.
Records a RunRegistry run tagged pgvector-v4 (the test_qdrant.py precedent).
"""
from __future__ import annotations

import asyncio
import copy
import uuid

import pytest

pytest.importorskip("psycopg")

from tests.conftest import postgres_dsn

pytestmark = pytest.mark.live

_DSN = postgres_dsn()


@pytest.mark.skipif(_DSN is None, reason="local Postgres (ragcore_test) not reachable")
def test_pgvector_round_trip(surreal_url):
    from ragcore.config import SurrealConfig, load_config
    from ragcore.embedding import generate_embedding
    from ragcore.vectorstores.pgvector_store import PgVectorStore

    cfg = copy.deepcopy(load_config())
    table = f"ragcore_live_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(dsn=_DSN, collection=table)

    async def _run():
        try:
            texts = {
                "c1": "SurrealDB is ragcore's default document and vector store.",
                "c2": "pgvector adds approximate nearest neighbour search to Postgres via an HNSW index.",
                "c3": "Basil grows best with six hours of direct sunlight.",
            }
            chunks = []
            for cid, text in texts.items():
                emb = await generate_embedding(text, cfg, chunk_size=cfg.chunking.chunk_size)
                chunks.append({"id": cid, "text": text, "embedding": emb})
            await store.add_embeddings("live-src", chunks)

            q = await generate_embedding(
                "Which extension gives Postgres vector search?", cfg,
                chunk_size=cfg.chunking.chunk_size)
            hits = await store.vector_search(q, k=2)
            assert hits and hits[0]["id"] == "c2"
            assert set(hits[0]) == {"id", "text", "score"}

            # registry evidence (the *-v4 convention); the registry lives in the
            # per-test SurrealDB the surreal_url fixture starts (test_qdrant.py precedent).
            from ragcore.llmops.registry import RunRegistry
            cfg.surreal = SurrealConfig(
                url=surreal_url, namespace="ragcore_pg", database="ragcore_pg")
            run_id = await RunRegistry(cfg.surreal).record(
                metrics={"pgvector": {"top1_correct": 1.0, "hits": float(len(hits))}},
                config_snapshot={"vector_backend": "pgvector", "table": table,
                                 "index": "hnsw vector_cosine_ops"},
                dataset=None, tag="pgvector-v4", gate_passed=True)
            assert run_id and ":" in run_id
        finally:
            try:
                conn = await store._get_conn()
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            finally:
                await store.close()

    asyncio.run(_run())
