"""LIVE: hybrid sparse+dense beats dense-only on a keyword-heavy fixture.

The fixture plants a rare exact token ("ZX-9943-K") inside a chunk whose
prose is semantically bland, among distractor chunks that are semantically
CLOSE to the query wording — dense alone struggles to rank the target top-1;
the sparse (BM42) signal locks onto the rare token. Fusion is server-side
``FusionQuery(RRF)`` inside Qdrant (Task 6, commit 40a242e). First run
downloads the fastembed sparse model (network gate). Records RunRegistry
tag hybrid-v4.

Host-specific note: on this MPS host the dense model (nomic-embed-text)
also encodes the literal rare token "ZX-9943-K" strongly, producing
``dense_rank=0`` and therefore ``hybrid_rank=0`` as well — so the hybrid arm
alone only proves "hybrid doesn't hurt."  The sparse contribution is therefore
proven via the **sparse-only arm**: BM42 must rank the rare-token chunk top-1
on its own (``sparse_rank=0``).  This is honest evidence for the
``sparse-retrieval`` skill.
"""
from __future__ import annotations

import asyncio
import copy
import socket

import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("fastembed")

pytestmark = pytest.mark.live


def _network_up(host="huggingface.co", port=443) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_up(), reason="network needed for sparse-model download")
def test_hybrid_beats_dense_on_keyword_query(tmp_path, surreal_url):
    from ragcore.config import SurrealConfig, load_config
    from ragcore.embedding import generate_embedding
    from ragcore.llmops.registry import RunRegistry
    from ragcore.vectorstores.qdrant_store import QdrantStore

    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(
        url=surreal_url, namespace="ragcore_hyb", database="ragcore_hyb")

    target = "Error code ZX-9943-K indicates a checksum fault in the firmware loader."
    distractors = [
        "Troubleshooting firmware update failures and recovery procedures for embedded devices.",
        "A guide to diagnosing common error codes raised during device boot.",
        "Firmware loaders verify image integrity before flashing the device.",
        "Checksum validation protects firmware images against corruption in transit.",
    ]
    texts = {f"d{i}": t for i, t in enumerate(distractors)}
    texts["target"] = target
    query = "what does ZX-9943-K mean?"

    dense_store = QdrantStore(path=str(tmp_path / "dense"), collection="live")
    hybrid_store = QdrantStore(
        path=str(tmp_path / "hyb"), collection="live",
        hybrid=True, sparse_model=cfg.store.qdrant_sparse_model)

    def _rank_of_target(hits: list[dict]) -> int:
        # A miss counts as rank len(hits) (worst possible).
        return next(
            i for i, h in enumerate(hits + [{"id": "target"}]) if h["id"] == "target")

    async def _run():
        try:
            chunks = []
            for cid, text in texts.items():
                emb = await generate_embedding(
                    text, cfg, chunk_size=cfg.chunking.chunk_size)
                chunks.append({"id": cid, "text": text, "embedding": emb})
            await dense_store.add_embeddings("live-src", chunks)
            await hybrid_store.add_embeddings("live-src", chunks)

            qv = await generate_embedding(
                query, cfg, chunk_size=cfg.chunking.chunk_size)
            dense_hits = await dense_store.vector_search(qv, k=5)
            hybrid_hits = await hybrid_store.hybrid_search(query, qv, k=5)

            dense_rank = _rank_of_target(dense_hits)
            hybrid_rank = _rank_of_target(hybrid_hits)

            # --- sparse-only arm: BM42 must rank the rare-token chunk top-1 ---
            from qdrant_client.models import SparseVector as _SparseVector
            se = hybrid_store._encode_sparse_batch([query])[0]
            sparse_hits = hybrid_store._get_client().query_points(
                hybrid_store._collection,
                query=_SparseVector(
                    indices=list(se.indices), values=list(se.values)),
                using="sparse",
                limit=5,
            ).points
            sparse_ids = [p.payload["chunk_id"] for p in sparse_hits]
            sparse_rank = next(
                i for i, cid in enumerate(sparse_ids + ["target"]) if cid == "target"
            )
            assert sparse_rank == 0, (
                f"sparse-only retrieval failed to lock onto the rare token "
                f"(rank {sparse_rank}, ids {sparse_ids})"
            )
            # --- end sparse-only arm ---

            assert hybrid_rank == 0 or hybrid_rank < dense_rank, (
                f"sparse signal did not contribute: hybrid_rank={hybrid_rank} "
                f"vs dense_rank={dense_rank}")

            run_id = await RunRegistry(cfg.surreal).record(
                metrics={"hybrid": {"dense_rank": float(dense_rank),
                                    "hybrid_rank": float(hybrid_rank),
                                    "sparse_rank": float(sparse_rank)}},
                config_snapshot={"vector_backend": "qdrant", "qdrant_hybrid": True,
                                 "sparse_model": cfg.store.qdrant_sparse_model},
                dataset=None, tag="hybrid-v4", gate_passed=True)
            assert run_id and ":" in run_id
        finally:
            dense_store.close()
            hybrid_store.close()

    asyncio.run(_run())
