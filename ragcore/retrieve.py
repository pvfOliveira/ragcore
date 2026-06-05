"""Hybrid retrieval: fuse vector + full-text rankings with Reciprocal Rank Fusion."""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], key: str = "id", k: int = 60
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists. Higher score = better.

    RRF score for an item = sum over lists of 1 / (k + rank), rank starting at 1.
    """
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}
    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            ident = item[key]
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            items.setdefault(ident, item)
    fused = []
    for ident, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        merged = dict(items[ident])
        merged["score"] = score
        fused.append(merged)
    return fused


def _adapter_hit_to_ragcore(hit: dict[str, Any]) -> dict[str, Any]:
    """Bridge an adapter result ``{"id","text","score"}`` to ragcore's chunk
    shape. RRF fuses on ``id``; downstream generation reads ``content`` and
    citations read ``source`` (adapters carry no source link, so fall back to
    the chunk id to keep the shape total)."""
    out = dict(hit)
    out["content"] = hit.get("text", hit.get("content", ""))
    out.setdefault("source", hit["id"])
    out.pop("text", None)
    return out


def vector_store_for(config) -> Any | None:
    """Return the configured pluggable VectorStore, or None for the default
    surreal backend (whose vector ranking comes from the SurrealDB `store`
    itself, so no adapter is routed). Returns None when config is absent."""
    if config is None:
        return None
    if config.store.vector_backend == "surreal":
        return None
    from ragcore.vectorstores.base import make_vector_store

    return make_vector_store(config)


async def hybrid_search(store, embedder_fn, query: str, k: int = 10, vector_store=None):
    """Run vector + full-text search and fuse with RRF.

    `embedder_fn(query) -> list[float]` produces the query embedding.
    `store` exposes async vector_search / text_search (BM25).

    When `vector_store` is provided, the vector ranking comes from that
    pluggable backend (adapter-shaped results are bridged to ragcore's chunk
    shape); otherwise it comes from `store` (default, byte-identical behavior).
    The BM25 ranking always comes from `store.text_search`.
    """
    query_vec = await embedder_fn(query)
    if vector_store is not None:
        hits = await vector_store.vector_search(query_vec, k=k)
        vres = [_adapter_hit_to_ragcore(h) for h in hits]
    else:
        vres = await store.vector_search(query_vec, k=k)
    tres = await store.text_search(query, k=k)
    return reciprocal_rank_fusion([vres, tres], key="id", k=60)[:k]
