"""FlashRank cross-encoder reranking stage — applied after RRF, before generation."""
from __future__ import annotations

from typing import Any, Callable


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int,
    *,
    model: str | None = None,
    text_key: str = "content",
    _scorer: Callable[[str, list[str]], list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Reorder retrieval results by query-relevance via a cross-encoder; return top_k.

    Each result keeps its original fields and gains a ``rerank_score`` key.
    ``_scorer(query, passages) -> list[float]`` is injectable for tests; in
    production a FlashRank cross-encoder is used (lazy-imported so the
    dependency is only loaded when reranking is enabled).
    """
    if not results:
        return []

    passages = [r.get(text_key, "") for r in results]

    if _scorer is not None:
        scores = _scorer(query, passages)
    else:
        from flashrank import Ranker, RerankRequest  # lazy import — optional dep

        ranker = Ranker(model_name=model or "ms-marco-MiniLM-L-12-v2")
        ranked = ranker.rerank(
            RerankRequest(
                query=query,
                passages=[{"id": i, "text": p} for i, p in enumerate(passages)],
            )
        )
        # Map FlashRank's returned items back to our positional index.
        # FlashRank returns dicts with "id", "text", "score" sorted by score desc.
        scores = [0.0] * len(passages)
        for item in ranked:
            scores[item["id"]] = float(item["score"])

    order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order[:top_k]:
        merged = dict(results[i])
        merged["rerank_score"] = scores[i]
        out.append(merged)
    return out
