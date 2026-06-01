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
