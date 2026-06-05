"""In-house semantic cache backed by SQLite (stdlib only).

Stores query embeddings and their answers; short-circuits ask on a
cosine-similarity hit at or above the configured threshold.
"""
from __future__ import annotations

import json
import math
import sqlite3


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Persistent semantic cache using SQLite + cosine similarity."""

    def __init__(self, path: str, threshold: float) -> None:
        self.threshold = threshold
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache("
            "query TEXT, embedding TEXT, answer TEXT, sources TEXT)"
        )
        self._conn.commit()

    def put(
        self,
        query: str,
        embedding: list[float],
        answer: str,
        sources: list[str],
    ) -> None:
        self._conn.execute(
            "INSERT INTO cache(query, embedding, answer, sources) VALUES (?, ?, ?, ?)",
            (query, json.dumps(embedding), answer, json.dumps(sources)),
        )
        self._conn.commit()

    def get(self, query_embedding: list[float]) -> dict | None:
        cursor = self._conn.execute(
            "SELECT query, embedding, answer, sources FROM cache"
        )
        best_sim = -1.0
        best_row = None
        for query, emb_json, answer, sources_json in cursor:
            stored_emb: list[float] = json.loads(emb_json)
            sim = _cosine(query_embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_row = (query, answer, sources_json)
        if best_sim >= self.threshold and best_row is not None:
            query, answer, sources_json = best_row
            return {
                "query": query,
                "answer": answer,
                "sources": json.loads(sources_json),
            }
        return None
