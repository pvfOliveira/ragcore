"""Qdrant-backed VectorStore adapter using in-process local mode (no server,
no Docker). qdrant-client is lazy-imported so the module imports without it.

API note: qdrant-client >= 1.10 removed the legacy ``QdrantClient.search()``
method entirely. This adapter uses ``query_points()`` (introduced in 1.10) and
handles the ``QueryResponse`` return type (a dataclass with a ``.points``
attribute, each element being a ``ScoredPoint``).
"""
from __future__ import annotations

import uuid


class QdrantStore:
    """VectorStore over an embedded on-disk Qdrant (``QdrantClient(path=...)``)."""

    def __init__(self, path: str, collection: str = "ragcore") -> None:
        self._path = path
        self._collection = collection
        self._client = None
        self._dim = None

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy — optional extra
            self._client = QdrantClient(path=self._path)
        return self._client

    def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_client()
        if self._dim is None:
            existing = {c.name for c in client.get_collections().collections}
            if self._collection not in existing:
                client.create_collection(
                    self._collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self._dim = dim

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        if not chunks:
            return
        self._ensure_collection(len(chunks[0]["embedding"]))
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                vector=c["embedding"],
                payload={"chunk_id": c["id"], "text": c["text"], "source_id": source_id},
            )
            for c in chunks
        ]
        self._get_client().upsert(self._collection, points=points)

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict]:
        """Return the *k* nearest chunks to *query*.

        Uses ``query_points()`` (qdrant-client >= 1.10); the legacy
        ``search()`` method was removed in 1.10+.  The return is a
        ``QueryResponse`` whose ``.points`` is a list of ``ScoredPoint``.
        """
        self._ensure_collection(len(query))
        response = self._get_client().query_points(
            self._collection, query=query, limit=k
        )
        return [
            {
                "id": h.payload["chunk_id"],
                "text": h.payload["text"],
                "score": float(h.score),
            }
            for h in response.points
        ]
