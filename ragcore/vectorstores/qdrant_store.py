"""Qdrant-backed VectorStore adapter using in-process local mode (no server,
no Docker). qdrant-client is lazy-imported so the module imports without it.

API note: qdrant-client >= 1.10 removed the legacy ``QdrantClient.search()``
method entirely. This adapter uses ``query_points()`` (introduced in 1.10) and
handles the ``QueryResponse`` return type (a dataclass with a ``.points``
attribute, each element being a ``ScoredPoint``).

Hybrid mode (``hybrid=True``, default OFF): named "dense" + "sparse" vectors
with fastembed learned-sparse encodings (BM42/SPLADE) and dense+sparse fusion
INSIDE Qdrant. HONEST DISTINCTION: this is distinct from retrieve.py's
client-side RRF over SurrealDB BM25 full-text — the sparse signal here is a
learned sparse embedding, fused server-side by Qdrant's Query API. Hybrid OFF
keeps the v3 unnamed-vector schema and query path byte-identical. The schema
is chosen at collection-creation time, so switching hybrid on/off requires a
fresh collection (or path) — existing collections are never migrated.

Schema-mismatch detection: if a collection already exists with a vector config
that does not match the requested hybrid flag, _ensure_collection raises
ValueError so the caller fails fast rather than ingesting into the wrong schema.
"""
from __future__ import annotations

import uuid


class QdrantStore:
    """VectorStore over an embedded on-disk Qdrant (``QdrantClient(path=...)``)."""

    def __init__(self, path: str, collection: str = "ragcore",
                 hybrid: bool = False,
                 sparse_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions") -> None:
        self._path = path
        self._collection = collection
        self._client = None
        self._dim = None
        self._hybrid = hybrid
        self._sparse_model = sparse_model
        self._sparse_encoder = None

    @property
    def supports_hybrid(self) -> bool:
        return self._hybrid

    def _encode_sparse_batch(self, texts: list[str]) -> list:
        """fastembed sparse embeddings for *texts* (lazy model init).

        Returns a list of sparse embeddings parallel to *texts*.
        """
        if self._sparse_encoder is None:
            from fastembed import SparseTextEmbedding  # lazy — optional extra
            self._sparse_encoder = SparseTextEmbedding(model_name=self._sparse_model)
        return list(self._sparse_encoder.embed(texts))

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy — optional extra
            self._client = QdrantClient(path=self._path)
        return self._client

    def close(self) -> None:
        """Release the file lock on the storage folder (local mode only)."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._dim = None

    def _ensure_collection(self, dim: int):
        from qdrant_client.models import Distance, VectorParams
        client = self._get_client()
        if self._dim is None:
            existing = {c.name for c in client.get_collections().collections}
            if self._collection not in existing:
                if self._hybrid:
                    from qdrant_client.models import Modifier, SparseVectorParams
                    client.create_collection(
                        self._collection,
                        vectors_config={
                            "dense": VectorParams(size=dim, distance=Distance.COSINE)
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(modifier=Modifier.IDF)
                        },
                    )
                else:
                    client.create_collection(
                        self._collection,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
            else:
                # Collection exists: verify schema matches requested hybrid flag.
                # info.config.params.vectors is a dict for named (hybrid) collections
                # and a VectorParams instance for unnamed (dense-only) collections.
                info = client.get_collection(self._collection)
                existing_is_hybrid = isinstance(info.config.params.vectors, dict)
                if existing_is_hybrid != self._hybrid:
                    raise ValueError(
                        f"collection {self._collection!r} was created "
                        f"{'with' if existing_is_hybrid else 'without'} named "
                        f"dense+sparse vectors; toggling qdrant_hybrid requires a "
                        f"fresh collection/qdrant_path (re-ingest)"
                    )
            self._dim = dim

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        from qdrant_client.models import PointStruct
        if not chunks:
            return
        self._ensure_collection(len(chunks[0]["embedding"]))
        points = []
        if self._hybrid:
            from qdrant_client.models import SparseVector
            sparse_results = self._encode_sparse_batch([c["text"] for c in chunks])
            for c, se in zip(chunks, sparse_results):
                vector = {
                    "dense": c["embedding"],
                    "sparse": SparseVector(indices=list(se.indices),
                                           values=list(se.values)),
                }
                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                        vector=vector,
                        payload={"chunk_id": c["id"], "text": c["text"], "source_id": source_id},
                    )
                )
        else:
            for c in chunks:
                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                        vector=c["embedding"],
                        payload={"chunk_id": c["id"], "text": c["text"], "source_id": source_id},
                    )
                )
        self._get_client().upsert(self._collection, points=points)

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict]:
        """Return the *k* nearest chunks to *query*.

        Uses ``query_points()`` (qdrant-client >= 1.10); the legacy
        ``search()`` method was removed in 1.10+.  The return is a
        ``QueryResponse`` whose ``.points`` is a list of ``ScoredPoint``.
        """
        self._ensure_collection(len(query))
        if self._hybrid:
            response = self._get_client().query_points(
                self._collection, query=query, using="dense", limit=k
            )
        else:
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

    async def hybrid_search(self, text: str, query: list[float], k: int = 10) -> list[dict]:
        """Server-side dense+sparse fusion (Query API prefetch + RRF).

        Uses Qdrant's Query API with prefetch for both dense and sparse signals,
        fused via Reciprocal Rank Fusion (RRF) server-side. Local-mode
        FusionQuery support confirmed for qdrant-client>=1.10 at build time
        (2026-06-11), so no client-side fallback exists — this is the shipped
        path.
        """
        self._ensure_collection(len(query))
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector
        se = self._encode_sparse_batch([text])[0]
        sparse = SparseVector(indices=list(se.indices), values=list(se.values))
        response = self._get_client().query_points(
            self._collection,
            prefetch=[
                Prefetch(query=query, using="dense", limit=k * 2),
                Prefetch(query=sparse, using="sparse", limit=k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
        return [
            {"id": p.payload["chunk_id"], "text": p.payload["text"],
             "score": float(p.score)}
            for p in response.points
        ]
