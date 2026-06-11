"""Weaviate-backed VectorStore over **Embedded Weaviate** (no Docker).

CORRECTION (2026-06-11, supersedes the RAG v3 holdout note that lived here):
Embedded Weaviate IS supported on macOS — the v4 Python client downloads a
universal ``Darwin-all.zip`` server binary and runs it as a child process
(weaviate docs: deploy/installation-guides/embedded). The RAG v3 claim
("no arm64-mac embedded mode; needs Docker") was outdated. First use needs
network for the one-time binary download; the server version is pinned via
config for reproducibility.

ragcore brings its own Ollama embeddings, so the collection is created with
self-provided vectors (no Weaviate vectorizer modules). The sync v4 client is
called inside the async protocol methods (the QdrantStore precedent).
"""
from __future__ import annotations

import re
import uuid

_NAME = re.compile(r"^[A-Z][_0-9A-Za-z]*$")


class WeaviateStore:
    """VectorStore over Embedded Weaviate (self-provided vectors, cosine)."""

    def __init__(self, path: str, collection: str = "ragcore",
                 version: str = "1.30.5") -> None:
        self._path = path
        # Weaviate collection names must start uppercase.
        self._name = collection[:1].upper() + collection[1:]
        if not _NAME.fullmatch(self._name):
            raise ValueError(f"invalid Weaviate collection name {self._name!r}")
        self._version = version
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import weaviate  # lazy — optional extra
            self._client = weaviate.connect_to_embedded(
                version=self._version,
                persistence_data_path=self._path,
                environment_variables={"DISABLE_TELEMETRY": "true"},
            )
            if not self._client.collections.exists(self._name):
                from weaviate.classes.config import (
                    Configure,
                    DataType,
                    Property,
                    VectorDistances,
                )
                self._client.collections.create(
                    name=self._name,
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                        ),
                    ),
                    properties=[
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="source_id", data_type=DataType.TEXT),
                    ],
                )
            self._collection = self._client.collections.get(self._name)
        return self._collection

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        if not chunks:
            return
        col = self._get_collection()
        with col.batch.dynamic() as batch:
            for c in chunks:
                batch.add_object(
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                    properties={"chunk_id": c["id"], "text": c["text"],
                                "source_id": source_id},
                    vector=c["embedding"],
                )
        failed = getattr(col.batch, "failed_objects", None)
        if failed:
            from ragcore.errors import RagcoreError
            raise RagcoreError(
                f"Weaviate batch insert failed for {len(failed)} object(s): "
                f"{failed[0].message if hasattr(failed[0], 'message') else failed[0]}"
            )

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict]:
        col = self._get_collection()
        from_meta = None
        try:
            from weaviate.classes.query import MetadataQuery
            from_meta = MetadataQuery(distance=True)
        except ImportError:  # deterministic tests run without weaviate installed
            pass
        res = col.query.near_vector(near_vector=query, limit=k,
                                    return_metadata=from_meta)
        return [
            {
                "id": o.properties["chunk_id"],
                "text": o.properties["text"],
                # cosine distance in [0,2] → similarity-style score, higher=better
                "score": 1.0 - float(o.metadata.distance),
            }
            for o in res.objects
        ]

    def close(self) -> None:
        """Shut down the embedded server child process."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._collection = None
