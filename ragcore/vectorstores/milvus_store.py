"""Milvus Lite-backed VectorStore adapter.

pymilvus is an optional dependency; it is lazy-imported inside each method so
the module can be imported without pymilvus installed (an ImportError is only
raised at call time if the package is missing).

Milvus Lite is the embedded, local-file mode of pymilvus — no server required.
Pass a ``.db`` filesystem path as ``uri`` to activate it::

    store = MilvusStore(uri="./ragcore.db", dim=1536)

The collection uses a VARCHAR primary key (field ``id``), a FLOAT_VECTOR field
(field ``vector``), and a VARCHAR ``text`` field for the document content.
Metric type is COSINE; Milvus returns distance where 0=identical, so we
convert to a higher-is-better score via ``score = 1.0 - distance``.
"""
from __future__ import annotations


class MilvusStore:
    """VectorStore implementation backed by a Milvus Lite local-file database.

    Args:
        uri:        Filesystem path ending in ``.db`` (Milvus Lite) or a
                    Milvus/Zilliz server URL.
        dim:        Embedding dimension.  Required on first use when the
                    collection does not yet exist; inferred from the first
                    batch of embeddings otherwise.
        collection: Name of the Milvus collection to use.
    """

    def __init__(
        self,
        uri: str,
        dim: int | None = None,
        collection: str = "ragcore",
    ) -> None:
        self._uri = uri
        self._dim = dim
        self._collection_name = collection
        self._client = None  # lazily initialised on first use

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):  # type: ignore[return]
        """Return the MilvusClient, creating it on first access."""
        if self._client is None:
            from pymilvus import MilvusClient  # lazy import — optional extra

            self._client = MilvusClient(uri=self._uri)
        return self._client

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection if it does not already exist."""
        client = self._get_client()
        if not client.has_collection(self._collection_name):
            client.create_collection(
                self._collection_name,
                dimension=dim,
                primary_field_name="id",
                id_type="str",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        """Insert *chunks* into the Milvus collection.

        Each chunk must contain ``{"id": str, "text": str, "embedding": list[float]}``.
        The *source_id* is accepted for protocol compatibility but is not stored
        because Milvus Lite auto-schema does not include a dynamic metadata field
        in the default schema; callers may add it to the chunk dict if needed.
        """
        if not chunks:
            return

        # Infer / validate dimension
        dim = len(chunks[0]["embedding"])
        if self._dim is None:
            self._dim = dim

        self._ensure_collection(self._dim)

        rows = [
            {"id": c["id"], "vector": c["embedding"], "text": c["text"]}
            for c in chunks
        ]
        client = self._get_client()
        client.insert(self._collection_name, rows)

    async def vector_search(
        self, query: list[float], k: int = 10
    ) -> list[dict]:
        """Return the *k* most similar chunks to *query* (higher score = more similar).

        Milvus COSINE metric returns distance where 0=identical and 2=opposite.
        We convert to a higher-is-better score via ``score = 1.0 - distance``.

        Returns:
            List of ``{"id": str, "text": str, "score": float}`` dicts,
            ordered from highest score (most similar) to lowest.
            Returns ``[]`` when the collection does not exist or is empty.
        """
        client = self._get_client()

        if not client.has_collection(self._collection_name):
            return []

        results = client.search(
            self._collection_name,
            data=[query],
            limit=k,
            output_fields=["text"],
            search_params={"metric_type": "COSINE"},
        )

        # results is a list[list[Hit]]; index [0] for our single query vector.
        # Each Hit is a dict with keys: "id", "distance", "entity" (dict of output_fields).
        return [
            {
                "id": hit["id"],
                "text": hit["entity"]["text"],
                "score": 1.0 - float(hit["distance"]),
            }
            for hit in results[0]
        ]
