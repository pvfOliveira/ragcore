"""ChromaDB-backed VectorStore adapter.

chromadb is an optional dependency; it is lazy-imported inside each method so
the module can be imported without chromadb installed (an ImportError is only
raised at call time if the package is missing).
"""
from __future__ import annotations


class ChromaStore:
    """VectorStore implementation backed by a local ChromaDB PersistentClient.

    Args:
        path: Filesystem path where ChromaDB will persist its data.
        collection: Name of the ChromaDB collection to use.
    """

    def __init__(self, path: str, collection: str = "ragcore") -> None:
        self._path = path
        self._collection_name = collection
        self._collection = None  # lazily initialised on first use

    def _get_collection(self):  # type: ignore[return]
        """Return the ChromaDB collection, creating it on first access."""
        if self._collection is None:
            import chromadb  # lazy import — optional extra

            client = chromadb.PersistentClient(path=self._path)
            self._collection = client.get_or_create_collection(
                self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        """Upsert *chunks* into the collection, tagging each with *source_id*.

        Each chunk must contain ``{"id": str, "text": str, "embedding": list[float]}``.
        """
        collection = self._get_collection()
        ids = [c["id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source_id": source_id} for _ in chunks]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def vector_search(
        self, query: list[float], k: int = 10
    ) -> list[dict]:
        """Return the *k* nearest chunks to *query*.

        ChromaDB cosine space returns distances where 0 = identical and
        higher = more distant.  We convert to a higher-is-better score via
        ``score = 1.0 - distance``.

        Returns:
            List of ``{"id": str, "text": str, "score": float}`` dicts,
            ordered from highest score (most similar) to lowest.
        """
        collection = self._get_collection()
        result = collection.query(
            query_embeddings=[query],
            n_results=k,
        )
        # result["ids"], result["documents"], result["distances"] are each
        # lists-of-lists (one inner list per query vector); index [0] for ours.
        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        return [
            {"id": doc_id, "text": doc, "score": 1.0 - dist}
            for doc_id, doc, dist in zip(ids, documents, distances)
        ]
