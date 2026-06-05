"""VectorStore protocol and backend factory.

All vector backends implement this async protocol so the storage layer is
config-selectable without changing call sites.

Chunk dict shape (input to add_embeddings):
    {"id": str, "text": str, "embedding": list[float], ...extra fields allowed}

Result dict shape (returned by vector_search):
    {"id": str, "text": str, "score": float}
"""
from __future__ import annotations

import typing
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Async protocol for a vector index backend.

    Implementations must provide:
    - ``add_embeddings(source_id, chunks)``: store a list of chunk dicts.
      Each chunk must contain at minimum ``{"id", "text", "embedding"}``.
    - ``vector_search(query, k)``: return up to *k* nearest neighbours as
      ``[{"id": str, "text": str, "score": float}, ...]`` ordered
      highest-score first.
    """

    async def add_embeddings(
        self, source_id: str, chunks: list[dict]
    ) -> None:
        """Persist *chunks* associated with *source_id*."""
        ...

    async def vector_search(
        self, query: list[float], k: int = 10
    ) -> list[dict]:
        """Return the *k* most similar chunks to *query* embedding."""
        ...


def make_vector_store(config: typing.Any) -> VectorStore:
    """Construct and return a VectorStore for the backend named in *config*.

    Dispatches on ``config.store.vector_backend``:
    - ``"surreal"``  — existing SurrealDB-backed ``Store`` (default).
    - ``"chroma"``   — ``ragcore.vectorstores.chroma_store`` (Task 3).
    - ``"faiss"``    — ``ragcore.vectorstores.faiss_store`` (Task 4).
    - ``"milvus"``   — ``ragcore.vectorstores.milvus_store`` (Task 5).
    - anything else  — raises ``ValueError``.

    The chroma/faiss/milvus branches are wired lazily; they will raise an
    ``ImportError`` at call time until the adapter modules are added.
    """
    backend: str = config.store.vector_backend

    if backend == "surreal":
        from ragcore.store import Store

        return Store(config.surreal)

    if backend == "chroma":
        from ragcore.vectorstores.chroma_store import (
            ChromaStore,  # type: ignore[import]
        )

        return ChromaStore(config)

    if backend == "faiss":
        from ragcore.vectorstores.faiss_store import FaissStore  # type: ignore[import]

        return FaissStore(config)

    if backend == "milvus":
        from ragcore.vectorstores.milvus_store import (
            MilvusStore,  # type: ignore[import]
        )

        return MilvusStore(config)

    raise ValueError(
        f"Unknown vector_backend {backend!r}. "
        "Valid options: 'surreal', 'chroma', 'faiss', 'milvus'."
    )
