"""Weaviate adapter — an HONEST HOLDOUT on this host.

Weaviate has no maintained arm64-macOS embedded mode; running it requires a Docker
(a memory-competing Linux VM on Apple Silicon), against ragcore's standing
no-Docker rule. The protocol-shaped surface is designed here, but construction
raises — the `bench/serving.py vllm_serve` (CUDA) / `awq` precedent. Provision a
real Weaviate (Docker/cloud) to lift this holdout."""
from __future__ import annotations

from ragcore.errors import RagcoreError


class WeaviateStore:
    """VectorStore-shaped, but unavailable on this host (see module docstring)."""

    def __init__(self, path: str, collection: str = "ragcore") -> None:
        raise RagcoreError(
            "Weaviate is a documented holdout on this host: no arm64-mac embedded "
            "mode; running it needs Docker (against the no-Docker rule). Design-only."
        )

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:  # pragma: no cover
        raise RagcoreError("Weaviate holdout")

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict]:  # pragma: no cover
        raise RagcoreError("Weaviate holdout")
