"""Batched embedding with retry + mean-pooling for oversized text.

Batch size is configurable because local CPU embedding endpoints often need
smaller batches than cloud APIs (lesson from open-notebook embedding.py).
"""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np

from ragcore.chunking import chunk_text, token_count
from ragcore.providers import build_embedding_model
from ragcore.routing import select_model

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


def _get_embedder(config):
    """Resolve the embedding model for the configured role (patched in tests)."""
    provider, model = select_model(config, "embedding")
    return build_embedding_model(provider, model)


def _mean_pool(vectors: list[list[float]]) -> list[float]:
    arr = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    mean = np.mean(arr / norms, axis=0)
    n = np.linalg.norm(mean)
    if n > 0:
        mean = mean / n
    return mean.tolist()


async def generate_embeddings(
    texts: list[str], config, batch_size: int = 50
) -> list[list[float]]:
    if not texts:
        return []
    embedder = _get_embedder(config)
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                out.extend(await embedder.aembed(batch))
                break
            except Exception:
                if attempt == _MAX_RETRIES:
                    raise
                await asyncio.sleep(_RETRY_DELAY)
    return out


async def generate_embedding(
    text: str, config, chunk_size: int = 400, file_path: Optional[str] = None
) -> list[float]:
    text = text.strip()
    if not text:
        raise ValueError("Cannot embed empty text")
    if token_count(text) <= chunk_size:
        return (await generate_embeddings([text], config))[0]
    chunks = chunk_text(text, chunk_size=chunk_size, file_path=file_path)
    vecs = await generate_embeddings(chunks, config)
    return _mean_pool(vecs) if len(vecs) > 1 else vecs[0]
