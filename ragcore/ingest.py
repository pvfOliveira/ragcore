"""Synchronous ingestion: extract -> chunk -> embed -> store."""
from __future__ import annotations

from content_core import extract_content

from ragcore.chunking import chunk_text
from ragcore.embedding import generate_embeddings


async def _extract(path_or_url: str) -> dict:
    """Extract content to markdown via content-core. Returns title/content/origin."""
    is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
    request: dict = {"output_format": "markdown"}
    if is_url:
        request["url"] = path_or_url
    else:
        request["file_path"] = path_or_url
    state = await extract_content(request)
    content = getattr(state, "content", "") or ""
    title = getattr(state, "title", None) or path_or_url
    return {"title": title, "content": content, "origin": path_or_url}


async def ingest_source(
    path_or_url: str, store, config, chunk_size: int = 400
) -> str:
    extracted = await _extract(path_or_url)
    content = extracted["content"]
    if not content.strip():
        raise ValueError(f"No content extracted from {path_or_url}")

    source_id = await store.create_source(
        title=extracted["title"], full_text=content, origin=extracted["origin"],
    )
    chunks = chunk_text(content, chunk_size=chunk_size, file_path=path_or_url)
    vectors = await generate_embeddings(chunks, config)
    rows = [
        {"order": i, "content": c, "embedding": v}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    await store.add_embeddings(source_id, rows)
    return source_id
