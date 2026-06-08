"""Synchronous ingestion: extract -> chunk -> embed -> store."""
from __future__ import annotations

from dataclasses import dataclass

from content_core import extract_content

from ragcore.chunking import chunk_text
from ragcore.embedding import generate_embeddings


@dataclass
class IngestResult:
    source_id: str
    created: bool  # False => skipped because this origin was already ingested


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
) -> IngestResult:
    existing = await store.find_source_id_by_origin(path_or_url)
    if existing:
        return IngestResult(source_id=existing, created=False)
    extracted = await _extract(path_or_url)
    content = extracted["content"]
    if not content.strip():
        raise ValueError(f"No content extracted from {path_or_url}")
    source_id = await store.create_source(
        title=extracted["title"], full_text=content, origin=extracted["origin"],
    )
    chunks = chunk_text(content, chunk_size=chunk_size, file_path=path_or_url)
    if not chunks:
        raise ValueError(f"Content from {path_or_url} produced no chunks to embed")
    vectors = await generate_embeddings(chunks, config)
    rows = [
        {"order": i, "content": c, "embedding": v}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    await store.add_embeddings(source_id, rows)
    # Mirror the embeddings into the configured pluggable vector backend, if any.
    # SurrealDB (default) keeps being the source of truth and is never routed
    # through an adapter; only non-surreal backends get a second write.
    if config is not None and config.store.vector_backend != "surreal":
        from ragcore.vectorstores.base import make_vector_store

        vs = make_vector_store(config)
        adapter_chunks = [
            {"id": f"{source_id}:{r['order']}", "text": r["content"], "embedding": r["embedding"]}
            for r in rows
        ]
        await vs.add_embeddings(source_id, adapter_chunks)
    # Graph extraction: extract entity/relation triples from each chunk and persist
    # them as a native SurrealDB graph. Opt-in via config.graph.enabled (default False).
    # Failures here must NEVER fail ingestion.
    if config is not None and getattr(config, "graph", None) is not None and config.graph.enabled:
        from ragcore.graph import GraphStore, _chat_fn, extract_triples
        chat_fn = _chat_fn(config)
        gstore = GraphStore(config.surreal)
        for chunk in chunks:
            try:
                triples = await extract_triples(chunk, chat_fn)
                if triples:
                    await gstore.upsert_triples(source_id, triples)
            except Exception:
                pass  # extraction/graph failures must NEVER fail ingestion
    return IngestResult(source_id=source_id, created=True)
