"""Tests for graph-augmented retrieval: fusion and real-DB graph_context."""
from __future__ import annotations

import pytest

from ragcore.retrieve import fuse_with_graph


def test_graph_chunks_fused_as_third_list():
    vector = [{"id": "a", "content": "alpha", "source": "s1"},
              {"id": "b", "content": "beta", "source": "s1"}]
    bm25   = [{"id": "b", "content": "beta", "source": "s1"},
              {"id": "c", "content": "gamma", "source": "s2"}]
    graph  = [{"id": "d", "content": "delta (graph-only)", "source": "s3"}]
    fused = fuse_with_graph(vector, bm25, graph, k=4)
    ids = [r["id"] for r in fused]
    assert "d" in ids               # graph-only chunk surfaces
    assert "b" in ids               # in two lists -> ranks well
    assert len(fused) <= 4


def test_fuse_with_graph_empty_graph_matches_two_list():
    from ragcore.retrieve import reciprocal_rank_fusion
    vector = [{"id": "a", "content": "x", "source": "s"}]
    bm25 = [{"id": "b", "content": "y", "source": "s"}]
    fused = fuse_with_graph(vector, bm25, [], k=10)
    base = reciprocal_rank_fusion([vector, bm25], key="id", k=60)[:10]
    assert [r["id"] for r in fused] == [r["id"] for r in base]


# ---------------------------------------------------------------------------
# Real-DB test for graph_context (gated on surreal_config fixture)
# ---------------------------------------------------------------------------

@pytest.fixture()
async def surreal_config(surreal_url):
    from ragcore.config import SurrealConfig
    from ragcore.store import Store

    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()
    yield cfg


@pytest.mark.asyncio
async def test_graph_context_returns_connected_chunks(surreal_config):
    """graph_context returns chunks from sources linked to entities that match the query."""
    from ragcore.graph import GraphStore, graph_context
    from ragcore.store import Store

    # Build a minimal Config-like object with the graph and surreal fields
    class FakeCfg:
        surreal = surreal_config

        class graph:
            enabled = True
            hops = 1

    cfg = FakeCfg()

    # Insert a source_embedding row directly (no real embedding — use zeroes)
    store = Store(surreal_config)

    # Create a source record so the FK link is valid
    source_id = await store.create_source("Test Doc", "full text here", "file://test.txt")
    # Add a chunk with a zero embedding (embedding search won't be used here)
    await store.add_embeddings(source_id, [
        {"order": 0, "content": "Paris is the capital of France.", "embedding": [0.0] * 768}
    ])

    # Build the graph: link entity "Paris" to this source
    gs = GraphStore(surreal_config)
    await gs.upsert_triples(source_id, [
        {"subject": "Paris", "predicate": "is capital of", "object": "France"},
    ])

    # Now call graph_context with a query mentioning "paris"
    chunks = await graph_context(cfg, "what do you know about paris?", k=10)

    # The chunk linked to the "Paris" entity's source must surface
    assert any("Paris" in c["content"] or "France" in c["content"] for c in chunks), (
        f"Expected Paris/France chunk; got {chunks}"
    )


@pytest.mark.asyncio
async def test_graph_context_returns_empty_on_no_match(surreal_config):
    """graph_context returns [] when no entity matches the query terms."""
    from ragcore.graph import graph_context

    class FakeCfg:
        surreal = surreal_config

        class graph:
            enabled = True
            hops = 1

    chunks = await graph_context(FakeCfg(), "zzz nonexistent xyzzy", k=10)
    assert chunks == []
