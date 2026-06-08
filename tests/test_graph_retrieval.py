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


@pytest.mark.asyncio
async def test_graph_context_traversal_surfaces_neighbor_chunk(surreal_config):
    """graph_context surfaces a chunk reachable ONLY via a neighbor entity, not a direct seed.

    Setup:
    - S1 has chunk "the eiffel tower stands tall"
    - S2 has chunk "a large country in western europe"
    - upsert_triples(S1, [Paris→located_in→France]):
        both Paris and France get S1 as source; edge Paris→France created.
    - upsert_triples(S2, [France→part_of→Europe]):
        France gains S2 as an additional source; Europe gets S2.
        Critically, Paris is NOT a subject or object here, so Paris.sources stays {S1} only.

    Query: "paris"
    - Seed: entity "paris" (norm len=5 >= 3, matches query); Paris.sources = {S1}
    - Step 2 (own sources): collects S1 → "the eiffel tower stands tall" surfaces.
    - Step 3 (traversal Paris→France): France.sources = {S1, S2} → adds S2.
    - S2's chunk "a large country in western europe" surfaces ONLY via the traversal leg.
    - Without traversal, only S1's chunk would appear (Paris.sources = {S1} only).
    """
    from ragcore.config import Config, GraphConfig, ModelRole
    from ragcore.graph import GraphStore, graph_context
    from ragcore.store import Store

    cfg = Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="m"),
            "embedding": ModelRole(local_provider="ollama", local_model="e"),
        },
        surreal=surreal_config,
        graph=GraphConfig(enabled=True, hops=1),
    )

    store = Store(surreal_config)

    # S1: chunk content "the eiffel tower stands tall"
    s1_id = await store.create_source("Source1", "the eiffel tower stands tall", "file://s1.txt")
    await store.add_embeddings(s1_id, [
        {"order": 0, "content": "the eiffel tower stands tall", "embedding": [0.0] * 768}
    ])

    # S2: chunk content "a large country in western europe"
    s2_id = await store.create_source("Source2", "a large country in western europe", "file://s2.txt")
    await store.add_embeddings(s2_id, [
        {"order": 0, "content": "a large country in western europe", "embedding": [0.0] * 768}
    ])

    gs = GraphStore(surreal_config)
    # Paris→located_in→France under S1 (both Paris and France get S1 as source)
    await gs.upsert_triples(s1_id, [
        {"subject": "Paris", "predicate": "located_in", "object": "France"},
    ])
    # France→part_of→Europe under S2 (France gains S2; Europe gets S2; Paris untouched)
    await gs.upsert_triples(s2_id, [
        {"subject": "France", "predicate": "part_of", "object": "Europe"},
    ])

    chunks = await graph_context(cfg, "paris", k=10)

    contents = [c["content"] for c in chunks]
    # S2's chunk must appear — it is reachable ONLY via the France neighbor of seed Paris.
    # Paris.sources = {S1} only; S2 is attached only to France (and Europe), so it can
    # only surface through the traversal leg. Removing step 3 in graph_context causes
    # this assertion to fail.
    assert any("a large country in western europe" in c for c in contents), (
        f"Expected S2's traversal-only chunk to appear; got contents: {contents}"
    )
    # S1's chunk should also appear (seed's own sources)
    assert any("eiffel tower" in c for c in contents), (
        f"Expected S1's direct-seed chunk to appear; got contents: {contents}"
    )
