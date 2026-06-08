import pytest

from ragcore.config import SurrealConfig
from ragcore.graph import GraphStore
from ragcore.store import Store


@pytest.fixture()
async def surreal_config(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()
    yield cfg


@pytest.mark.asyncio
async def test_upsert_and_traverse(surreal_config):
    gs = GraphStore(surreal_config)
    await gs.upsert_triples("source:doc1", [
        {"subject": "Ada Lovelace", "predicate": "wrote", "object": "the first algorithm"},
    ])
    # traverse: the subject entity should reach the object entity via ->relates->entity
    objs = await gs.neighbors("Ada Lovelace")
    names = [o["name"] for o in objs]
    assert "the first algorithm" in names


@pytest.mark.asyncio
async def test_upsert_dedups_entities(surreal_config):
    gs = GraphStore(surreal_config)
    await gs.upsert_triples("source:d1", [{"subject": "A", "predicate": "rel", "object": "B"}])
    await gs.upsert_triples("source:d2", [{"subject": "A", "predicate": "rel2", "object": "C"}])
    # 'A' must be a single deduped entity (norm unique), now with two outgoing edges
    objs = await gs.neighbors("A")
    names = sorted(o["name"] for o in objs)
    assert names == ["B", "C"]


@pytest.mark.asyncio
async def test_reingest_same_triple_dedups_edge(surreal_config):
    gs = GraphStore(surreal_config)
    await gs.upsert_triples("source:d1", [{"subject": "A", "predicate": "rel", "object": "B"}])
    await gs.upsert_triples("source:d2", [{"subject": "A", "predicate": "rel", "object": "B"}])
    # neighbors still returns B exactly once (single edge, deduped)
    objs = await gs.neighbors("A")
    assert [o["name"] for o in objs] == ["B"]
    # and the single edge accumulated both sources — assert via a direct count helper
    edges = await gs.edge_count("A", "B")
    assert edges == 1


@pytest.mark.asyncio
async def test_neighbors_two_hops(surreal_config):
    gs = GraphStore(surreal_config)
    await gs.upsert_triples("source:d1", [
        {"subject": "A", "predicate": "r1", "object": "B"},
        {"subject": "B", "predicate": "r2", "object": "C"},
    ])
    two = await gs.neighbors("A", hops=2)
    names = [o["name"] for o in two]
    assert "C" in names      # reachable in 2 hops
