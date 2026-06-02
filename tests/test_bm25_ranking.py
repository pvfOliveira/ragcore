"""Regression tests for BM25 / hybrid ranking on a multi-document corpus.

Background: on a single-document corpus, SurrealDB's BM25 `search::score` collapses
to 0.0 (the IDF term has no signal when every term is in the only document). These
tests confirm that on a realistic multi-doc corpus the scores are non-zero and
discriminating, and that `hybrid_search` ranks a relevant document above unrelated
ones. See `scripts/validate_bm25.py` for the larger (50-doc) empirical benchmark.

Requires the `surreal` binary (the `surreal_url` fixture skips if it is absent).
"""
import pytest

from ragcore.config import SurrealConfig
from ragcore.retrieve import hybrid_search
from ragcore.store import Store

_TARGET = "SurrealDB is a multi-model database storing documents and vector embeddings."
_DB_RELATED = [
    "A relational database arranges data into tables with rows and columns.",
    "Graph databases model relationships as edges between nodes.",
]
_UNRELATED = [
    "The telescope captured a faint galaxy billions of light years away.",
    "Sourdough bread relies on a wild yeast starter fermented for days.",
    "The violinist tuned her strings before the symphony began.",
    "Compound interest makes investments grow exponentially over time.",
    "Photosynthesis converts sunlight and water into glucose.",
    "The marathon runner paced herself through the final miles.",
    "Coral reefs host an extraordinary diversity of marine life.",
    "Quantum computers exploit superposition to process information.",
    "Bees pollinate flowers while gathering nectar for honey.",
]


@pytest.fixture()
async def populated_store(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    store = Store(cfg)
    await store.init_schema()
    # The target doc's embedding is aligned with the query embedding ([1,0,0]); every
    # other doc is orthogonal ([0,1,0]) so the target wins on BOTH vector and keyword.
    docs = [(_TARGET, [1.0, 0.0, 0.0])]
    docs += [(c, [0.0, 1.0, 0.0]) for c in _DB_RELATED + _UNRELATED]
    for i, (content, emb) in enumerate(docs):
        sid = await store.create_source(title=f"doc{i}", full_text=content, origin=f"doc{i}")
        await store.add_embeddings(sid, [{"order": 0, "content": content, "embedding": emb}])
    return store


async def test_bm25_relevance_is_nonzero_and_ordered(populated_store):
    # 'database' (incl. stemmed 'databases') appears in only 3 of 12 docs, so BM25
    # IDF yields positive, discriminating scores — not the N==1 collapse to 0.0.
    results = await populated_store.text_search("database", k=10)

    assert len(results) >= 3
    assert all(r["relevance"] > 0 for r in results), "BM25 scores collapsed to zero"
    rels = [r["relevance"] for r in results]
    assert rels == sorted(rels, reverse=True), "results not ordered by relevance"
    assert all("telescope" not in r["content"] for r in results), "unrelated doc leaked in"


async def test_hybrid_search_ranks_relevant_above_unrelated(populated_store):
    async def embedder_fn(query: str):
        return [1.0, 0.0, 0.0]  # aligned with the target doc's embedding

    fused = await hybrid_search(populated_store, embedder_fn, "database", k=20)

    assert fused, "expected fused results"
    positions = {r["content"]: i for i, r in enumerate(fused)}
    # The top result must be a genuinely relevant (database-mentioning) doc. We do NOT
    # assert _TARGET is exactly rank 0: the two other database docs are also relevant and
    # can interleave, and the zero-cosine docs' vector tie-order is non-deterministic.
    relevant = {_TARGET, *_DB_RELATED}
    assert fused[0]["content"] in relevant, "an unrelated doc ranked first"
    # every unrelated doc that surfaced must rank strictly below the target
    for u in _UNRELATED:
        if u in positions:
            assert positions[u] > positions[_TARGET]
