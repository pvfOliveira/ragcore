"""Empirical benchmark: BM25 ranking quality on a realistic multi-document corpus.

This is the reproducible artifact for the "validate BM25 ranking on a realistic
corpus" follow-up. It ingests ~50 topically-diverse documents into a SurrealDB
instance and prints `search::score` for several queries.

WHY: on a single-document corpus, SurrealDB's BM25 `search::score` returns 0.0 —
the IDF term has no signal when there is only one document. The concern was that
hybrid retrieval might therefore rank by noise. This benchmark confirms that on a
realistic corpus the scores are non-zero and discriminating.

FINDINGS (observed on SurrealDB 3.1.2, 50-doc corpus):
    text_search('telescope')  -> 3.40   (rare term: 1 doc  -> high IDF)
    text_search('database')   -> 2.27 .. 1.90  (in ~4 docs -> ranked sensibly)
    text_search('embeddings') -> 2.88 / 2.41
    text_search('the')        -> ~0.21 (common term: in most docs -> IDF ~ 0)

CONCLUSION: BM25 scores are meaningful on real corpora; the 0.0 seen earlier was
purely the N==1 IDF-collapse artifact. `hybrid_search` fuses by RANK position
(Reciprocal Rank Fusion), not raw score, so it is robust even to low common-term
scores. No code change is required. Regression coverage lives in
`tests/test_bm25_ranking.py`.

USAGE:
    # start a throwaway DB in another terminal:
    surreal start --user root --pass root rocksdb:/tmp/bm25db --bind 127.0.0.1:8011
    # then:
    .venv/bin/python scripts/validate_bm25.py ws://127.0.0.1:8011/rpc
"""
import asyncio
import sys

from ragcore.config import SurrealConfig
from ragcore.store import Store

CORPUS = [
    "SurrealDB is a multi-model database that stores documents, graphs, and vector embeddings together.",
    "A relational database organizes data into tables with rows and columns and enforces schemas.",
    "Vector databases index high-dimensional embeddings to power semantic similarity search.",
    "The telescope captured a faint galaxy billions of light years from Earth.",
    "Astronomers discovered a new exoplanet orbiting a distant red dwarf star.",
    "The black hole at the center of the Milky Way is called Sagittarius A star.",
    "Sourdough bread relies on a wild yeast starter fermented over several days.",
    "Caramelizing onions slowly brings out a deep, sweet, savory flavor.",
    "A good risotto requires gradually adding warm stock while stirring constantly.",
    "The violinist tuned her strings before the orchestra began the symphony.",
    "Jazz improvisation blends melody, rhythm, and spontaneous harmonic invention.",
    "The drummer kept a steady backbeat throughout the entire concert.",
    "Compound interest causes investments to grow exponentially over long horizons.",
    "Diversifying a portfolio across asset classes reduces overall risk.",
    "Central banks adjust interest rates to manage inflation and employment.",
    "The marathon runner paced herself carefully through the final miles.",
    "Swimming engages nearly every major muscle group with low joint impact.",
    "The cyclist climbed the steep mountain pass in the final stage of the race.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "Mitochondria are the energy-producing organelles inside eukaryotic cells.",
    "Coral reefs host an extraordinary diversity of marine life.",
    "The volcano erupted, sending ash plumes high into the atmosphere.",
    "Glaciers carve deep valleys as they slowly advance over millennia.",
    "Tectonic plates shift gradually, causing earthquakes along fault lines.",
    "The novelist spent years drafting and revising the sprawling epic.",
    "Poetry often compresses vivid imagery into a few carefully chosen words.",
    "The library archived thousands of rare manuscripts and first editions.",
    "Watercolor painting depends on controlling the flow of pigment and water.",
    "The sculptor chiseled the marble block into a lifelike human figure.",
    "Renaissance frescoes adorned the ceilings of grand Italian cathedrals.",
    "Quantum computers exploit superposition and entanglement to process information.",
    "A compiler translates high-level source code into machine instructions.",
    "Encryption protects data confidentiality by scrambling it with keys.",
    "The neural network learned to classify images after training on examples.",
    "Distributed systems coordinate many machines to act as a single service.",
    "The chess grandmaster sacrificed a knight to launch a decisive attack.",
    "Gardeners rotate crops each season to keep the soil healthy.",
    "Bees pollinate flowers while gathering nectar to make honey.",
    "The hummingbird beats its wings dozens of times per second.",
    "Migratory birds navigate thousands of miles using the Earth magnetic field.",
    "The locomotive pulled a long line of freight cars across the prairie.",
    "Electric vehicles store energy in large lithium-ion battery packs.",
    "The bridge suspension cables bear the enormous weight of the deck.",
    "Skyscrapers use steel frames to support their towering height.",
    "The chef plated the dessert with a delicate drizzle of raspberry coulis.",
    "Tea leaves are oxidized to different degrees to make green, oolong, or black tea.",
    "The hikers pitched their tent beside a clear alpine lake.",
    "A graph database represents relationships as first-class edges between nodes.",
    "Full-text search engines rank documents using term frequency and inverse document frequency.",
    "The meteor shower peaked just after midnight under a moonless sky.",
]


async def main() -> None:
    if len(sys.argv) < 2:
        print("usage: validate_bm25.py <surreal-ws-url>")
        raise SystemExit(2)
    cfg = SurrealConfig(url=sys.argv[1], namespace="ragcore", database="ragcore",
                        user="root", password="root")
    store = Store(cfg)
    await store.init_schema()

    for i, content in enumerate(CORPUS):
        sid = await store.create_source(title=f"doc{i}", full_text=content, origin=f"doc{i}")
        await store.add_embeddings(sid, [{"order": 0, "content": content,
                                          "embedding": [0.1, 0.2, 0.3]}])
    print(f"Ingested {len(CORPUS)} chunks.\n")

    for term in ["database", "telescope", "search", "the", "embeddings"]:
        res = await store.text_search(term, k=5)
        print(f"=== text_search({term!r}) -> {len(res)} hits ===")
        for r in res:
            print(f"  rel={r.get('relevance')!s:>12}  {r['content'][:64]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
