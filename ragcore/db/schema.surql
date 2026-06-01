DEFINE TABLE IF NOT EXISTS source SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS title ON TABLE source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS full_text ON TABLE source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS origin ON TABLE source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created ON source DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD IF NOT EXISTS updated ON source DEFAULT time::now() VALUE time::now();

DEFINE TABLE IF NOT EXISTS source_embedding SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS source ON TABLE source_embedding TYPE record<source>;
DEFINE FIELD IF NOT EXISTS order ON TABLE source_embedding TYPE int;
DEFINE FIELD IF NOT EXISTS content ON TABLE source_embedding TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON TABLE source_embedding TYPE array<float>;

DEFINE EVENT IF NOT EXISTS source_delete ON TABLE source WHEN ($after == NONE) THEN {
    DELETE source_embedding WHERE source == $before.id;
};

DEFINE ANALYZER IF NOT EXISTS ragcore_analyzer
    TOKENIZERS blank,class,camel,punct
    FILTERS snowball(english), lowercase;

DEFINE INDEX IF NOT EXISTS idx_source_full_text ON TABLE source
    COLUMNS full_text SEARCH ANALYZER ragcore_analyzer BM25 HIGHLIGHTS;
DEFINE INDEX IF NOT EXISTS idx_source_emb_content ON TABLE source_embedding
    COLUMNS content SEARCH ANALYZER ragcore_analyzer BM25 HIGHLIGHTS;

-- Full-text search over chunk content; returns chunk rows with BM25 relevance.
DEFINE FUNCTION IF NOT EXISTS fn::text_search($q: string, $k: int) {
    RETURN (
        SELECT id, source, content, search::score(1) AS relevance
        FROM source_embedding
        WHERE content @1@ $q
        ORDER BY relevance DESC
        LIMIT $k
    );
};

-- Brute-force cosine similarity over all chunks (mirrors open-notebook's approach;
-- swap to an MTREE/HNSW KNN index here if the corpus outgrows a full scan).
DEFINE FUNCTION IF NOT EXISTS fn::vector_search($query: array<float>, $k: int) {
    RETURN (
        SELECT id, source, content,
               vector::similarity::cosine(embedding, $query) AS similarity
        FROM source_embedding
        ORDER BY similarity DESC
        LIMIT $k
    );
};
