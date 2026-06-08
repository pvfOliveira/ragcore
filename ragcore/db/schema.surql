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
    COLUMNS full_text FULLTEXT ANALYZER ragcore_analyzer BM25 HIGHLIGHTS;
DEFINE INDEX IF NOT EXISTS idx_source_emb_content ON TABLE source_embedding
    COLUMNS content FULLTEXT ANALYZER ragcore_analyzer BM25 HIGHLIGHTS;

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

DEFINE TABLE IF NOT EXISTS ingestion_job SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS origin ON TABLE ingestion_job TYPE string;
DEFINE FIELD IF NOT EXISTS status ON TABLE ingestion_job TYPE string DEFAULT 'queued';
DEFINE FIELD IF NOT EXISTS source_id ON TABLE ingestion_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS error ON TABLE ingestion_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS attempts ON TABLE ingestion_job TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS next_attempt_at ON TABLE ingestion_job TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS created ON ingestion_job DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD IF NOT EXISTS updated ON ingestion_job DEFAULT time::now() VALUE time::now();

DEFINE TABLE IF NOT EXISTS chat_session SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS title ON TABLE chat_session TYPE option<string>;
DEFINE FIELD IF NOT EXISTS created ON chat_session DEFAULT time::now() VALUE $before OR time::now();
DEFINE FIELD IF NOT EXISTS updated ON chat_session DEFAULT time::now() VALUE time::now();

DEFINE TABLE IF NOT EXISTS chat_message SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS session ON TABLE chat_message TYPE record<chat_session>;
DEFINE FIELD IF NOT EXISTS seq ON TABLE chat_message TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS role ON TABLE chat_message TYPE string;
DEFINE FIELD IF NOT EXISTS content ON TABLE chat_message TYPE string;
DEFINE FIELD IF NOT EXISTS citations ON TABLE chat_message TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS created ON chat_message DEFAULT time::now() VALUE $before OR time::now();

DEFINE EVENT IF NOT EXISTS chat_session_delete ON TABLE chat_session WHEN ($after == NONE) THEN {
    DELETE chat_message WHERE session == $before.id;
};

DEFINE TABLE IF NOT EXISTS eval_run SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS tag ON TABLE eval_run TYPE option<string>;
DEFINE FIELD IF NOT EXISTS dataset ON TABLE eval_run TYPE option<string>;
DEFINE FIELD IF NOT EXISTS config_snapshot ON TABLE eval_run TYPE string;
DEFINE FIELD IF NOT EXISTS metrics ON TABLE eval_run TYPE string;
DEFINE FIELD IF NOT EXISTS gate_passed ON TABLE eval_run TYPE option<bool>;
DEFINE FIELD IF NOT EXISTS created ON eval_run DEFAULT time::now() VALUE $before OR time::now();

DEFINE TABLE IF NOT EXISTS deployment SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS run_id ON TABLE deployment TYPE string;
DEFINE FIELD IF NOT EXISTS history ON TABLE deployment TYPE option<array<string>>;
DEFINE FIELD IF NOT EXISTS updated ON deployment DEFAULT time::now() VALUE time::now();
