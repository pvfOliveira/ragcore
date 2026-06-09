# ragcore — Architecture Deep Dive

*A Senior Staff Engineer's critical walkthrough. Everything below is grounded in code that actually exists in this repo; file:line citations point at the real thing. The story: a clean local-first RAG core that grew into a full local LLM platform.*

---

## 1. What it is

`ragcore` is a local-first Retrieval-Augmented Generation engine that answers questions over your own documents using only hardware you control. Feed it files or URLs; it extracts, chunks, embeds, and stores them in a single SurrealDB instance; then it answers questions with a local LLM (Ollama), fusing vector similarity and BM25 full-text search via Reciprocal Rank Fusion, optionally enriched by a third knowledge-graph signal. It ships as a Typer CLI, a FastAPI + vanilla-JS web UI, and the importable `ragcore` Python package — three faces over one library. Over one development sprint it grew a production-rag upgrade layer (pluggable vector backends, FlashRank reranking, semantic caching, Ragas+TruLens eval), and then a full LLM platform layer: llmops lifecycle (eval registry, gate, drift, promote/rollback), AI cost optimization (SQLite token ledger), graph-RAG (LLM triple extraction to SurrealDB graph), multimodal CLIP ingest, and an Ollama serving benchmark. It is deliberately single-user, single-machine, and sized to fit in one developer's head.

---

## 2. The big picture

Think of ragcore as a **personal research librarian working at a single desk, who over time became the department's data science lead**. When you hand over a document (`ingest`), the librarian photocopies it into index cards (chunks), files each card under two separate catalog systems — a "meaning" catalog (vector embeddings) and a "keyword" catalog (BM25 full-text) — and also draws a relationship map between the entities mentioned (graph-RAG). Everything goes into one filing cabinet (SurrealDB). When you ask a question (`ask`), the librarian plans which lookups to run, pulls cards from *all three* catalogs in parallel, reconciles the ranked piles via Reciprocal Rank Fusion, and writes a cited answer. If the answer is nearly the same as one they wrote yesterday, they hand you that one instead (semantic cache). And in the expanded role: the librarian now grades their own work with independent judges (Ragas, TruLens), tracks spending on expensive consultants (cost ledger), and keeps a deployment pointer so you can roll back to the version of themselves that scored better (llmops deploy/rollback).

The mental model that matters: **SurrealDB is the one cabinet that holds everything** — documents, embeddings, the job queue, chat history, eval runs, deployment pointer, entity/relation graph, and image embeddings. There is no message broker, no second datastore for the platform layer except two purpose-sized SQLite files (cost ledger and semantic cache). The "smart" parts — which model, local vs. cloud — are injected, not hardwired, so adding a new provider is a one-file change.

---

## 3. Architecture & data flow

```
               ┌─────────────┐  ┌──────────────────────┐  ┌────────────────┐
  USER  ──►    │  cli.py     │  │  web/app.py (FastAPI) │  │  worker.py     │
               │  (Typer)    │  │  + static/index.html  │  │  (bg loop)     │
               └──────┬──────┘  └──────────┬────────────┘  └───────┬────────┘
                      │                    │                        │
                      └──────────┬─────────┴───────────┬───────────┘
                                 ▼                     ▼
          ┌──────────────────────── ragcore library ─────────────────────────┐
          │  CORE LAYER                                                        │
          │  ingest.py  ──►  chunking.py  embedding.py  store.py              │
          │  retrieve.py (RRF + graph_context)                                 │
          │  ask.py (LangGraph: strategy → fan-out → synthesize)              │
          │  chat.py (linear, history-aware)                                   │
          │  routing.py  providers.py (esperanto AIFactory)                   │
          │  cache.py (SemanticCache/SQLite)  rerank.py (FlashRank)           │
          │                                                                    │
          │  RAG UPGRADE LAYER                                                 │
          │  vectorstores/base.py (VectorStore protocol)                      │
          │  vectorstores/chroma_store.py  faiss_store.py  milvus_store.py    │
          │                                                                    │
          │  PLATFORM LAYER                                                    │
          │  graph.py (extract_triples, GraphStore, graph_context)            │
          │  multimodal.py (ClipEmbedder, ImageStore)                         │
          │  llmops/ (registry.py  gates.py  deploy.py)                       │
          │  cost/ (ledger.py  report.py)                                      │
          │  eval/ (harness.py — Ragas + TruLens)                             │
          │  bench/ (harness.py — Ollama sweep; serving.py — vllm stub)       │
          └────────────────┬──────────────────────────┬───────────────────────┘
                           │                          │
              ┌────────────▼──────────────┐   ┌───────▼──────────────────────┐
              │    SurrealDB 3.x           │   │  Ollama (local LLM)          │
              │    rocksdb:./data/db       │   │  qwen3:8b / nomic-embed-text │
              │    ─────────────────────  │   │  + cloud (Anthropic optional) │
              │    source                  │   └──────────────────────────────┘
              │    source_embedding        │
              │    ingestion_job           │   ┌──────────────────────────────┐
              │    chat_session/message    │   │  SQLite (stdlib, 2 files)    │
              │    eval_run                │   │  data/cost.db (token ledger) │
              │    deployment:current      │   │  data/semantic_cache.db      │
              │    entity / relates        │   └──────────────────────────────┘
              │    image_embedding         │
              │    fn::vector_search (BF)  │   ┌──────────────────────────────┐
              │    fn::text_search (BM25)  │   │  External adapters (opt-in)  │
              │    fn::image_search        │   │  data/chroma/  (ChromaDB)    │
              └────────────────────────────┘   │  data/faiss/   (IndexFlatIP) │
                                               │  data/milvus.db (Lite)       │
                                               └──────────────────────────────┘
```

### Walking an `ingest` (synchronous path)

All paths: `ragcore/ingest.py:35` → `ingest_source`.

1. **Dedup check.** `store.find_source_id_by_origin` (`store.py:81`) — if this exact file path or URL exists, return `created=False` immediately. No double-indexing.
2. **Extraction.** `_extract` (`ingest.py:20`) calls `content_core.extract_content` to produce markdown + title. Works on local files and URLs alike.
3. **Chunking.** `chunk_text` (`chunking.py:80`) recursively splits on `\n\n` → `\n` → `. ` → ` ` boundaries, measured in tiktoken tokens, targeting 400 tokens with 60-token overlap. Tiny fragments (< 5 tokens) are dropped.
4. **Embedding.** `generate_embeddings` (`embedding.py:38`) batches chunks (50 at a time) through the configured embedding model with 3× exponential retry.
5. **Storage.** `store.create_source` writes the `source` row; `store.add_embeddings` (`store.py:62`) writes one `source_embedding` row per chunk (content + embedding vector). A `DEFINE EVENT` in the schema (`schema.surql:14`) cascades chunk deletion when a source is removed.
6. **Mirror write.** If a non-surreal vector backend is configured (`config.store.vector_backend != "surreal"`), the same chunks are mirrored to that adapter (`ingest.py:60`). SurrealDB remains the single source of truth.
7. **Graph extraction (opt-in).** If `config.graph.enabled`, each chunk is passed to `extract_triples` (`graph.py:33`) — an LLM call returning subject/predicate/object triples — and results are upserted into `entity`/`relates` tables via `GraphStore.upsert_triples` (`graph.py:133`). Failures are caught and logged; they never abort ingestion (`ingest.py:82`).

For async ingestion (`ragcore ingest --async`), steps 2–7 are deferred: `JobQueue.enqueue` (`jobs.py:38`) writes a row to `ingestion_job`, and a separate `ragcore worker` process claims and runs it with exponential backoff (`worker.py:18`).

### Walking an `ask` (the LangGraph)

Built in `ask.py:98` as a compiled `StateGraph`. All state is typed via `AskState` (`ask.py:46`).

```
START → strategy → (Send × N terms) → retrieve_answer (parallel) → synthesize → END
```

1. **Semantic cache check (pre-graph).** If `config.cache.enabled`, the question is embedded and compared against all cached query embeddings via `SemanticCache.get` (`cache.py:47`) — a full linear scan with cosine similarity. A hit above threshold (`default 0.95`) short-circuits the entire graph and returns the cached answer (`ask.py:163`).
2. **Budget enforcement.** If cost enforcement is on, `over_budget` (`cost/ledger.py:81`) checks the question token count against the configured limit before any LLM call (`ask.py:149`).
3. **strategy node** (`ask.py:56`): the chat model decomposes your question into up to 5 focused search terms. Prompt: `prompts/ask_strategy.jinja`. JSON parse failure falls back to `[question]`.
4. **fan-out** (`ask.py:70`): `_fan_out` emits one `Send("retrieve_answer", ...)` per term — these run concurrently. Results accumulate via `Annotated[list, operator.add]`, LangGraph's reducer.
5. **retrieve_answer node** (`ask.py:72`, once per term): `hybrid_search` (`retrieve.py:64`) runs vector + BM25 (+ optional graph signal via `graph_context`) and fuses them with RRF. If `config.rerank.enabled`, FlashRank cross-encoder reranks the fused candidates. The chat model drafts a partial answer over the top chunks. Prompt: `prompts/ask_answer.jinja`.
6. **synthesize node** (`ask.py:89`): one final model call merges partial answers into a cited response. Citations are deduped source IDs. Prompt: `prompts/ask_final.jinja`.
7. **Cache write + cost record.** Answer + embeddings are stored in the semantic cache. Token counts are written to the cost ledger.

### Walking the llmops cycle

```bash
ragcore eval --tag v2       # run eval, record eval_run in SurrealDB
ragcore gate --baseline v1  # re-run eval, compare metrics, exit 1 if regression
ragcore promote <run-id>    # set deployment:current (refuses if gate not passed)
ragcore rollback            # revert deployment:current to previous history entry
```

`RunRegistry.record` (`llmops/registry.py:27`) serialises metrics + config snapshot + dataset centroid (a mean of question embeddings — the "where in semantic space" fingerprint) as JSON into the `eval_run` SurrealDB table. `check_gate` (`llmops/gates.py:19`) flattens `{family: {metric: float}}` to `"family.metric"` keys and checks each for regression beyond `tolerance`. `check_drift` (`llmops/gates.py:44`) computes cosine distance between current and baseline centroids. `DeploymentStore.promote` (`llmops/deploy.py:43`) reads the `gate_passed` field from the run before overwriting `deployment:current`; it maintains a `history` list enabling one-level rollback.

---

## 4. Codebase tour

Dependency direction is strictly one-way: **cli/web/worker → library → store → SurrealDB/SQLite/adapters**. The library modules never import from cli/web.

### Core layer

| Module | Role |
|---|---|
| `config.py` | Pydantic `Config` parsed from TOML; enforces `[models.chat]` and `[models.embedding]` at load time. Sixteen typed sub-models covering every section. |
| `errors.py` | `RagcoreError` hierarchy + `classify_error` (raw exception → user message) + `is_transient` (retry gate). The split between these two functions is subtle and important — see §7. |
| `chunking.py` | tiktoken `token_count`, content-type detection, recursive token-bounded splitter with overlap prepend. |
| `routing.py` | `select_model(config, role, content, force_cloud) → (provider, model)` — the local↔cloud routing policy, the headline feature. |
| `providers.py` | Thin esperanto `AIFactory` wrapper. `_ChatAdapter` normalises the chat interface; `build_embedding_model` returns an esperanto embedder. This is the one file that knows what esperanto is. |
| `embedding.py` | Batched embeddings with 3× retry and mean-pooling for oversized text. |
| `store.py` | SurrealDB repository: schema init, source CRUD, `vector_search`/`text_search` via DB stored functions. Connection-per-operation. |
| `retrieve.py` | `reciprocal_rank_fusion` + `hybrid_search` + `fuse_with_graph`. The `vector_store_for(config)` factory decides whether vector ranking comes from SurrealDB or a pluggable adapter. |
| `ingest.py` | Extract → chunk → embed → store pipeline; mirrors into pluggable backend + graph extraction if enabled. |
| `ask.py` | Compiled LangGraph: strategy → parallel fan-out → synthesize. Cache and cost wiring live here. |
| `chat.py` | Linear history-aware multi-turn RAG. Deliberately not a graph — it reuses `_build_chat`/`_clean` from `ask.py` to avoid duplication. |
| `sessions.py` | `SessionStore`: chat sessions + messages with cascade-delete event in DB. |
| `jobs.py` | `JobQueue`: SurrealDB-table queue with claim-via-conditional-UPDATE, backoff stored in `next_attempt_at`. |
| `worker.py` | `run_worker` loop: claim → `ingest_source` → mark done/retry/failed. |
| `cache.py` | `SemanticCache`: SQLite + pure-Python cosine, no dependencies. Linear scan at query time. |
| `rerank.py` | FlashRank cross-encoder post-RRF reranking. Injectable `_scorer` for tests. |
| `cli.py` | Typer entrypoint; every command is a thin `asyncio.run(...)` over a library call. Three sub-apps: root, `graph`, `cost`. |
| `web/app.py` | FastAPI app-factory + DI seams (`get_config`, `get_store`, …) + one catch-all exception handler. |
| `db/schema.surql` | All table definitions, BM25 analyzer, `fn::vector_search` (brute-force cosine), `fn::text_search` (BM25), `fn::image_search`, cascade-delete events. |

### RAG upgrade layer

| Module | Role |
|---|---|
| `vectorstores/base.py` | `VectorStore` runtime-checkable Protocol + `make_vector_store(config)` factory. |
| `vectorstores/chroma_store.py` | `ChromaStore`: local persistent ChromaDB, HNSW cosine space. |
| `vectorstores/faiss_store.py` | `FaissStore`: in-process `IndexFlatIP` on L2-normalised vectors, sidecar `meta.json` for id/text. |
| `vectorstores/milvus_store.py` | `MilvusStore`: Milvus Lite local-file mode or remote Milvus/Zilliz URL. |

### Platform layer

| Module | Role |
|---|---|
| `graph.py` | `extract_triples` (LLM JSON), `GraphStore` (entity/`RELATE`-edge CRUD), `graph_context` (entity-seeded graph traversal → chunk retrieval). |
| `multimodal.py` | `ClipEmbedder` (lazy open_clip/torch load, MPS/CUDA/CPU), `ImageStore` (SurrealDB `image_embedding` table), `ingest_image`, `cross_modal_rank`. |
| `llmops/registry.py` | `RunRegistry`: record/list/resolve `eval_run` records in SurrealDB. |
| `llmops/gates.py` | `check_gate` (metric regression test) + `check_drift` (centroid cosine distance). Pure Python, no DB. |
| `llmops/deploy.py` | `DeploymentStore`: `deployment:current` singleton with history list for one-level rollback. |
| `cost/ledger.py` | `CostLedger`: SQLite usage ledger + `aggregate()` + `over_budget()`. |
| `cost/report.py` | `build_report`: spend by model, cache savings, right-sizing hints, bench summary. |
| `eval/harness.py` | `run_eval` (live), `compute_metrics` (stubbed-judge path for deterministic tests), `_OllamaJudge` (litellm + Ragas + TruLens). |
| `bench/harness.py` | `run_bench`: cartesian sweep, `OllamaBenchClient` via raw `urllib.request`. |
| `bench/serving.py` | `vllm_serve`: design-only CUDA-gated stub; raises on no `nvidia-smi`. |

---

## 5. Tech stack & why

**Python 3.11–3.12, uniformly async.** The entire library is `async def`. The CLI bridges with one `asyncio.run` per command. This dodges the nested-loop hazards that plagued earlier sync/async hybrids in the predecessor project.

**SurrealDB 3.x (rocksdb backend) as the single datastore.** Multi-model: document store, vector search (brute-force cosine via `vector::similarity::cosine`), BM25 full-text engine, job queue, chat history, eval run registry, deployment pointer, entity-relation graph, image embeddings — all in one binary, one connection URL, one `surreal start` command. The `DEFINE EVENT` cascade-delete mechanism means the application never has to clean up child rows manually. The cost: it's a process you must run, it's a younger project with sharper edges (see §7), and the vector index is a full scan.

**Ollama (llama.cpp/GGUF under the hood) for local inference.** `qwen3:8b` (or `qwen2.5:7b-instruct`) for chat, `nomic-embed-text` for embeddings. One `ollama pull`, an OpenAI-compatible endpoint, automatic model load/unload. The abstraction means swapping models is a config change.

**esperanto `AIFactory` for provider abstraction** (`providers.py:28`). ragcore never imports `ollama-python` or `anthropic` directly. The thin `_ChatAdapter` wrapper (`providers.py:13`) normalises the interface to `async ainvoke(prompt) → .content`. If esperanto ceased to exist, you'd rewrite exactly one 37-line file.

**LangGraph for the `ask` graph** — specifically for the parallel fan-out with additive reducer (`Annotated[list, operator.add]`, `ask.py:47`) and the `Send` primitive (`ask.py:71`). Notably `chat.py` consciously rejected LangGraph for a linear flow — the right call for a use case that doesn't need it.

**content-core** for file/URL extraction to markdown. **tiktoken** (`o200k_base` encoding) for token-bounded chunking. **ai-prompter + Jinja2** for editable prompt templates under `ragcore/prompts/` — every model-facing string is a `.jinja` file, not a hardcoded literal. **Typer** for the CLI, **FastAPI + uvicorn** for the web layer, **loguru** for structured worker logs, **Pydantic v2** for typed config and request bodies.

**FlashRank** (optional `[rerank]` extra) for cross-encoder reranking post-RRF. **open_clip + torch** (optional `[multimodal]`) for CLIP image embeddings. **Ragas + TruLens + litellm** (optional `[eval]`) for eval metrics with a local Ollama judge.

**SQLite (stdlib)** for the semantic cache and cost ledger. No dependency beyond the standard library — a deliberate choice for components that are opt-in, simple, and single-user.

---

## 6. Architectural review — alternatives matrix

### Category: Primary datastore — chose SurrealDB

SurrealDB does the work of five components: document store, vector index, BM25 FTS, job queue, and graph database, all behind one connection URL. That's the bet.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **PostgreSQL + pgvector + tsvector** | Battle-tested, one DB for docs + vectors + FTS | Mature HNSW/IVFFlat indexes, real ACID transactions, proven ops ecosystem, rich SQL | Heavier to run locally; BM25 requires `pg_bm25`/ParadeDB extension; no native graph; record-link and cascade-delete patterns require triggers or application logic |
| **SQLite + sqlite-vec + FTS5** | True zero-server, single-file, perfect for one user | Embeds in-process (no separate terminal), trivial backup, FTS5 BM25 is excellent, community `sqlite-vec` for ANN | No graph layer; vector support is immature; concurrency story is weak (WAL only); no record links; no DEFINE EVENT equivalent |

**Verdict:** SurrealDB is a learning-forward pick. It buys a genuinely unified model in one binary — and the entity/`RELATE` graph tables (`schema.surql:91`) would require a separate graph DB or PostgreSQL adjacency lists under any alternative. The price is real: younger project with sharper SDK edges (see §7's pitfalls), brute-force O(n) vector scan, and a mandatory separate process. For a single-user local platform the trade holds. For a multi-tenant service, reach for Postgres.

### Category: Vector retrieval strategy — chose brute-force cosine + BM25 (RRF)

The retrieval core is `fn::vector_search` (full-table cosine scan, `schema.surql:40`) + `fn::text_search` (BM25 via SurrealDB's built-in BM25 index, `schema.surql:24`) fused by Reciprocal Rank Fusion (`retrieve.py:7`). A third optional signal is graph-connected chunks.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **HNSW/MTREE approximate index in SurrealDB** | SurrealDB v3 supports MTREE (exact) and, experimentally, HNSW | Sub-linear query time; scales to tens of millions of chunks | Index build time; extra schema definition; still SurrealDB — no operational difference for small corpora |
| **Vector-only retrieval (no BM25)** | Simpler; many RAG tutorials do this | One ranking signal to tune; faster to implement | Consistently worse retrieval quality on named-entity and exact-term queries where keyword matching dominates; known RAG failure mode |

**Verdict:** The hybrid RRF approach is the right default. Brute-force cosine is the honest ceiling: the schema comment at `schema.surql:38` names the MTREE/HNSW swap explicitly — "if the corpus outgrows a full scan." For a personal corpus of hundreds to low thousands of chunks, a full scan is fast and correct. The scaling cliff is real: add 100k chunks and every query does 100k cosine products. The three-signal RRF fusion (vector + BM25 + graph) when graph-RAG is enabled is sophisticated for a local tool and the implementation is clean — `fuse_with_graph` (`retrieve.py:29`) is 4 lines of actual logic.

### Category: LLM provider abstraction — chose esperanto AIFactory

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **LangChain chat/embeddings models** | Already pulling `langchain-core` for LangGraph | Enormous provider coverage, rich ecosystem, well-tested Ollama/Anthropic integrations | Heavy, high-churn APIs; would bind the whole stack to LangChain abstractions and their upgrade cadence |
| **litellm directly** | Already present as a transitive dependency via `[eval]`; broad provider coverage | 100+ provider support behind one interface; used for the eval judge anyway | Another abstraction that must be kept current; slightly more verbose than esperanto's factory pattern; not currently used in the hot path |

**Verdict:** esperanto earns its place specifically because `providers.py` is 37 lines and the `_ChatAdapter` deliberately avoids esperanto's optional LangChain integrations (`providers.py:25`). The risk is supply-chain dependency on a niche library — mitigated by the discipline of the thin-wrapper pattern. The interesting candidate for future replacement is litellm, which is already a transitive dependency and has broader provider coverage; swapping would be a one-file change.

### Category: Retrieval fusion — chose Reciprocal Rank Fusion

RRF is the glue between all ranking signals. The formula is `1 / (k + rank)` summed over lists, with `k=60` (`retrieve.py:9`), a standard choice. It's parameter-free to tune beyond `k`, handles lists of different lengths, and never requires score calibration between vector similarity and BM25 relevance scores (which live in completely different value spaces).

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Linear score combination** | Simple; many RAG systems use it | Directly expressive — if you know vector scores are 10× more useful, encode that | Requires calibrating scores across incompatible spaces (cosine ∈ [-1,1] vs BM25 ∈ [0, ∞]); brittle when either signal is absent or has outliers |
| **Learned re-ranker as sole ranking signal** | FlashRank cross-encoder is already present as an optional step | A cross-encoder directly models query-passage relevance — highest quality signal | Cross-encoder requires N forward passes (slow on long lists); needs the initial candidate set anyway; expensive to train from scratch |

**Verdict:** RRF is the correct choice here. It is provably robust: two lists with no overlap just don't contribute to overlapping scores; a list with a unanimous #1 drives that item to the top regardless of absolute score values. The graph signal as a third RRF input (`fuse_with_graph`, `retrieve.py:29`) is particularly clean because graph-retrieved chunks are a completely different retrieval modality — mixing their "relevance" score with cosine scores would be arbitrary, but RRF just needs ranks.

### Category: LLM operations lifecycle — chose eval-gate-promote-rollback over SurrealDB

The llmops pattern records every eval run as a `eval_run` record, gates promotion on metric regression checks, tracks a `deployment:current` singleton with a `history` list, and enables one-level rollback.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLflow or W&B** | Industry-standard ML experiment tracking | Rich UI, artifact storage, comparison views, team-facing dashboards | Server or cloud account required; overkill for a single-developer local tool; adds infra complexity |
| **Plain JSON files + git** | Simple, zero-dependency | Version-controlled, human-readable, diff-friendly | No query API; rollback is manual; gate comparisons require scripting; doesn't scale to many runs |

**Verdict:** Rolling the llmops layer on top of SurrealDB is the right call given the existing infrastructure. The result is surprisingly capable for its line count: `check_gate` (`gates.py:19`) and `check_drift` (`gates.py:44`) are pure Python with no DB access — easy to unit test; the DB is only accessed for record I/O. The one genuine weakness is the one-level rollback history — the `history` list in `deployment:current` grows unbounded but `rollback` only restores the last entry. A deeper rollback requires querying `eval_run` history directly, which is possible but not exposed as a command.

### Category: Local inference serving — chose Ollama (GGUF/llama.cpp)

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLX (Apple framework)** | Native Apple Silicon, unified-memory + Metal path | Measurably faster prefill/throughput on M-series; lower memory overhead; well-supported quantisation for Apple GPUs | macOS-only; smaller model ecosystem; no drop-in OpenAI-compatible server; more glue code to integrate |
| **vLLM or TGI (GPU serving)** | Production-grade inference with continuous batching, paged KV-cache, AWQ/GPTQ quantisation | 10–24× throughput vs naive serving; production-grade | Requires CUDA; explicitly gated in `bench/serving.py:39` — `vllm_serve` raises `RuntimeError` when `nvidia-smi` is absent; not runnable on this MPS host |

**Verdict:** Ollama is the pragmatic local default — one `ollama pull`, an OpenAI-compatible endpoint esperanto already speaks, automatic model load/unload. The vLLM/TGI path is documented honestly as a design-only holdout (`bench/serving.py:1`): the file exists, the function is named, and it raises cleanly when CUDA is absent rather than silently faking the capability. The MLX alternative is the most interesting future path on Apple Silicon — it would be a new `providers.py` entry behind the existing `select_model` abstraction, a surgical change to one module.

### Category: Evaluation framework — chose Ragas + TruLens with local Ollama judge

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Custom LLM-as-judge with a handwritten rubric** | Total control over metrics, no third-party framework | Simpler, no framework compat issues, easier to debug | Metrics are idiosyncratic and not comparable to published benchmarks; re-invents what Ragas/TruLens ship |
| **DeepEval** | Modern RAG eval library, similar metrics | Active development, more RAG-specific metrics, easier to set up | Same transitive dependency complexity; no proven advantage for the three RAG-triad metrics used here |

**Verdict:** The Ragas + TruLens choice is defensible but maintenance-heavy. `eval/harness.py` contains three compatibility patches just to make these frameworks coexist with each other and with a local Ollama judge: `_patch_trulens_litellm_instrumentation` (`harness.py:62`), `litellm.drop_params = True` (`harness.py:124`), and careful `langchain-community<0.4` + `setuptools<81` pins in `pyproject.toml:33`. This is real technical debt — two external eval frameworks with mismatched assumptions about the LLM provider API, patched at runtime. The payoff is industry-standard metric names (faithfulness, groundedness, context precision) that make the scores legible outside this project.

---

## 7. Lessons

### Best practices this codebase demonstrates

**One seam per concern, injected not hardwired.** `routing.select_model` and `providers.build_*` are the only two places that know about local-vs-cloud or which provider SDK. Everything downstream takes `config` and an `embedder_fn`. Adding a new provider is a one-file change; adding a new routing rule is a two-line change to `routing.py`.

**Editable prompts, never inline strings.** Every model-facing prompt is a Jinja2 template in `ragcore/prompts/` rendered via ai-prompter (`ask.py:29`). `ask_strategy.jinja`, `ask_answer.jinja`, `ask_final.jinja`, `chat_answer.jinja`, `chat_reformulate.jinja`, `graph_extract.jinja` — all editable without touching Python.

**Transient vs. permanent error classification as a first-class concern.** `classify_error` (`errors.py:21`) maps raw exceptions to user-friendly messages; `is_transient` (`errors.py:39`) is a *separate* predicate because retryability and error type are independent. A `ValueError("No content extracted")` must fail permanently; a connection timeout retries. The worker (`worker.py:36`) correctly uses `is_transient` to gate backoff and `classify_error` only for the stored failure message.

**Graph errors must never fail ingestion.** The `try/except` around graph extraction in `ingest.py:78` is an explicit architectural choice: the knowledge graph is enrichment, not core, and a flaky LLM response or parse error should not roll back a successful embed. Same pattern in `graph_context` (`graph.py:273`) — graph retrieval failures return `[]` silently so a graph problem cannot break query responses.

**Cascade deletes in the database, not the application.** `DEFINE EVENT` on `source` and `chat_session` (`schema.surql:14,73`) reaps child rows server-side. The application issues one `DELETE` and walks away.

### Clever bits worth stealing

**Race-safe job claiming without a real lock** (`jobs.py:62`). The `claim_next` design: SELECT the oldest `queued` job id, then `UPDATE ... WHERE status = 'queued' RETURN AFTER`. An empty result means someone else claimed it; a SurrealDB transaction conflict also means someone else claimed it. Both paths return `None`. This is correct without a database-level `SELECT FOR UPDATE`.

**Backoff stored, not scheduled.** A retrying job is still `queued` with a future `next_attempt_at` timestamp (`jobs.py:107`). The `claim_next` query gates on `next_attempt_at <= time::now()` (`jobs.py:75`). No cron, no broker, no sleep loop — the delay lives in a DB field.

**Mean-pooling for oversized embeddings.** `generate_embedding` (`embedding.py:58`) falls back to chunking + mean-pooling over chunk embeddings when a text exceeds the chunk size. The pooling is L2-normalised at the chunk level before averaging and re-normalised after — not just a naive mean of raw vectors.

**`<think>` stripping for reasoning models.** `_clean` (`ask.py:37`) strips `<think>...</think>` blocks from reasoning models like qwen3:8b, including a truncated/unclosed `<think>` that hits a token limit mid-thought. A small detail, important for correctness when using chain-of-thought models.

**The benchmark client uses only `urllib.request`** (`bench/harness.py:115`). No `httpx`, no `requests`. One fewer dependency for the opt-in bench layer, and the Ollama `/api/generate` endpoint returns JSON from a simple POST.

### Real pitfalls (and how this code avoids them)

**The SurrealDB `created`-tie ordering problem.** Two rows created in the same millisecond tie on `ORDER BY created`. Every list query in the codebase breaks ties with a secondary sort on `id` — `ORDER BY created DESC, id DESC` (`store.py:103`, `jobs.py:159`, `sessions.py:84`). `chat_message` uses an explicit integer `seq` (`schema.surql:67`) rather than trusting timestamps, with an accepted single-writer assumption documented in `sessions.py:46`.

**`ORDER BY` requires the sorted field in the SELECT projection** in SurrealDB — `claim_next` selects `created` even though only `id` is used (`jobs.py:73`). The worker retry spec calls this out explicitly.

**Graph entity deduplication via `norm`.** Entities are deduped on their lowercased normalised name (`graph.py:109`), preventing "Ragcore", "ragcore", and "RAGCORE" from becoming three separate nodes. Seeds in `graph_context` skip norms shorter than 3 characters (`graph.py:294`) to avoid flooding matches on common words like "a" or "is".

**The sync/async boundary is held at one layer.** Each CLI command wraps with exactly one `asyncio.run`. The library is uniformly async. This avoids nested-loop hazards that break async code when called from a sync context that's already inside an event loop.

**KMP_DUPLICATE_LIB_OK for torch + faiss.** `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch and faiss to avoid OpenMP duplicate-library crashes on macOS. Documented as a known pitfall, not hidden.

### Honest weaknesses

**Brute-force cosine is a scaling cliff.** `fn::vector_search` does a full-table scan (`schema.surql:40`). The schema comment points at MTREE/HNSW as the upgrade path. For a personal corpus of a few thousand chunks this is imperceptible; for tens of thousands of chunks, every query pays for a full scan. The SurrealDB MTREE index would require a schema change and re-indexing — the seam exists but no migration path is provided.

**Connection-per-operation is real per-call overhead.** Every store call opens a new WebSocket to SurrealDB, signs in, and closes it (`store.py:29`). For one user that's acceptable and pooling is named as the documented seam. `add_embeddings` issues one `CREATE` per chunk in a loop (`store.py:69`) — 40 chunks means 40 round-trips. A bulk insert via SurrealDB's transaction or array `CREATE` would be a meaningful latency improvement on ingest.

**The semantic cache is O(n) linear scan.** `SemanticCache.get` (`cache.py:47`) compares the query embedding against every stored entry with a Python loop. For a small cache this is fine. For hundreds of cached queries it becomes the bottleneck of every `ask` call when cache is enabled. An obvious fix would be a FAISS index over the cache, but that adds a dependency to what is currently a zero-dependency stdlib component.

**No streaming.** `ask` and `chat` block until the full answer is generated and synthesized. On a local 8B model a multi-search `ask` (5 parallel retrievals, 1 synthesis) can take 20–60 seconds with nothing on screen. Streaming from the LangGraph synthesize node through the FastAPI response would require SSE or WebSocket plumbing; the current architecture has no obvious seam for it.

**The eval framework compatibility is fragile.** Three runtime patches in `eval/harness.py` (lines 62, 91, 124) and two version pins in `pyproject.toml:38` exist because Ragas, TruLens, and litellm make incompatible assumptions. Any upstream upgrade of these three packages has a meaningful probability of breaking `ragcore eval`. The eval layer is opt-in (`[eval]` extra), which is the right mitigation — it cannot break the core.

**vLLM/GPU serving is design-only.** `bench/serving.py:39` raises `RuntimeError` on any machine without `nvidia-smi`. This is documented honestly and the MPS Ollama bench is the real artifact on Apple Silicon. For any CUDA deployment, this module is a stub that needs a real implementation.

**Multimodal CLIP quality is dependent on model size.** `ViT-B-32` on LAION2B (`config.example.toml:77`) is a strong baseline but not a state-of-the-art vision-language model. Cross-modal search quality degrades significantly on domain-specific images (technical diagrams, medical imagery) where general LAION pretraining does not transfer. The architecture is sound; the quality ceiling on specialized corpora is a real limitation for the default model.

---

## 8. Requirements, setup, build & run

### System prerequisites

- **Python 3.12** (project pins `>=3.11,<3.13`), recommended via `uv`
- **SurrealDB v3** (`surreal start --user root --pass root rocksdb:./data/db`). The `file:` scheme was removed in v3 — use `rocksdb:` as shown.
- **Ollama** serving a chat model and an embedding model. Recommended: `ollama pull qwen3:8b && ollama pull nomic-embed-text`. Any pulled chat model works — set `[models.chat] local_model` accordingly.
- `OLLAMA_API_BASE` in `.env` (typically `http://localhost:11434`)
- `ANTHROPIC_API_KEY` in `.env` only if using `--cloud` escalation

### One-time setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                               # or: uv pip install -e ".[dev]"
cp config.example.toml config.toml
cp .env.example .env         # set OLLAMA_API_BASE; add ANTHROPIC_API_KEY for --cloud
ollama pull qwen3:8b         # or qwen2.5:7b-instruct; update [models.chat] local_model
ollama pull nomic-embed-text
```

### The two-terminal flow

*Terminal A — database (leave running):*
```bash
surreal start --user root --pass root rocksdb:./data/db
```

*Terminal B — every feature, end to end:*
```bash
# Schema (run once, or after a DB reset)
ragcore init

# Ingestion + source management
ragcore ingest examples/ragcore_demo.md          # file path
ragcore ingest https://example.com/page          # URL
ragcore list                                     # id, chunk count, title, origin
ragcore search "how does retrieval work"         # hybrid chunk preview
ragcore remove source:abc123                     # delete source + embeddings (cascade)

# One-shot Q&A
ragcore ask "How does ragcore do retrieval, and where are embeddings stored?"
ragcore ask --cloud "Summarize everything"       # force cloud escalation
ragcore models                                   # show configured roles

# Async ingestion + worker
ragcore ingest https://example.com/big --async   # enqueue, returns job id immediately
ragcore jobs                                     # list all jobs (newest first)
ragcore jobs --status failed                     # filter by queued|running|done|failed
ragcore retry ingestion_job:xyz                  # requeue a failed job
# Terminal C: ragcore worker        # poll loop (Ctrl-C to stop)
# Terminal C: ragcore worker --once # drain due jobs once, exit (good for cron)

# Multi-turn chat (history-aware)
ragcore chat                                     # new session; prints session id
ragcore chat --session chat_session:abc          # resume existing session
ragcore sessions                                 # list all sessions

# Web UI (FastAPI + single HTML page)
ragcore serve                                    # http://127.0.0.1:8080 (localhost, no auth)
```

### Production RAG upgrade features (optional extras)

```bash
pip install -e ".[rag-upgrade]"   # all: rerank, chroma, faiss, milvus, eval, multimodal
# — or pick individually —
pip install -e ".[rerank]"        # FlashRank cross-encoder
pip install -e ".[chroma]"        # ChromaDB backend
pip install -e ".[faiss]"         # FAISS backend
pip install -e ".[milvus]"        # Milvus Lite backend
pip install -e ".[eval]"          # Ragas + TruLens eval
pip install -e ".[multimodal]"    # CLIP image ingest
```

Enable in `config.toml`:
```toml
[store]
vector_backend = "chroma"         # surreal (default) | chroma | faiss | milvus

[rerank]
enabled = true
model   = "ms-marco-MiniLM-L-12-v2"
top_k   = 5

[cache]
enabled   = true
threshold = 0.95

[graph]
enabled = true
hops    = 1
```

```bash
ragcore eval                                     # Ragas + TruLens report (local judge)
ragcore eval --tag v1                            # label this run
ragcore search --images "a red sunset"           # cross-modal CLIP text→image search
ragcore ingest photo.png                         # CLIP-embed an image
```

### Platform layer (LLMOps, cost, graph, bench)

```bash
pip install -e ".[platform]"      # llmops + cost + multimodal (no GPU deps)
```

Enable and use:
```toml
[llmops]
tolerance       = 0.05
drift_threshold = 0.15

[cost]
enabled     = true
enforce     = false
budget_tokens = 0
ledger_path = "data/cost.db"
```

```bash
# LLMOps lifecycle
ragcore eval --tag v1                            # record baseline eval run
ragcore runs                                     # list all runs
ragcore gate --baseline v1                       # exit 1 if regression beyond tolerance
ragcore drift --baseline v1                      # exit 1 if centroid drift > threshold
ragcore promote <run-id>                         # set deployment:current (gate required)
ragcore promote <run-id> --no-gate               # force promote without gate
ragcore rollback                                 # revert to previous deployment

# Cost tracking
ragcore cost report                              # spend by model, cache hit-rate, hints

# Graph-RAG (triple extraction + retrieval)
ragcore graph build                              # back-fill triples for ingested sources

# Ollama serving benchmark (MPS; GPU serving is design-only on this host)
ragcore bench                                    # sweep num_ctx × num_batch × concurrency
```

### Tests

```bash
# Deterministic suite — no Ollama required; SurrealDB tests auto-skip if binary absent
pytest tests
ruff check ragcore tests

# Live suite — requires Ollama running at http://localhost:11434
pytest tests/live -m live
```

Note: tests set `KMP_DUPLICATE_LIB_OK=TRUE` in `tests/conftest.py` before importing torch and faiss. This is required on macOS to avoid OpenMP duplicate-library crashes and must be set in the environment before any torch/faiss import.
