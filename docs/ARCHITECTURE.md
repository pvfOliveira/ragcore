# ragcore — Architecture Deep Dive

*A Senior Staff Engineer's walkthrough and critical review. Everything below is grounded in code that actually exists in this repo; file:line citations point at the real thing. The story: a clean local-first RAG core that grew into a full local LLM platform.*

---

## 1. What it is

`ragcore` is a local-first Retrieval-Augmented Generation engine you can hold in your head. You feed it files or URLs; it extracts, chunks, embeds, and stores them in a single SurrealDB instance; then it answers questions over that corpus with a local LLM (Ollama), fusing vector similarity and BM25 full-text search via Reciprocal Rank Fusion, optionally enriched by a third knowledge-graph signal. It ships as three faces over one library: a Typer CLI (`ragcore.cli`), a small FastAPI + vanilla-JS web UI (`ragcore.web`), and the importable `ragcore` package itself — which the design docs rightly call "the product."

Over one development sprint it grew a production-rag upgrade layer (pluggable vector backends behind a protocol, FlashRank cross-encoder reranking, a SQLite semantic cache, and a Ragas + TruLens eval harness), and then a full LLM platform layer: llmops lifecycle (eval registry, quality gate, semantic drift check, promote/rollback), AI cost optimization (SQLite token ledger + spending report), graph-RAG (LLM triple extraction into a native SurrealDB entity graph), multimodal CLIP image ingest with cross-modal search, and an Ollama serving benchmark. It is deliberately single-user, single-machine, no-Docker, and small enough that one developer can own every line.

---

## 2. The big picture

Think of ragcore as a **personal research librarian working at a single desk, who over time became the department's data science lead**. When you hand over a document (`ingest`), the librarian photocopies it into index cards (chunks), files each card under two separate catalog systems — a "meaning" catalog (vector embeddings) and a "keyword" catalog (BM25 full-text) — and also draws a relationship map between the entities mentioned (graph-RAG). Everything goes into one filing cabinet (SurrealDB). When you ask a question (`ask`), the librarian first plans which catalog lookups to run, pulls cards from *all three* catalogs in parallel, reconciles the ranked piles via Reciprocal Rank Fusion, and writes you a cited answer. If the answer is nearly identical to one they wrote yesterday, they hand you that one instead (semantic cache). For a casual back-and-forth (`chat`), the same librarian keeps a running memory of the conversation so "what about its license?" still makes sense.

And in the expanded role: the librarian now grades their own work with independent judges (Ragas, TruLens), tracks spending on expensive cloud consultants (cost ledger), and keeps a deployment pointer so you can roll back to the version of themselves that scored better last week (llmops promote/rollback).

The mental model that matters: **SurrealDB is the one cabinet that holds everything** — documents, embeddings, the job queue, chat history, eval run records, the deployment pointer, the entity/relation graph, and image embeddings. There is no message broker and no second datastore for the platform layer, *except* two purpose-sized SQLite files (cost ledger at `data/cost.db` and semantic cache at `data/semantic_cache.db`). The "smart" parts — which model, local vs. cloud — are *injected*, not hardwired, so the librarian can swap pens without relearning how to file.

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

All paths lead to `ragcore/ingest.py:35` → `ingest_source`.

1. **Dedup check.** `store.find_source_id_by_origin` (`store.py:81`) — if this exact file path or URL was already ingested, return `created=False` immediately. No double-indexing. Dedup is by `origin`, by design.
2. **Extraction.** `_extract` (`ingest.py:20`) calls `content_core.extract_content` to produce markdown + title. Works on local files and URLs alike.
3. **Chunking.** `chunk_text` (`chunking.py:80`) recursively splits on paragraph → line → sentence → word boundaries, measured in tiktoken tokens, targeting 400 tokens with 60-token overlap. Fragments under `min_chunk_size` (5 tokens) are dropped.
4. **Embedding.** `generate_embeddings` (`embedding.py:38`) batches chunks (50 at a time) through the configured embedding model with 3× exponential retry.
5. **Storage.** `store.create_source` writes the `source` row; `store.add_embeddings` (`store.py:62`) writes one `source_embedding` row per chunk — content text *and* the embedding vector together. A `DEFINE EVENT` in the schema (`schema.surql:14`) cascades chunk deletion when the source is removed.
6. **Mirror write.** If a non-surreal vector backend is configured (`config.store.vector_backend != "surreal"`), the same chunks are mirrored to that adapter (`ingest.py:60`). SurrealDB remains the source of truth and is never routed through an adapter itself.
7. **Graph extraction (opt-in).** If `config.graph.enabled`, each chunk is passed to `extract_triples` (`graph.py:33`) — an LLM call returning subject/predicate/object triples — and results are upserted into `entity`/`relates` tables via `GraphStore.upsert_triples` (`graph.py:133`). Failures are caught and logged; they never abort ingestion (`ingest.py:78`). This is an explicit architectural choice: the knowledge graph is enrichment, not core.

With `--async`, steps 2–7 are deferred: `JobQueue.enqueue` (`jobs.py:38`) writes a row to `ingestion_job`, and a separate `ragcore worker` process claims and runs it with exponential backoff (`worker.py:18`).

### Walking an `ask` (the LangGraph)

Built in `ask.py:98` as a compiled `StateGraph`. All state is typed via `AskState` (`ask.py:46`).

```
START → strategy → (Send × N terms) → retrieve_answer (parallel) → synthesize → END
```

1. **Semantic cache check (pre-graph).** If `config.cache.enabled`, the question is embedded and compared against all cached query embeddings via `SemanticCache.get` (`cache.py:47`) — a full linear scan with cosine similarity. A hit above `threshold` (default 0.95) short-circuits the entire graph and returns the cached answer (`ask.py:165`).
2. **Budget enforcement.** If cost enforcement is on, `over_budget` (`cost/ledger.py:81`) checks the question token count against the configured limit before any LLM call (`ask.py:149`).
3. **strategy node** (`ask.py:58`): the chat model decomposes your question into up to 5 focused search terms via `prompts/ask_strategy.jinja`. JSON parse failure falls back to `[question]` — a belt-and-suspenders move.
4. **fan-out** (`ask.py:70`): `_fan_out` emits one `Send("retrieve_answer", ...)` per term — these run concurrently. Results accumulate via `Annotated[list, operator.add]` (`ask.py:49`), LangGraph's reducer pattern.
5. **retrieve_answer node** (`ask.py:74`, once per term): `hybrid_search` (`retrieve.py:64`) runs vector + BM25 (+ optional graph signal via `graph_context`) and fuses them with Reciprocal Rank Fusion. If `config.rerank.enabled`, FlashRank cross-encoder reranks the fused candidates. The chat model drafts a partial answer over the top chunks via `prompts/ask_answer.jinja`.
6. **synthesize node** (`ask.py:89`): one final model call merges partial answers into a cited response via `prompts/ask_final.jinja`. Citations are deduped source IDs.
7. **Cache write + cost record.** Answer + embeddings are stored in the semantic cache. Token counts are written to the cost ledger via `_record_usage` (`ask.py:132`).

`chat` (`ragcore/chat.py:18`) is intentionally *not* a graph — it's a linear async function: pull history → reformulate the follow-up into a standalone query if needed → `hybrid_search` → answer with history in context → persist user + assistant turns to `chat_message`. It reuses `_build_chat` and `_clean` directly from `ask.py` (`chat.py:8`) so model provisioning and `<think>`-stripping aren't duplicated — a clean reuse touch.

### Walking the llmops cycle

```bash
ragcore eval --tag v2       # run eval, record eval_run in SurrealDB
ragcore gate --baseline v1  # re-run eval, compare metrics, exit 1 if regression
ragcore promote <run-id>    # set deployment:current (refuses if gate not passed)
ragcore rollback            # revert deployment:current to previous history entry
```

`RunRegistry.record` (`llmops/registry.py:27`) serialises metrics + config snapshot + dataset centroid (mean of question embeddings — a semantic-space fingerprint) as JSON into the `eval_run` table. `check_gate` (`llmops/gates.py:19`) flattens `{family: {metric: float}}` to `"family.metric"` keys and checks each for regression beyond `tolerance`. `check_drift` (`llmops/gates.py:44`) computes cosine distance between current and baseline centroids — pure Python, no DB access. `DeploymentStore.promote` (`llmops/deploy.py:43`) reads the `gate_passed` field from the run before overwriting `deployment:current`; it maintains a `history` list enabling one-level rollback.

---

## 4. Codebase tour

Dependency direction is strict and one-way: **cli/web/worker → library → store → SurrealDB/SQLite/adapters**. The library modules never import from cli or web.

### Core layer

| Module | Job |
|---|---|
| `config.py` | Pydantic models for TOML config; `load_config` enforces that `[models.chat]` and `[models.embedding]` exist. Sixteen typed sub-models covering every section. |
| `errors.py` | Typed exception tree (`RagcoreError` → Configuration/Provider/Store); `classify_error` maps raw exceptions to friendly text; `is_transient` decides retryability — a separate predicate, not the same thing. |
| `chunking.py` | tiktoken-based `token_count`, content-type detection, recursive token-bounded splitter with overlap. |
| `routing.py` | `select_model(config, role, content, force_cloud)` — the local-vs-cloud policy, the project's headline feature. |
| `providers.py` | Thin wrapper over esperanto's `AIFactory`; `_ChatAdapter` exposes a uniform `async ainvoke(prompt) → .content`. This is the one file that knows esperanto exists. |
| `embedding.py` | Batched embeddings with 3× retry + mean-pooling (L2-normalised) for oversized text. |
| `store.py` | SurrealDB repository: schema init, source CRUD, `vector_search`/`text_search` via DB stored functions. Connection-per-operation. |
| `retrieve.py` | `reciprocal_rank_fusion` + `hybrid_search` + `fuse_with_graph`; `vector_store_for` factory routes between SurrealDB and pluggable backends. |
| `ingest.py` | Synchronous extract→chunk→embed→store pipeline; mirrors into pluggable backend + graph extraction if enabled. |
| `ask.py` | Compiled LangGraph: strategy → parallel fan-out → synthesize. Cache and cost wiring live here. |
| `chat.py` | Linear history-aware multi-turn RAG. Deliberately not a graph. |
| `sessions.py` | `SessionStore`: chat sessions + messages. |
| `jobs.py` | `JobQueue`: SurrealDB-table queue with claim-via-conditional-UPDATE and backoff stored in `next_attempt_at`. |
| `worker.py` | `run_worker`: claim → `ingest_source` → mark done/retry/failed loop. |
| `cache.py` | `SemanticCache`: SQLite + pure-Python cosine, zero dependencies beyond stdlib. Linear scan at query time. |
| `rerank.py` | FlashRank cross-encoder post-RRF reranking. Injectable `_scorer` for tests. |
| `cli.py` | Typer entrypoint; every command is a thin `asyncio.run(...)` over a library function. Three sub-apps: root, `graph`, `cost`. |
| `web/app.py` | FastAPI app-factory; DI seams (`get_config`/`get_store`/…) for testing; one catch-all exception handler. |
| `web/static/index.html` | The entire frontend: HTML + CSS + vanilla `fetch()` in one file, no build step. |
| `db/schema.surql` | Tables, BM25 analyzer/indexes, `fn::vector_search` (brute-force cosine), `fn::text_search` (BM25), `fn::image_search`, and cascade-delete events. |

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
| `multimodal.py` | `ClipEmbedder` (lazy open_clip/torch load, MPS/CUDA/CPU), `ImageStore` (`image_embedding` table), `ingest_image`, `cross_modal_rank`. |
| `llmops/registry.py` | `RunRegistry`: record/list/resolve `eval_run` records in SurrealDB. |
| `llmops/gates.py` | `check_gate` (metric regression test) + `check_drift` (centroid cosine distance). Pure Python, no DB. |
| `llmops/deploy.py` | `DeploymentStore`: `deployment:current` singleton with history list for one-level rollback. |
| `cost/ledger.py` | `CostLedger`: SQLite usage ledger + `aggregate()` + `over_budget()`. |
| `cost/report.py` | `build_report`: spend by model, cache savings, right-sizing hints, bench summary. |
| `eval/harness.py` | `run_eval` (live), `compute_metrics` (stubbed-judge path for deterministic tests), `_OllamaJudge` (litellm + Ragas + TruLens). |
| `bench/harness.py` | `run_bench`: cartesian sweep, `OllamaBenchClient` via raw `urllib.request`. |
| `bench/serving.py` | `vllm_serve`: design-only CUDA-gated stub; raises on no `nvidia-smi`. |

### LLMOps & Observability v2 layer

A fourth layer of industry-standard LLMOps tooling, every piece **opt-in and default-off**. The invariant that governs the whole layer: with its config flags unset, `ingest`/`ask`/`chat`/`retrieve` run byte-identical to the core — no traces emitted, no gateway hop, no optimized prompt loaded. Each capability is gated behind its own optional extra (`[observability]`, `[gateway]`, `[dspy]`; DeepEval + Promptfoo ride the existing `[eval]`).

| Module | Role |
|---|---|
| `observability/attributes.py` | Helpers emitting the stable `gen_ai.*` OpenTelemetry-GenAI semantic-convention subset (`gen_ai.system`, `gen_ai.request.model`, token usage). |
| `observability/tracing.py` | Lazy OTel `TracerProvider` → **Arize Phoenix** in-process collector; **Langfuse** wired as a second OTLP exporter *code path* (Basic-auth header), not a running dependency. `get_tracer` returns `None` (zero overhead) when observability is disabled. |
| `eval/deepeval_judge.py` | **DeepEval** metrics behind a local-Ollama judge, selected via `config.eval.framework`. |
| `eval/promptfoo/config.yaml` + runner | **Promptfoo** declarative eval driven through the Node `promptfoo` CLI (the YAML now ships in the wheel via `[tool.setuptools.package-data]`). |
| `gateway.py` | **LiteLLM** unified gateway behind `routing.select_model`; opt-in `[gateway]`. The `ask` pipeline routes its model calls through it only when `config.gateway.enabled`. |
| `dspy_optimizer.py` | **DSPy** strategy-prompt optimizer (`compile_strategy`/`load_compiled_strategy`) + the `optimize` CLI; the compiled prompt preserves the `{{question}}`/`{{max_searches}}` placeholders so the optimized strategy slots straight into the existing Jinja template. |

**Eval framework selector.** `config.eval.framework` chooses among the Ragas+TruLens harness (`harness.py`), DeepEval (`deepeval_judge.py`), and Promptfoo (Node CLI) — three interchangeable graders over the same local judge.

**Live proof (each capability has its own live test, all green):** `tests/live/test_observability.py` (Phoenix OTel trace captures an `ask`), `tests/live/test_eval_frameworks.py` (DeepEval gate + Promptfoo run over Ollama), `tests/live/test_gateway.py` (LiteLLM routes a real call to Ollama), `tests/live/test_dspy.py` (BootstrapFewShot compile produces a loadable artifact preserving both prompt placeholders).

---

## 5. Tech stack & why

**Python 3.11–3.12, async throughout.** The whole library is `async def`; the CLI bridges to it with one `asyncio.run` per command. This avoids the "fragile sync/async bridging" trap (`new_event_loop()` hacks) that plagued the predecessor project.

**SurrealDB 3.x (rocksdb backend), one instance.** Multi-model: it's the document store, the vector store (`vector::similarity::cosine`), the BM25 full-text engine, the job queue, the chat history, the eval run registry, the deployment pointer, the entity-relation graph, and the image embedding table — all in SurrealQL, one binary, one connection URL. The `DEFINE EVENT` cascade-delete mechanism means the application never has to clean up child rows. The cost is real: it's a younger product with sharper edges (see §7), the vector index is a full scan, and you must run a separate process. Worth it for the unified model.

**Start command is `rocksdb:`, not `file:`.** SurrealDB v3 removed the `file:` scheme. The correct incantation is `surreal start --user root --pass root rocksdb:./data/db`. This is a corrected-as-built fact — any design doc or tutorial that says `file:./data/db` is wrong.

**esperanto `AIFactory`** for provider abstraction. ragcore never imports Ollama or Anthropic SDKs directly; `providers.py` is the only file that knows esperanto exists. The `_ChatAdapter` (`providers.py:13`) exposes a uniform `async ainvoke(prompt) → .content`. If esperanto vanished, you'd rewrite one 37-line file.

**Ollama for local inference** (`qwen3:8b`/`qwen2.5:7b-instruct` chat, `nomic-embed-text` embeddings) over its OpenAI-compatible endpoint. Two resident models, not seven — unified memory on an M2 is the binding constraint. One `ollama pull`, automatic load/unload, and esperanto already speaks its endpoint.

**LangGraph** for the `ask` graph — specifically for the parallel fan-out with an additive reducer (`Annotated[list, operator.add]`, `ask.py:49`) and the `Send` primitive (`ask.py:71`). Notably, `chat.py` consciously rejected LangGraph for a linear flow — a good signal the framework is applied only where its concurrency model actually pays off.

**content-core** for extraction (files/URLs → markdown), **tiktoken** (`o200k_base`) for token counting, **ai-prompter/Jinja2** for editable prompt templates (`prompts/*.jinja` — every model-facing string is a file, not a hardcoded literal), **Typer** for the CLI, **FastAPI + uvicorn** for the web layer, **loguru** for worker logs, **Pydantic v2** for typed config and request bodies.

**FlashRank** (optional `[rerank]` extra) for cross-encoder reranking post-RRF. **open_clip + torch** (optional `[multimodal]`) for CLIP image embeddings with MPS/CUDA/CPU autodetect. **Ragas + TruLens + litellm** (optional `[eval]`) for eval metrics with a local Ollama judge. **SQLite (stdlib)** for the semantic cache and cost ledger — a deliberate zero-extra-dependency choice for opt-in single-user components.

---

## 6. Architectural review — alternatives matrix

### Category: Primary datastore — chose SurrealDB

SurrealDB does the work of five components: document store, vector index, BM25 FTS, job queue, and graph database, all behind one connection URL. That's the bet.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **PostgreSQL + pgvector + tsvector** | Battle-tested, one DB for docs + vectors + FTS | Mature HNSW/IVFFlat indexes, real ACID transactions, proven ops ecosystem | Heavier to run locally; BM25 requires `pg_bm25`/ParadeDB extension; no native graph; cascade-delete needs triggers |
| **SQLite + sqlite-vec + FTS5** | True zero-server, single-file, perfect for one user | Embeds in-process (no separate terminal), trivial backup, FTS5 BM25 is excellent | No graph layer; vector support is immature; concurrency story weak; no DEFINE EVENT equivalent |

**Verdict:** SurrealDB is a *learning-forward* pick. It buys a genuinely unified model — and the entity/`RELATE` graph tables (`schema.surql:91`) would require a separate graph DB or PostgreSQL adjacency lists under any alternative. The price is real: younger project with sharper SDK edges (see §7), brute-force O(n) vector scan, and a mandatory separate process. For a single-user local tool that's a fair trade; for a multi-tenant service I'd reach for Postgres without hesitation.

### Category: Vector retrieval strategy — chose brute-force cosine + BM25 (RRF)

The retrieval core is `fn::vector_search` (full-table cosine scan, `schema.surql:40`) + `fn::text_search` (BM25 via SurrealDB's built-in BM25 index, `schema.surql:24`) fused by Reciprocal Rank Fusion (`retrieve.py:7`). A third optional signal is graph-connected chunks via `graph_context`.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **HNSW/MTREE approximate index in SurrealDB** | SurrealDB v3 supports MTREE (exact) and experimentally HNSW | Sub-linear query time; scales to large corpora | Index build time; extra schema definition; still SurrealDB — no operational difference for small corpora |
| **Vector-only retrieval (no BM25)** | Simpler; many RAG tutorials do this | One ranking signal to tune | Consistently worse on named-entity and exact-term queries where keyword matching dominates; a known RAG failure mode |

**Verdict:** The hybrid RRF approach is the right default. Brute-force cosine is the honest ceiling — the schema comment at `schema.surql:38` names the MTREE/HNSW swap explicitly. For a personal corpus this is imperceptible; the scaling cliff is real (add 100k chunks and every query pays for 100k cosine products). The three-signal fusion (vector + BM25 + graph) when graph-RAG is enabled is sophisticated for a local tool and the implementation is clean — `fuse_with_graph` (`retrieve.py:29`) is effectively 4 lines of logic.

### Category: AI provider abstraction — chose esperanto's AIFactory

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **LangChain chat/embeddings models** | Already pulling in `langchain-core` for LangGraph | Enormous provider coverage, rich ecosystem | Heavy, high-churn APIs; would couple the whole stack to LangChain's abstractions and upgrade cadence |
| **Raw provider SDKs (ollama-python, anthropic)** | Maximum control, minimal deps | No abstraction tax, exact feature access | You hand-write the local↔cloud switch per provider; swapping providers becomes code, not config |

**Verdict:** esperanto is the right call *because* of `providers.py`'s discipline — it's the single seam, ~37 lines, and the `_ChatAdapter` deliberately avoids esperanto's optional LangChain integrations (`providers.py:25`). The risk is supply-chain dependency on a niche library — mitigated exactly by the thin-wrapper pattern. The interesting candidate for future replacement is litellm, which is already a transitive dependency via `[eval]`; swapping would be a one-file change.

### Category: Retrieval fusion — chose Reciprocal Rank Fusion

RRF is the glue between all ranking signals. The formula is `1 / (k + rank)` summed over lists, with `k=60` (`retrieve.py:9`), a standard choice. It's parameter-free to tune beyond `k`, handles lists of different lengths, and never requires score calibration between vector similarity and BM25 relevance (which live in completely different value spaces).

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Linear score combination** | Simple; many RAG systems use it | Directly expressive — if you know vector scores are 10× more useful, encode that | Requires calibrating scores across incompatible spaces (cosine ∈ [-1,1] vs BM25 ∈ [0, ∞]); brittle when either signal is absent |
| **Learned re-ranker as sole ranking signal** | FlashRank cross-encoder is already present as an optional step | A cross-encoder directly models query-passage relevance — highest quality signal | Requires N forward passes (slow on long lists); needs the initial candidate set anyway |

**Verdict:** RRF is the correct choice here. The graph signal as a third RRF input (`fuse_with_graph`, `retrieve.py:29`) is particularly clean because graph-retrieved chunks are a completely different retrieval modality — mixing their "relevance" with cosine scores would be arbitrary, but RRF just needs ranks.

### Category: Background job queue — chose a home-grown SurrealDB-table queue

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Celery + Redis/RabbitMQ** | Industry default for Python task queues | Mature retries, scheduling, monitoring, scale-out | A broker + worker convention + new infra for a single-user tool — absurd weight here |
| **`surreal-commands`** (the lfnovo lib open-notebook uses) | Already SurrealDB-native, live-query push, built-in retries | No reinvention; push instead of poll | Another dependency + its worker-entrypoint conventions; less transparent than ~150 lines you own |

**Verdict:** Rolling their own (`jobs.py`, ~180 lines) is the most defensible "build vs. buy" in the repo. The `claim_next` design (`jobs.py:62`) is genuinely thoughtful: SELECT oldest `queued` job, then conditional `UPDATE ... WHERE status='queued'`, treating both an empty result *and* a transaction conflict as "someone else got it." Backoff is stored, not scheduled — a retrying job is just `queued` with a future `next_attempt_at` (`jobs.py:107`), gated in the claim query. Honest costs: it polls (2 s default), sequential by design (local embedder is the bottleneck), and there's a benign TOCTOU dedup window the code explicitly accepts (`jobs.py:39`).

### Category: LLM operations lifecycle — chose eval-gate-promote-rollback over SurrealDB

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLflow or W&B** | Industry-standard ML experiment tracking | Rich UI, artifact storage, comparison views, team dashboards | Server or cloud account required; overkill for a single-developer local tool |
| **Plain JSON files + git** | Simple, zero-dependency | Version-controlled, human-readable, diff-friendly | No query API; rollback is manual; gate comparisons require scripting |

**Verdict:** Rolling the llmops layer on top of SurrealDB is the right call. The result is surprisingly capable: `check_gate` and `check_drift` are pure Python with no DB access — easy to unit test; the DB is only touched for record I/O. The one genuine weakness is the one-level rollback history — `rollback` only restores the last entry. A deeper rollback requires querying `eval_run` history directly, which is possible but not exposed as a command.

### Category: Local inference serving — chose Ollama (GGUF/llama.cpp)

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLX (Apple framework)** | Native Apple Silicon, unified-memory + Metal path | Measurably faster prefill/throughput on M-series; lower memory overhead | macOS-only; smaller model ecosystem; no drop-in OpenAI-compatible server; more glue code |
| **vLLM or TGI (GPU serving)** | Production-grade with continuous batching, paged KV-cache | 10–24× throughput vs naive serving; production-grade | Requires CUDA; explicitly gated in `bench/serving.py:39` — raises `RuntimeError` when `nvidia-smi` is absent |

**Verdict:** Ollama is the pragmatic local default — one `ollama pull`, an OpenAI-compatible endpoint, automatic model load/unload. The vLLM/TGI path is documented honestly as a design-only holdout: `bench/serving.py` exists, the function is named, and it raises cleanly when CUDA is absent rather than silently faking the capability. The MLX alternative is the most interesting future path on Apple Silicon — it would be a new `providers.py` entry behind the existing `select_model` abstraction, a surgical one-file change.

### Category: Evaluation framework — chose Ragas + TruLens with local Ollama judge

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Custom LLM-as-judge with a handwritten rubric** | Total control over metrics | Simpler, no framework compat issues, easier to debug | Metrics are idiosyncratic and not comparable to published benchmarks |
| **DeepEval** | Modern RAG eval library, similar metrics | Active development, more RAG-specific metrics | Same transitive dependency complexity; no proven advantage for the three RAG-triad metrics used here |

**Verdict:** The Ragas + TruLens choice is defensible but maintenance-heavy. `eval/harness.py` contains three compatibility patches just to make these frameworks coexist with each other and with a local Ollama judge: `_patch_trulens_litellm_instrumentation` (`harness.py:62`), `litellm.drop_params = True` (`harness.py:124`), and `langchain-community<0.4` + `setuptools<81` pins in `pyproject.toml`. This is real technical debt — two external eval frameworks with mismatched assumptions about the LLM provider API, patched at runtime. The payoff is industry-standard metric names (faithfulness, groundedness, context precision) that make scores legible outside this project. The eval layer is `[eval]` opt-in, so it cannot break the core.

---

## 7. Lessons

### Best practices this codebase demonstrates

**One seam per concern, injected not hardwired.** `routing.select_model` and `providers.build_*` are the only two places that know about local-vs-cloud or which provider SDK. Everything downstream takes `config` and an `embedder_fn`. Adding a new provider is a one-file change; adding a new routing rule is a two-line change to `routing.py`.

**Editable prompts, never inline strings.** Every model-facing prompt is a Jinja2 template in `ragcore/prompts/` rendered via ai-prompter. `ask_strategy.jinja`, `ask_answer.jinja`, `ask_final.jinja`, `chat_answer.jinja`, `chat_reformulate.jinja`, `graph_extract.jinja` — all editable without touching Python source.

**Transient vs. permanent error classification as a first-class concern.** `classify_error` (`errors.py:21`) maps raw exceptions to user-friendly messages; `is_transient` (`errors.py:39`) is a *separate* predicate because retryability and error class are independent. A `ValueError("No content extracted")` must fail permanently; a connection timeout retries. The worker (`worker.py:36`) correctly uses `is_transient` to gate backoff and `classify_error` only for the stored failure message. That split is subtle and correct.

**Graph errors must never fail ingestion.** The `try/except` around graph extraction in `ingest.py:78` is an explicit architectural choice: the knowledge graph is enrichment, not core. Same pattern in `graph_context` (`graph.py:273`) — graph retrieval failures return `[]` silently so a graph problem cannot break query responses.

**Cascade deletes in the database, not the application.** `DEFINE EVENT` on `source` and `chat_session` (`schema.surql:14,73`) reaps child rows server-side. The application issues one `DELETE` and walks away.

### Clever bits worth stealing

**Race-safe job claiming without a real lock** (`jobs.py:62`). The `claim_next` design: SELECT the oldest `queued` job id, then `UPDATE ... WHERE status = 'queued' RETURN AFTER`. An empty result means someone else claimed it; a SurrealDB transaction conflict also means someone else claimed it. Both return `None`. Correct without a database-level `SELECT FOR UPDATE`.

**Backoff stored, not scheduled.** A retrying job is still `queued` with a future `next_attempt_at` timestamp (`jobs.py:107`). The `claim_next` query gates on `next_attempt_at <= time::now()` (`jobs.py:75`). No cron, no broker, no sleep loop — the delay lives in a DB field.

**Mean-pooling for oversized embeddings.** `generate_embedding` (`embedding.py:58`) falls back to chunking + mean-pooling over chunk embeddings when text exceeds chunk size. The pooling is L2-normalised at the chunk level before averaging and re-normalised after — not a naive mean of raw vectors.

**`<think>` stripping for reasoning models.** `_clean` (`ask.py:32`) strips `<think>...</think>` blocks from reasoning models like `qwen3:8b`, including a truncated/unclosed `<think>` that hit a token limit mid-thought. Small detail, important for correctness when using chain-of-thought models.

**The benchmark client uses only `urllib.request`** (`bench/harness.py:115`). No `httpx`, no `requests`. One fewer dependency for the opt-in bench layer.

### Real pitfalls (and how this code avoids them)

**The SurrealDB `created`-tie ordering trap.** Two rows created in the same millisecond tie on `ORDER BY created`. Every list query breaks ties with a secondary sort on `id` — `ORDER BY created DESC, id DESC` (`store.py:103`, `jobs.py:159`, `sessions.py:84`). `chat_message` uses an explicit integer `seq` (`schema.surql:67`) rather than trusting timestamps, with an accepted single-writer assumption documented in `sessions.py:46`.

**`ORDER BY` requires the sorted field in the SELECT projection** in SurrealDB. `claim_next` selects `created` even though it only needs `id` (`jobs.py:73`). The retry spec calls this out explicitly.

**Graph entity deduplication via `norm`.** Entities are deduped on their lowercased normalised name (`graph.py:109`), preventing "Ragcore", "ragcore", and "RAGCORE" from becoming three separate nodes. Seeds in `graph_context` skip norms shorter than 3 characters (`graph.py:294`) to avoid flooding matches on common words.

**The sync/async boundary is held at exactly one layer.** Each CLI command wraps with exactly one `asyncio.run`. The library is uniformly async. This avoids nested-loop hazards.

**KMP_DUPLICATE_LIB_OK for torch + faiss.** `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch and faiss. Required on macOS to avoid OpenMP duplicate-library crashes; documented, not hidden.

### Honest weaknesses

**Brute-force cosine is a scaling cliff.** `fn::vector_search` does a full-table scan (`schema.surql:40`). The comment names the MTREE/HNSW swap as the upgrade path — the seam exists but no migration is provided. For a personal corpus of a few thousand chunks it's imperceptible. At tens of thousands of chunks, every query pays for a full scan with no escape hatch short of a schema change and re-indexing.

**Connection-per-operation is real per-call overhead.** Every store call opens a new WebSocket to SurrealDB, signs in, and closes it (`store.py:29`). Acceptable for one user, and pooling is the documented future seam. `add_embeddings` issues one `CREATE` per chunk in a loop (`store.py:69`) — 40 chunks means 40 round-trips. A bulk insert via SurrealDB's transaction or array `CREATE` would be a meaningful latency improvement.

**The semantic cache is O(n) linear scan with no eviction.** `SemanticCache.get` (`cache.py:47`) compares the query embedding against every stored entry with a Python loop. For a small cache this is fine; for hundreds of cached queries it becomes the bottleneck of every `ask` call when cache is enabled. There is also no eviction policy — the cache grows unbounded.

**No streaming anywhere.** `ask` and `chat` block until the full answer is built. On a local 8B model a multi-search `ask` (5 parallel retrievals, 1 synthesis) can take 20–60 seconds with nothing on screen. Streaming from the LangGraph synthesize node through the FastAPI response would require SSE or WebSocket plumbing; the current architecture has no seam for it.

**The eval framework compatibility is fragile.** Three runtime patches in `eval/harness.py` (lines 62, 91, 124) and two version pins in `pyproject.toml` exist because Ragas, TruLens, and litellm make incompatible assumptions. Any upstream upgrade of these three packages has meaningful probability of breaking `ragcore eval`. The `[eval]` opt-in is the right mitigation — it cannot break the core.

**vLLM/GPU serving is design-only.** `bench/serving.py:39` raises `RuntimeError` on any machine without `nvidia-smi`. Documented honestly. For any CUDA deployment, this module is a stub that needs a real implementation.

**Embedding model has no cloud fallback by design.** `routing.select_model` only escalates when a cloud model is configured for that role; embedding has none. A down local embedder takes down both ingest and retrieval with no escape hatch.

**The "500 gap" is closed, but the spec lied about how.** The web-UI design says errors map `ConfigurationError → 400, StoreError/ProviderError → 502, else 500`. In reality the catch-all handler in `app.py` only ever returns **400 (config) or 502 (everything else)** — there is no 500 branch, which is the point. The code is *stricter* than the spec; the spec is stale.

---

## 8. Requirements, setup, build & run

### System prerequisites

- **Python 3.12** (project pins `>=3.11,<3.13`), recommended via `uv`
- **SurrealDB v3** — start with `rocksdb:` scheme. The `file:` scheme was removed in v3; any doc saying `surreal start file:./data/db` is wrong.
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

SurrealDB must be running for everything below.

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
```

*Terminal C (optional) — background worker for `--async` ingests:*
```bash
ragcore worker          # poll loop; processes queued jobs until Ctrl-C
ragcore worker --once   # drain currently-due jobs once and exit (good for cron/tests)
```
Note: `--once` drains only jobs whose `next_attempt_at` is due; a job still backing off is left for a later run.

```bash
# Multi-turn chat (history-aware)
ragcore chat                                     # new session; prints session id
#   you> What is ragcore?
#   you> What about its license?                # follow-up resolved against history
#   you> /exit
ragcore chat --session chat_session:abc          # resume existing session
ragcore sessions                                 # list sessions (id, msg count, title)

# Web UI (FastAPI + single HTML page)
ragcore serve                                    # http://127.0.0.1:8080 (localhost, no auth)
ragcore serve --host 127.0.0.1 --port 8080       # explicit host/port
```
In the browser: add sources by URL/path in the sidebar, start a chat, switch sessions, and **rename (✎) or delete (✕) sessions** — two controls the CLI doesn't expose. Do not put this on a network as-is.

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

Enable in `config.toml`:
```toml
[llmops]
tolerance       = 0.05
drift_threshold = 0.15

[cost]
enabled       = true
enforce       = false
budget_tokens = 0
ledger_path   = "data/cost.db"
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

### LLMOps & Observability v2 (optional extras)

All default-off; with the flags unset, ingest/ask/chat/retrieve behave byte-identically to the core.

```bash
pip install -e ".[observability]"   # Arize Phoenix OTel tracing (Langfuse OTLP code path)
pip install -e ".[gateway]"         # LiteLLM unified gateway
pip install -e ".[dspy]"            # DSPy strategy-prompt optimizer
pip install -e ".[eval]"            # now also installs DeepEval + pyyaml (Promptfoo is a Node CLI)
# all of the above (plus the rag-upgrade set) at once:
pip install -e ".[rag-upgrade]"
npm install -g promptfoo            # Promptfoo runner is a global Node CLI, not a pip dep
```

Enable in `config.toml`:
```toml
[observability]
enabled       = true
otlp_endpoint = "http://localhost:6006/v1/traces"   # Phoenix

[gateway]
enabled = true

[dspy]
enabled       = true
compiled_path = "data/dspy_compiled.json"

[eval]
framework = "deepeval"            # ragas (default) | deepeval | promptfoo
```

```bash
ragcore optimize                                 # DSPy-compile the strategy prompt → compiled_path
```

> Known pin tension: `deepeval` requires `click<8.4.0` while `huggingface-hub`/`dspy` allow `>=8.4`; `pip check` surfaces one benign metadata warning depending on which click is resolved. Both import and the full suite stays green either way (see the comment above the `eval`/`dspy` extras in `pyproject.toml`).

### Tests / lint (dev workflow)

```bash
# Deterministic suite — no Ollama required; SurrealDB tests auto-skip if binary absent
pytest tests
ruff check ragcore tests

# Live suite — requires Ollama running at http://localhost:11434
pytest tests/live -m live
```

Tests set `KMP_DUPLICATE_LIB_OK=TRUE` in `tests/conftest.py` before importing torch and faiss. Required on macOS to avoid OpenMP duplicate-library crashes; must be set before any torch/faiss import.

---

### Appendix: where the code contradicts the docs

1. **Vector index.** ~~The local-RAG design (§4) assumed an **HNSW** vector index~~ — *corrected 2026-06-02*: the as-built implementation uses a **brute-force cosine full scan** (`db/schema.surql:40`), with HNSW/MTREE noted as the future seam. (The README's worked example was never index-specific.)
2. **SurrealDB start command.** ~~Design §10 said `surreal start file:./data/db`~~ — *corrected 2026-06-02* to `rocksdb:` (SurrealDB v3 removed the `file:` scheme). README and config were already correct.
3. **Web error mapping.** The web-UI design says errors map `ConfigurationError → 400, StoreError/ProviderError → 502, else 500`. The implementation (`web/app.py`) only ever returns **400 or 502** — there is no 500 branch, which closes the raw-500 gap. The code is stricter than the spec.
4. **MVP scope drift (expected, not a bug).** The original design listed web UI, async worker, and chat-history as explicit non-goals/YAGNI. All three now exist via later approved specs — the seams held, which is the design vindicating itself.
