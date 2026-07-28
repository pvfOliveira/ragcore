# ragcore — Architecture Deep Dive

*A Senior Staff Engineer's walkthrough and critical review. Everything below is grounded in code that actually exists in this repo; file:line citations point at the real thing. The story: a clean local-first RAG core that grew into a full local LLM platform, through four expansion waves (production-rag → platform → LLMOps/observability v2 → RAG v3 → RAG v4).*

---

## 1. What it is

`ragcore` is a local-first Retrieval-Augmented Generation engine you can hold in your head. You feed it files, URLs, images, or audio; it extracts, chunks, embeds, and stores them in a single SurrealDB instance; then it answers questions over that corpus with a local LLM (Ollama), fusing vector similarity and BM25 full-text search via Reciprocal Rank Fusion, optionally enriched by a third knowledge-graph signal. It ships as three faces over one library: a Typer CLI (`ragcore.cli`), a small FastAPI + vanilla-JS web UI (`ragcore.web`), and the importable `ragcore` package itself — which the design docs rightly call "the product."

Over several development sprints it grew, in order: a **production-rag** upgrade layer (pluggable vector backends behind a protocol, FlashRank cross-encoder reranking, a SQLite semantic cache, a Ragas + TruLens eval harness); a **platform** layer (llmops lifecycle, cost ledger, graph-RAG, multimodal CLIP, serving bench); an **LLMOps & observability v2** layer (OTel-GenAI tracing → Phoenix, a LiteLLM gateway, DSPy prompt optimization, DeepEval/Promptfoo eval frameworks); a **RAG v3** layer (query rewriting, agentic/corrective RAG, structured generation, document-AI parsing, an in-process Qdrant backend); and a **RAG v4** layer (pgvector and Embedded-Weaviate backends, Qdrant hybrid dense+sparse with BM42, LLMLingua-2 context compression, mlx-whisper audio ingest, versioned golden datasets, and sampled online eval). It is deliberately single-user, single-machine, no-Docker, and small enough that one developer can own every line. Every capability past the core is **opt-in and default-off** — with the flags unset, ingest/ask/chat run byte-identical to the original core.

---

## 2. The big picture

Think of ragcore as a **personal research librarian working at a single desk, who over time became the department's data science lead**. When you hand over a document (`ingest`), the librarian photocopies it into index cards (chunks), files each card under two separate catalog systems — a "meaning" catalog (vector embeddings) and a "keyword" catalog (BM25 full-text) — and also draws a relationship map between the entities mentioned (graph-RAG). Everything goes into one filing cabinet (SurrealDB). When you ask a question (`ask`), the librarian first plans which catalog lookups to run, pulls cards from *all three* catalogs in parallel, reconciles the ranked piles via Reciprocal Rank Fusion, and writes you a cited answer. If the answer is nearly identical to one they wrote yesterday, they hand you that one instead (semantic cache). For a casual back-and-forth (`chat`), the same librarian keeps a running memory of the conversation so "what about its license?" still makes sense.

And in the expanded role: the librarian now rewrites your fuzzy question into several sharper ones before searching (query rewrite), will re-search when the first pile of cards looks weak (agentic RAG), can summarise long cards down to their load-bearing sentences before quoting them (LLMLingua-2 compression), transcribes a meeting recording into searchable text (audio ingest), grades their own work with independent judges (Ragas, TruLens, DeepEval), tracks spending on expensive cloud consultants (cost ledger), records a full activity log of every call (OTel tracing), and keeps a deployment pointer so you can roll back to the version of themselves that scored better last week (llmops promote/rollback).

The mental model that matters: **SurrealDB is the one cabinet that holds everything** — documents, embeddings, the job queue, chat history, eval run records, the deployment pointer, the entity/relation graph, and image embeddings. There is no message broker and no second datastore for the platform layer, *except* two purpose-sized SQLite files (cost ledger at `data/cost.db` and semantic cache at `data/semantic_cache.db`) and whatever external vector store you opt into. The "smart" parts — which model, local vs. cloud — are *injected*, not hardwired, so the librarian can swap pens without relearning how to file.

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
          │  ingest.py ─► chunking.py embedding.py store.py                    │
          │  retrieve.py (RRF + graph_context + adapter bridge)                │
          │  ask.py (LangGraph: strategy → fan-out → synthesize)              │
          │  chat.py (linear, history-aware)                                   │
          │  routing.py  providers.py (esperanto AIFactory)                   │
          │  cache.py (SemanticCache/SQLite)  rerank.py (FlashRank)           │
          │                                                                    │
          │  RAG UPGRADE LAYER — pluggable vector backends                     │
          │  vectorstores/base.py (VectorStore protocol + factory)            │
          │  chroma_store · faiss_store · milvus_store                        │
          │  qdrant_store (in-proc; BM42 hybrid) · pgvector_store · weaviate  │
          │                                                                    │
          │  RETRIEVAL/GENERATION v3                                           │
          │  query_rewrite.py (multi_query/hyde/decompose)                    │
          │  agentic.py (grade-docs → re-retrieve → grade-answer)            │
          │  structured.py (instructor/outlines)                              │
          │  docai.py (docling/pymupdf + VLM caption)                         │
          │  audio.py (mlx-whisper)  compress.py (LLMLingua-2)               │
          │                                                                    │
          │  PLATFORM LAYER                                                    │
          │  graph.py · multimodal.py (CLIP)                                  │
          │  llmops/ (registry · gates · deploy)                              │
          │  cost/ (ledger · report)                                          │
          │  eval/ (harness Ragas+TruLens · deepeval_judge · promptfoo       │
          │         · golden datasets · online sampled eval)                 │
          │  bench/ (harness Ollama sweep · serving.py vllm stub)            │
          │                                                                    │
          │  OBSERVABILITY/PLATFORM v2 (all opt-in)                            │
          │  observability/ (otel gen_ai attrs · spans · tracing→Phoenix)     │
          │  gateway.py (LiteLLM)  dspy_optimizer.py (compile strategy)       │
          └────────────────┬──────────────────────────┬───────────────────────┘
                           │                          │
              ┌────────────▼──────────────┐   ┌───────▼──────────────────────┐
              │    SurrealDB 3.x           │   │  Ollama (local LLM)          │
              │    rocksdb:./data/db       │   │  qwen3:8b / nomic-embed-text │
              │    ─────────────────────  │   │  + cloud (Anthropic optional) │
              │    source · source_emb     │   └──────────────────────────────┘
              │    ingestion_job           │
              │    chat_session/message    │   ┌──────────────────────────────┐
              │    eval_run · deployment   │   │  SQLite (stdlib, 2 files)    │
              │    entity / relates        │   │  data/cost.db (token ledger) │
              │    image_embedding         │   │  data/semantic_cache.db      │
              │    fn::vector_search (BF)   │   └──────────────────────────────┘
              │    fn::text_search (BM25)   │
              │    fn::image_search         │   ┌──────────────────────────────┐
              └────────────────────────────┘   │  External vector adapters    │
                                               │  data/chroma · data/faiss    │
              ┌────────────────────────────┐   │  data/milvus.db · qdrant     │
              │  Tracing sink (opt-in)     │   │  pgvector (local PG) ·        │
              │  Arize Phoenix OTLP :6006  │   │  weaviate embedded (binary)  │
              │  (Langfuse OTLP code path) │   └──────────────────────────────┘
              └────────────────────────────┘
```

### Walking an `ingest` (synchronous path)

All text/document paths lead to `ragcore/ingest.py:35` → `ingest_source`. The CLI front door (`cli.py:67`) first routes by file type: image extensions go to `multimodal.ingest_image`, audio extensions (`.mp3/.wav/.m4a/.flac`, `audio.py:19`) go to `ingest_audio`, `--async` enqueues a job, and everything else falls through to the synchronous text pipeline.

1. **Dedup check.** `store.find_source_id_by_origin` — if this exact path or URL was already ingested, return `created=False` immediately. Dedup is by `origin`, by design.
2. **Extraction.** `_extract` (`ingest.py:21`) calls `content_core.extract_content` to produce markdown + title. **Document-AI hook:** if `config.docai.enabled` and the file is a PDF/scan, `extract_if_document` (`docai.py:30`) overrides extraction with Docling (layout-aware) or PyMuPDF markdown (`ingest.py:44`).
3. **Chunking.** `chunk_text` recursively splits on paragraph → line → sentence → word boundaries, measured in tiktoken tokens, targeting 400 with 60 overlap; fragments under 5 tokens dropped.
4. **Embedding.** `generate_embeddings` batches chunks (50 at a time) through the configured embedding model with 3× exponential retry.
5. **Storage.** `store.create_source` writes the `source` row; `store.add_embeddings` writes one `source_embedding` row per chunk — content text *and* embedding vector together. A `DEFINE EVENT` (`schema.surql:14`) cascades chunk deletion when the source is removed.
6. **Mirror write.** If a non-surreal vector backend is configured (`config.store.vector_backend != "surreal"`), the same chunks are mirrored to that adapter (`ingest.py:70`) via `make_vector_store`. SurrealDB remains the source of truth and is never routed through an adapter itself.
7. **Graph extraction (opt-in).** If `config.graph.enabled`, each chunk is passed to `extract_triples` (`graph.py:33`) and upserted into `entity`/`relates`. Failures are caught and logged; they never abort ingestion (`ingest.py:91`).

**Audio ingest** (`audio.py:61`) is a thin shim, not a separate pipeline: `transcribe_audio` runs mlx-whisper on Apple-Silicon Metal, `format_transcript` joins the segments (optionally with inline `[mm:ss]` markers), and the transcript is fed straight into `ingest_source(content=...)` — so it becomes a normal text source: chunked, embedded, hybrid-searchable, citable.

With `--async`, steps 2–7 are deferred: `JobQueue.enqueue` writes a row to `ingestion_job`, and a separate `ragcore worker` process claims and runs it with exponential backoff.

### Walking an `ask` (the LangGraph + its bypasses)

The compiled `StateGraph` is built in `ask.py:169`; state is typed via `AskState` (`ask.py:99`). The public entry is `answer_question` (`ask.py:282`), which opens a root OTel `"ask"` span and delegates to `_answer_question_inner` (`ask.py:218`). That inner function chooses among **three** execution paths:

```
answer_question
 ├─ agentic.enabled  ──► run_agentic   (corrective loop, NOT the graph)
 ├─ structured flag  ──► answer_structured (single retrieve + schema gen, separate entry)
 └─ default          ──► [budget check] → [cache check] → LangGraph
                          START → strategy → (Send × N) → retrieve_answer ∥ → synthesize → END
```

1. **Agentic bypass** (`ask.py:219`). If `config.agentic.enabled`, control goes to `agentic.run_agentic` (`agentic.py:38`): search → `grade_documents` (LLM keeps only relevant chunks) → if too few relevant, rewrite the query and re-search up to `max_iterations` → answer → `grade_answer` for groundedness. This is a self/corrective-RAG loop that never touches the LangGraph.
2. **Budget enforcement.** If `config.cost.enforce`, `over_budget` checks the question token count against the limit before any LLM call (`ask.py:238`).
3. **Semantic cache check.** If `config.cache.enabled`, the question is embedded and compared against cached query embeddings via `SemanticCache.get` — a full linear scan with cosine similarity (`cache.py`). A hit above `threshold` (default 0.95) short-circuits the graph and returns the cached answer (`ask.py:253`).
4. **strategy node** (`ask.py:111`): the chat model decomposes the question into up to 5 focused search terms via `prompts/ask_strategy.jinja`. JSON parse failure falls back to `[question]`. If `config.dspy.enabled` and a compiled prompt exists, `_render_strategy` (`ask.py:43`) swaps in the DSPy-optimized template instead.
5. **fan-out** (`ask.py:125`): `_fan_out` emits one `Send("retrieve_answer", ...)` per term — these run concurrently, accumulating via `Annotated[list, operator.add]`.
6. **retrieve_answer node** (`ask.py:129`, once per term): the heart of v3/v4 retrieval.
   - If `config.query_rewrite.enabled`, the term is expanded via `expand_and_fuse` (`query_rewrite.py:46`) — multi_query/hyde/decompose variants, each searched and RRF-fused.
   - Otherwise a single `hybrid_search` (`retrieve.py:64`) runs vector + BM25 (+ optional graph) fused by RRF. When a pluggable backend that `supports_hybrid` is selected (Qdrant hybrid), the vector ranking comes from its server-side dense+sparse fusion instead (`retrieve.py:81`).
   - If `config.rerank.enabled`, FlashRank cross-encoder reranks the candidates.
   - If `config.compression.enabled`, `compress_context` (`compress.py:31`) runs each chunk through the LLMLingua-2 encoder before it enters the prompt.
   - The chat model drafts a partial answer over the top chunks via `prompts/ask_answer.jinja`.
7. **synthesize node** (`ask.py:159`): one final model call merges partials into a cited response via `prompts/ask_final.jinja`.
8. **Cache write + cost record.** Answer + embeddings stored in the semantic cache; token counts written to the cost ledger via `_record_usage` (`ask.py:203`).

Every model call in the graph goes through `_traced_invoke` (`ask.py:78`), which wraps the call in a `gen_ai` OTel span (no-op when tracing is disabled) carrying the conformant `gen_ai.*` attributes plus a `ragcore.pipeline.stage`. The gateway seam is also here: `_select_and_build` (`ask.py:63`) routes through the LiteLLM gateway when `config.gateway.enabled`, else the esperanto provider.

**Structured answers** (`answer_structured`, `ask.py:296`) are a fourth, separate entry: a single `hybrid_search` then `generate_structured` (`structured.py:56`) via Instructor (Ollama JSON mode) or Outlines (constrained decoding). Deliberately no `force_cloud` — structured generation is local-only.

`chat` (`chat.py:18`) is intentionally *not* a graph — a linear async function: pull history → reformulate the follow-up into a standalone query if needed → `hybrid_search` → optional compression → answer with history in context → optional **online eval** sampling (`online.maybe_score_turn`, `chat.py:43`) → persist user + assistant turns. It reuses `_build_chat` and `_clean` directly from `ask.py` so model provisioning and `<think>`-stripping aren't duplicated.

### Walking the llmops cycle

```bash
ragcore eval --tag v2       # run eval over golden v1, record eval_run in SurrealDB
ragcore gate --baseline v1  # re-run eval, compare metrics, exit 1 if regression
ragcore promote <run-id>    # set deployment:current (refuses if gate not passed)
ragcore rollback            # revert deployment:current to previous history entry
```

`RunRegistry.record` (`llmops/registry.py:27`) serialises metrics + config snapshot + dataset centroid (mean of question embeddings) as JSON into `eval_run`. `check_gate` (`llmops/gates.py:19`) flattens `{family: {metric: float}}` to `"family.metric"` keys and checks each for regression beyond `tolerance`. `check_drift` (`llmops/gates.py:44`) computes cosine distance between current and baseline centroids — pure Python, no DB access. `DeploymentStore.promote` reads `gate_passed` before overwriting `deployment:current`, maintaining a `history` list for one-level rollback. Eval now defaults to the versioned **golden v1** dataset (`eval/harness.py:36`, 10 provenance-noted items); `--dataset` still overrides, and the pre-v4 `dataset.jsonl` stays on disk for run-history comparability.

---

## 4. Codebase tour

Dependency direction is strict and one-way: **cli/web/worker → library → store → SurrealDB/SQLite/adapters**. The library modules never import from cli or web.

### Core layer

| Module | Job |
|---|---|
| `config.py` | Pydantic models for TOML config; `load_config` enforces `[models.chat]` + `[models.embedding]`. ~30 typed sub-models covering every section, all opt-in features default-off. |
| `errors.py` | Typed exception tree (`RagcoreError` → Configuration/Provider/Store); `classify_error` maps raw exceptions to friendly text; `is_transient` is a *separate* retryability predicate. |
| `chunking.py` | tiktoken-based `token_count`, content-type detection, recursive token-bounded splitter with overlap. |
| `routing.py` | `select_model(config, role, content, force_cloud)` — the local-vs-cloud policy, the project's headline feature. |
| `providers.py` | Thin wrapper over esperanto's `AIFactory`; `_ChatAdapter` exposes uniform `async ainvoke(prompt) → .content`. The one file that knows esperanto exists. |
| `embedding.py` | Batched embeddings with 3× retry + L2-normalised mean-pooling for oversized text. |
| `store.py` | SurrealDB repository: schema init, source CRUD, `vector_search`/`text_search` via DB stored functions. Connection-per-operation. Also serves as the default `VectorStore` (`base.py:62`). |
| `retrieve.py` | `reciprocal_rank_fusion` + `hybrid_search` + `fuse_with_graph`; `vector_store_for` factory routes between SurrealDB and pluggable backends; `_adapter_hit_to_ragcore` bridges adapter `{id,text,score}` to ragcore's chunk shape. |
| `ingest.py` | Synchronous extract→chunk→embed→store; docai override, pluggable-backend mirror, graph extraction. |
| `ask.py` | Compiled LangGraph + agentic/structured bypasses + cache/cost/gateway/tracing wiring. |
| `chat.py` | Linear history-aware multi-turn RAG with online-eval sampling. Deliberately not a graph. |
| `sessions.py` · `jobs.py` · `worker.py` | Chat session store; SurrealDB-table job queue with claim-via-conditional-UPDATE + stored backoff; worker claim→ingest→retry loop. |
| `cache.py` · `rerank.py` | SQLite + pure-Python cosine semantic cache; FlashRank cross-encoder post-RRF rerank. |
| `cli.py` | Typer entrypoint; every command a thin `asyncio.run(...)`. Root app + `graph` + `cost` sub-apps. |
| `web/app.py` · `web/static/index.html` | FastAPI app-factory with DI seams + one catch-all exception handler; entire frontend in one HTML file, no build step. |
| `db/schema.surql` | Tables, BM25 analyzer/indexes, `fn::vector_search` (brute-force cosine), `fn::text_search` (BM25), `fn::image_search`, cascade-delete events. |

### RAG upgrade layer — pluggable vector backends

| Module | Role |
|---|---|
| `vectorstores/base.py` | `VectorStore` runtime-checkable Protocol (`add_embeddings`/`vector_search`) + `make_vector_store(config)` factory dispatching on `vector_backend`. |
| `chroma_store.py` · `faiss_store.py` · `milvus_store.py` | Local persistent ChromaDB (HNSW cosine); in-process FAISS `IndexFlatIP` + sidecar `meta.json`; Milvus Lite local-file or remote. |
| `qdrant_store.py` | In-process Qdrant (`QdrantClient(path=...)`, no server). Uses `query_points()` (the 1.10 API; legacy `search()` removed). Optional **hybrid** mode: named dense + BM42 learned-sparse vectors (fastembed), fused server-side by Qdrant's `FusionQuery(RRF)` — exposes `supports_hybrid`/`hybrid_search` that `retrieve.hybrid_search` detects. Fails fast on a schema/flag mismatch. |
| `pgvector_store.py` | Postgres + pgvector over local Homebrew PG (no Docker). psycopg3 async, HNSW `vector_cosine_ops`, returns `1 - (embedding <=> q)` as similarity. pgvectorscale (StreamingDiskANN) is an honest holdout — separate extension, Rust/pgrx build, Linux-first. |
| `weaviate_store.py` | **Embedded** Weaviate — the v4 client downloads a `Darwin-all.zip` server binary and runs it as a child process; no Docker. Self-provided vectors (ragcore brings its own Ollama embeddings), HNSW cosine, version-pinned for reproducibility. |

### Retrieval / generation v3

| Module | Role |
|---|---|
| `query_rewrite.py` | `rewrite_query` (multi_query/hyde/decompose) + `expand_and_fuse` (per-variant search → RRF). |
| `agentic.py` | `grade_documents`, `grade_answer`, `run_agentic` corrective loop. |
| `structured.py` | `generate_structured` via Instructor (Ollama OpenAI-compat JSON mode) or Outlines (constrained decoding over a tiny local transformers model). |
| `docai.py` | `extract_if_document` (Docling/PyMuPDF PDF→markdown, wired into ingest) + `caption_image` (a modest Ollama-VLM/moondream caption assist). |
| `audio.py` | `is_audio_path`, `transcribe_audio` (mlx-whisper), `format_transcript`, `ingest_audio` → reuses `ingest_source`. |
| `compress.py` | `compress_context` — LLMLingua-2 query-agnostic token-classification compression; records ratio onto the active OTel span. |

### Platform layer

| Module | Role |
|---|---|
| `graph.py` | `extract_triples` (LLM JSON), `GraphStore` (entity/`RELATE`-edge CRUD), `graph_context` (entity-seeded traversal → chunk retrieval). |
| `multimodal.py` | `ClipEmbedder` (lazy open_clip/torch, MPS/CUDA/CPU), `ImageStore` (`image_embedding` table), `ingest_image`, `cross_modal_rank`. |
| `llmops/registry.py · gates.py · deploy.py` | `RunRegistry`; `check_gate`+`check_drift` (pure Python); `DeploymentStore` singleton with one-level rollback. |
| `cost/ledger.py · report.py` | `CostLedger` (SQLite usage + `aggregate()`/`over_budget()`); `build_report` (spend by model, cache savings, right-sizing hints, bench summary). |
| `eval/harness.py` | `run_eval` (live), `compute_metrics` (stubbed-judge path), `_OllamaJudge` (litellm + Ragas + TruLens), `_build_judge_for_framework` selector. |
| `eval/deepeval_judge.py` | DeepEval metrics behind a local-Ollama judge, selected via `config.eval.framework`. |
| `eval/promptfoo/runner.py` + `promptfooconfig.yaml` | Promptfoo declarative eval driven through the Node `promptfoo` CLI (a *separate* runner, not part of the `framework` selector). |
| `eval/golden.py` + `golden/v1.jsonl` + `MANIFEST.md` | Versioned, provenance-validated golden dataset (`load_golden` raises on a malformed item); the default eval corpus. |
| `eval/online.py` | `maybe_score_turn` — sampled scoring of live chat turns; runs the judge in a worker thread, never raises into the turn, emits an `online_eval` span. |
| `bench/harness.py · serving.py` | `run_bench` (Ollama param sweep via raw `urllib.request`); `vllm_serve` design-only CUDA-gated stub. |

### Observability / platform v2 layer

Every piece **opt-in and default-off**. The invariant: with the flags unset, the pipelines run byte-identical — no traces emitted, no gateway hop, no optimized prompt loaded.

| Module | Role |
|---|---|
| `observability/otel.py` | `set_gen_ai_attributes` — emits the stable `gen_ai.*` OpenTelemetry-GenAI subset (`gen_ai.system`, `gen_ai.operation.name`, `gen_ai.request.model`, token usage), deliberately skipping volatile event/tool attributes. |
| `observability/tracing.py` | Lazy OTel `TracerProvider` → **Arize Phoenix** OTLP collector; **Langfuse** wired as a second OTLP exporter *code path* (Basic-auth header), not a running dependency. `get_tracer` returns `None` (zero overhead) when disabled. |
| `observability/spans.py` | `traced_span` — a no-op-safe span context manager (`tracer is None` → yields `None`). |
| `gateway.py` | `LiteLLMGateway` behind `routing.select_model`; opt-in `[gateway]`, drop-in `ainvoke` adapter, fallback chain. |
| `dspy_optimizer.py` | DSPy strategy-prompt optimizer (`compile_strategy`/`load_compiled_strategy`) + the `optimize` CLI; the compiled prompt preserves the `{{question}}`/`{{max_searches}}` placeholders so it slots into the existing Jinja seam. |

**Eval framework selector.** `config.eval.framework` chooses between the Ragas+TruLens harness (`harness.py`, default) and DeepEval (`deepeval_judge.py`) — `_build_judge_for_framework` (`harness.py:90`). Promptfoo is a *third* grader but lives in its own Node-CLI runner, not the selector.

---

## 5. Tech stack & why

**Python 3.11–3.12, async throughout.** The whole library is `async def`; the CLI bridges with one `asyncio.run` per command. Avoids the "fragile sync/async bridging" trap that plagued the predecessor project.

**SurrealDB 3.x (rocksdb backend), one instance.** Multi-model: document store, vector store (`vector::similarity::cosine`), BM25 full-text engine, job queue, chat history, eval run registry, deployment pointer, entity-relation graph, image embedding table — all in SurrealQL, one binary, one URL. `DEFINE EVENT` cascade-delete means the application never cleans up child rows. The cost is real (younger product, full-scan vector index, separate process) but the unified model earns it.

**Start command is `rocksdb:`, not `file:`.** SurrealDB v3 removed the `file:` scheme. The correct incantation is `surreal start --user root --pass root rocksdb:./data/db` — a corrected-as-built fact; any doc saying `file:./data/db` is wrong.

**esperanto `AIFactory`** for provider abstraction. ragcore never imports Ollama/Anthropic SDKs directly; `providers.py` is the single seam (~37 lines). If esperanto vanished you'd rewrite one file. The LiteLLM gateway slots *behind* `select_model` as an alternative call path without disturbing this seam.

**Ollama for local inference** (`qwen3:8b`/`qwen2.5:7b-instruct` chat, `nomic-embed-text` embeddings) over its OpenAI-compatible endpoint. Two resident models — unified memory on an M2 is the binding constraint.

**LangGraph** for the `ask` graph — specifically the parallel fan-out with an additive reducer and the `Send` primitive. Notably `chat.py` and `agentic.py` consciously reject LangGraph for their linear/iterative flows — a good signal the framework is applied only where its concurrency model pays off.

**Pluggable vector backends.** Beyond SurrealDB's built-in cosine, six adapters behind one async protocol: Chroma, FAISS, Milvus Lite, **Qdrant** (in-process, optional BM42 hybrid), **pgvector** (local Postgres, HNSW), **Embedded Weaviate** (downloadable Darwin binary, no Docker). Each is lazy-imported behind its own extra; the factory picks one from config.

**fastembed BM42** for learned-sparse vectors in Qdrant hybrid mode — fused with dense vectors *server-side* by Qdrant's Query API RRF, a genuinely different fusion locus from the default client-side RRF over SurrealDB BM25.

**LLMLingua-2** (optional `[compress]`) for query-agnostic context/prompt compression — a small local token-classification transformer on CPU/MPS. **mlx-whisper** (optional `[audio]`) for Whisper transcription native to Apple-Silicon Metal. **Docling/PyMuPDF** (optional `[docai]`) for layout-aware PDF→markdown. **Instructor/Outlines** (optional `[structured]`) for schema-validated generation.

**content-core** for extraction, **tiktoken** (`o200k_base`) for token counting, **ai-prompter/Jinja2** for editable prompt templates (`prompts/*.jinja` — every model-facing string is a file), **Typer** CLI, **FastAPI + uvicorn** web, **loguru** logs, **Pydantic v2** config.

**FlashRank** (`[rerank]`) cross-encoder rerank. **open_clip + torch** (`[multimodal]`) CLIP image embeddings. **Ragas + TruLens + litellm** / **DeepEval** / **Promptfoo** (`[eval]`) for eval metrics with a local Ollama judge. **OpenTelemetry SDK → Arize Phoenix** (`[observability]`) for gen_ai tracing. **LiteLLM** (`[gateway]`), **DSPy** (`[dspy]`). **SQLite (stdlib)** for the semantic cache and cost ledger — a deliberate zero-extra-dependency choice for opt-in single-user components.

---

## 6. Architectural review — alternatives matrix

### Category: Primary datastore — chose SurrealDB

SurrealDB does the work of five components: document store, vector index, BM25 FTS, job queue, and graph database, all behind one connection URL. That's the bet.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **PostgreSQL + pgvector + tsvector** | Battle-tested, one DB for docs + vectors + FTS | Mature HNSW/IVFFlat, real ACID, proven ops; ragcore *already* ships a pgvector adapter | Heavier locally; BM25 needs `pg_bm25`/ParadeDB; no native graph; cascade-delete needs triggers |
| **SQLite + sqlite-vec + FTS5** | True zero-server, single-file, perfect for one user | Embeds in-process (no separate terminal), trivial backup, FTS5 BM25 is excellent | No graph layer; vector support immature; weak concurrency; no DEFINE EVENT equivalent |

**Verdict:** SurrealDB is a *learning-forward* pick. It buys a genuinely unified model — and the entity/`RELATE` graph tables (`schema.surql:90`) would need a separate graph DB or adjacency lists under any alternative. The price is real: younger SDK, brute-force O(n) vector scan, mandatory separate process. Fair for a single-user local tool; for a multi-tenant service I'd reach for Postgres without hesitation — and the existing `pgvector_store.py` is the seam that would make that move cheap.

### Category: Vector retrieval backend — chose SurrealDB cosine, with six pluggable alternatives

The default is `fn::vector_search` (full-table cosine scan, `schema.surql:40`) + `fn::text_search` (BM25, `schema.surql:28`) fused by RRF (`retrieve.py:7`), plus an optional graph signal. Behind a protocol, the vector half can be swapped for Chroma/FAISS/Milvus/Qdrant/pgvector/Weaviate. The most interesting two:

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Qdrant hybrid (dense + BM42 sparse, server-side RRF)** | One store does ANN + learned-sparse keyword in a single fused query | Learned-sparse beats classic BM25 on some queries; fusion server-side; in-process, no server | Schema is fixed at collection creation — toggling hybrid needs a fresh collection; an extra fastembed model to download |
| **pgvector HNSW on local Postgres** | Production-grade ANN with a real RDBMS underneath | Sub-linear HNSW, ACID, joins, the most portable-to-prod target | Separate server + extension; no BM25 (ragcore still leans on SurrealDB FTS); pgvectorscale is a Linux/Rust holdout |

**Verdict:** The hybrid RRF default is right, and the pluggable-protocol design is the standout — `make_vector_store` (`base.py:43`) makes the backend a one-line config change, with SurrealDB always the source of truth and adapters receiving a *mirror* write. Brute-force cosine is the honest ceiling (schema comment at `schema.surql:38` names the MTREE/HNSW swap); for a personal corpus it's imperceptible, but the scaling cliff is real. Two distinct hybrid loci now exist — client-side RRF over SurrealDB BM25 (default) and server-side RRF inside Qdrant (BM42) — and the code is careful to label them as different things rather than conflating "hybrid."

### Category: AI provider abstraction — chose esperanto's AIFactory

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **LangChain chat/embeddings models** | Already pulling `langchain-core` for LangGraph | Enormous provider coverage, rich ecosystem | Heavy, high-churn APIs; couples the whole stack to LangChain's cadence |
| **Raw provider SDKs (ollama-python, anthropic)** | Maximum control, minimal deps | No abstraction tax, exact feature access | You hand-write the local↔cloud switch per provider; swapping becomes code, not config |

**Verdict:** esperanto is the right call *because* of `providers.py`'s discipline — a single ~37-line seam. The risk is supply-chain dependency on a niche library, mitigated by the thin wrapper. The interesting future replacement is litellm, already a transitive dep via `[eval]` and now a first-class optional gateway — swapping the primary path would be a one-file change.

### Category: Retrieval fusion — chose Reciprocal Rank Fusion

RRF is the glue between all ranking signals: `1 / (k + rank)` summed over lists, `k=60` (`retrieve.py:8`). Parameter-free beyond `k`, handles lists of different lengths, never requires score calibration between cosine and BM25.

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Linear score combination** | Simple; many RAG systems use it | Directly expressive weighting | Requires calibrating scores across incompatible spaces (cosine ∈ [-1,1] vs BM25 ∈ [0,∞]); brittle when a signal is absent |
| **Learned re-ranker as sole signal** | FlashRank cross-encoder is already an optional step | Cross-encoder directly models query-passage relevance | N forward passes (slow); needs the candidate set anyway |

**Verdict:** RRF is correct here. It now fuses up to *four* ways — vector, BM25, graph (`fuse_with_graph`, `retrieve.py:29`), and query-rewrite variants (`expand_and_fuse`) — all of which live in different score spaces where RRF's rank-only inputs shine.

### Category: Background job queue — chose a home-grown SurrealDB-table queue

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Celery + Redis/RabbitMQ** | Industry default for Python task queues | Mature retries, scheduling, scale-out | A broker + new infra for a single-user tool — absurd weight |
| **`surreal-commands`** (the lfnovo lib open-notebook uses) | SurrealDB-native, live-query push, built-in retries | No reinvention; push instead of poll | Another dependency + worker conventions; less transparent than ~180 lines you own |

**Verdict:** Rolling their own (`jobs.py`) is the most defensible build-vs-buy in the repo. `claim_next` is genuinely thoughtful: SELECT oldest `queued`, then conditional `UPDATE ... WHERE status='queued'`, treating both empty result *and* transaction conflict as "someone else got it." Backoff is stored, not scheduled — a retrying job is just `queued` with a future `next_attempt_at`. Honest costs: it polls (2 s), is sequential by design (local embedder is the bottleneck), and accepts a benign TOCTOU dedup window.

### Category: LLM operations lifecycle — chose eval-gate-promote-rollback over SurrealDB

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLflow or W&B** | Industry-standard experiment tracking | Rich UI, artifact storage, comparison views | Server/cloud account; overkill for a single-developer local tool |
| **Plain JSON files + git** | Simple, zero-dependency | Version-controlled, diff-friendly | No query API; manual rollback; gate comparisons need scripting |

**Verdict:** Rolling llmops on SurrealDB is the right call. `check_gate`/`check_drift` are pure Python with no DB access — easy to unit test. The genuine weakness is one-level rollback history (`rollback` only restores the last entry); deeper rollback means querying `eval_run` directly, not exposed as a command.

### Category: Local inference serving — chose Ollama (GGUF/llama.cpp)

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLX (Apple framework)** | Native Apple Silicon, unified-memory + Metal | Faster prefill/throughput on M-series; lower overhead | macOS-only; smaller ecosystem; no drop-in OpenAI server; more glue |
| **vLLM or TGI (GPU serving)** | Production continuous batching, paged KV-cache | 10–24× throughput; production-grade | Requires CUDA; explicitly gated in `bench/serving.py:39` — raises when `nvidia-smi` absent |

**Verdict:** Ollama is the pragmatic local default. The vLLM/TGI path is documented honestly as a design-only holdout that raises cleanly rather than faking the capability. (Interesting wrinkle: ragcore *already* runs MLX in two places — mlx-whisper for audio, and Outlines/torch on MPS — so an MLX chat provider behind `select_model` is a credible future surgical change.)

### Category: Context/prompt compression — chose LLMLingua-2 (query-agnostic)

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **LLM-summarise each chunk** | Obvious "just ask the model" approach | Highest semantic fidelity; query-aware possible | An extra LLM call per chunk — slow and itself token-expensive; defeats the cost goal |
| **No compression / smaller top-k** | Simplest; just retrieve fewer chunks | Zero added latency or deps | Loses recall; a blunt instrument vs. token-level pruning |

**Verdict:** LLMLingua-2 is a smart fit — a *small* local encoder prunes tokens far cheaper than an LLM summary, and the code is scrupulously honest that the LLMLingua-2 path is **query-agnostic** (only LLMLingua-1's coarse mode is question-aware; `compress.py:5`). The measured live keep-ratio (0.526 at `rate=0.5`) is the cost evidence. The honest limit: query-agnostic pruning can drop tokens that *this* question needed.

### Category: Evaluation framework — chose Ragas + TruLens (+ DeepEval/Promptfoo), local Ollama judge

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Custom LLM-judge with a handwritten rubric** | Total control over metrics | Simpler, no framework compat issues | Metrics idiosyncratic, not comparable to published benchmarks |
| **DeepEval only** | Modern RAG eval, similar metrics | Active development, RAG-specific metrics | Same transitive-dep complexity; brings the `click<8.4` pin tension |

**Verdict:** Defensible but maintenance-heavy. `eval/harness.py` carries runtime patches just to make these frameworks coexist with a local Ollama judge: `_patch_trulens_litellm_instrumentation` (`harness.py:99`), `litellm.drop_params = True`, plus `langchain-community<0.4`/`setuptools<81`/`deepeval>=4`/`click` pins in `pyproject.toml`. Real technical debt — patched at runtime. The payoff is industry-standard metric names (faithfulness, groundedness, context precision) legible outside this project, and a pluggable selector so DeepEval/Promptfoo can be swapped in. All `[eval]` opt-in, so it cannot break the core.

---

## 7. Lessons

### Best practices this codebase demonstrates

**One seam per concern, injected not hardwired.** `routing.select_model` and `providers.build_*` (now optionally `gateway`) are the only places that know about local-vs-cloud or which provider SDK. Everything downstream takes `config` and an `embedder_fn`. Adding a provider or a vector backend is a one-file change.

**Default-off, byte-identical core.** Every wave (production-rag, platform, v2, v3, v4) is gated behind its own config flag *and* its own optional extra. With the flags unset, ingest/ask/chat run exactly as the original core — no traces, no gateway hop, no compression, no rewrite. This is the single most important property keeping a 30-config-section project sane.

**Editable prompts, never inline strings.** Every model-facing prompt is a Jinja2 template in `ragcore/prompts/` (now 12 of them: ask_strategy/answer/final, chat_answer/reformulate, graph_extract, query_multi/hyde/decompose, agentic_grade_docs/answer, structured_answer). Even the DSPy-optimized strategy preserves the template placeholders so it drops into the same seam.

**Transient vs. permanent error classification as a first-class concern.** `classify_error` maps raw exceptions to friendly messages; `is_transient` is a *separate* predicate because retryability and error class are independent. The worker uses `is_transient` to gate backoff and `classify_error` only for the stored failure message.

**Errors in enrichment must never fail the core path.** The `try/except` around graph extraction (`ingest.py:91`), online-eval scoring that "NEVER raises into the user's turn" (`online.py:42`), and the compression span-attr emit that swallows exceptions (`compress.py:67`) all encode the same rule: enrichment is enrichment, not core.

**Cascade deletes in the database, not the application.** `DEFINE EVENT` on `source` and `chat_session` reaps child rows server-side; the app issues one `DELETE` and walks away.

**Honest capability labelling.** Repeatedly the code refuses to overclaim: compression is documented query-agnostic; Qdrant hybrid is explicitly distinguished from client-side BM25 RRF; "prod" in online eval is labelled as the local chat path; pgvectorscale and vLLM/GPU are named as holdouts. This honesty-over-feature-count discipline (`structured.py:37`, `pgvector_store.py:7`, `online.py:4`) is the most transferable habit here.

### Clever bits worth stealing

**Race-safe job claiming without a real lock** (`jobs.py`). SELECT oldest `queued` id, then `UPDATE ... WHERE status='queued' RETURN AFTER`. Empty result *or* transaction conflict both mean someone else claimed it; both return `None`. Correct without `SELECT FOR UPDATE`.

**Backoff stored, not scheduled.** A retrying job is `queued` with a future `next_attempt_at`; the claim query gates on `next_attempt_at <= time::now()`. No cron, no broker, no sleep loop.

**Audio/docai reuse the text pipeline instead of forking it.** `ingest_audio` transcribes then calls `ingest_source(content=...)`; docai overrides only the extraction step. New input modalities became searchable+citable with almost no new pipeline code (`audio.py:61`, `ingest.py:44`).

**No-op-safe tracing.** `get_tracer` returns `None` when disabled and `traced_span(None, ...)` yields `None`, so the instrumentation sprinkled through `ask.py` costs nothing when observability is off — the seam is invisible until you flip it on.

**`<think>` stripping for reasoning models.** `_clean` (`ask.py:54`) strips `<think>...</think>` including a truncated/unclosed block that hit a token limit mid-thought.

**The benchmark client uses only `urllib.request`.** No `httpx`, no `requests` — one fewer dependency for the opt-in bench layer.

### Real pitfalls (and how this code avoids them)

**The SurrealDB `created`-tie ordering trap.** Two rows in the same millisecond tie on `ORDER BY created`. List queries break ties with a secondary `id` sort; `chat_message` uses an explicit integer `seq` (`schema.surql:67`) with a documented single-writer assumption.

**`ORDER BY` requires the sorted field in the SELECT projection** in SurrealDB — `claim_next` selects `created` even though it only needs `id`.

**Vector-backend schema is immutable once created.** Qdrant hybrid-vs-flat is fixed at collection creation; `_ensure_collection` (`qdrant_store.py:68`) raises on a flag mismatch rather than silently ingesting into the wrong schema. pgvector/Weaviate validate identifier/collection names to avoid injection and invalid DDL.

**KMP_DUPLICATE_LIB_OK for torch + faiss.** `tests/conftest.py` sets it before importing torch/faiss — required on macOS to avoid OpenMP duplicate-library crashes; documented, not hidden.

### Honest weaknesses

**Brute-force cosine is a scaling cliff** for the *default* backend. `fn::vector_search` is a full scan (`schema.surql:40`); the MTREE/HNSW swap is named but not implemented. The escape hatch is now real, though — switch to pgvector/Weaviate/Qdrant (all HNSW) via one config line.

**Connection-per-operation is real per-call overhead.** Every store call opens a new WebSocket to SurrealDB, signs in, closes it. `add_embeddings` issues one `CREATE` per chunk in a loop — 40 chunks = 40 round-trips. Pooling/bulk insert remains a documented future seam.

**The semantic cache is O(n) linear scan with no eviction.** `SemanticCache.get` compares the query embedding against every stored entry in Python; the cache grows unbounded.

**No streaming anywhere.** `ask`/`chat` block until the full answer is built; a multi-search `ask` on a local 8B model can take 20–60 s with nothing on screen. The architecture has no SSE/WebSocket seam for it.

**Eval-framework compatibility is fragile.** Runtime patches in `eval/harness.py` plus several `pyproject.toml` pins (`langchain-community<0.4`, `setuptools<81`, `deepeval>=4`, the `click<8.4` tension) exist because Ragas/TruLens/litellm/DeepEval/DSPy make incompatible assumptions. Any upstream bump has meaningful odds of breaking `ragcore eval`. `[eval]` opt-in is the mitigation.

**Online eval scoring is awaited, not detached.** `chat_turn` *awaits* `maybe_score_turn` (`chat.py:43`) because the CLI runs one `asyncio.run` per turn and a detached task would be cancelled at loop teardown — so a sampled turn pays the (slow) judge latency inline. Documented trade-off, but it means sampling at a high rate visibly slows chat.

**Heavy optional deps and first-run network.** Embedded Weaviate downloads a server binary on first use; LLMLingua-2/CLIP/Outlines pull large models. All lazy-imported and opt-in, but "no Docker" is not "no download."

**Embedding model has no cloud fallback by design.** `routing.select_model` only escalates when a cloud model is configured for that role; embedding has none. A down local embedder takes down both ingest and retrieval with no escape hatch.

**The "500 gap" is closed, but the spec lied about how.** The web-UI design says `ConfigurationError → 400, StoreError/ProviderError → 502, else 500`. The catch-all in `app.py` only ever returns **400 or 502** — there is no 500 branch. The code is *stricter* than the spec; the spec is stale.

---

## 8. Requirements, setup, build & run

### System prerequisites

- **Python 3.12** (project pins `>=3.11,<3.13`), recommended via `uv`
- **SurrealDB v3** — start with the `rocksdb:` scheme. The `file:` scheme was removed in v3.
- **Ollama** serving a chat model + an embedding model: `ollama pull qwen3:8b && ollama pull nomic-embed-text`. Any pulled chat model works — set `[models.chat] local_model`.
- `OLLAMA_API_BASE` in `.env` (typically `http://localhost:11434`); `ANTHROPIC_API_KEY` only if using `--cloud`.

### One-time setup

```bash
python -m venv .venv && source .venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                               # or: uv pip install -e ".[dev]"
cp config.example.toml config.toml
cp .env.example .env
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### The two-terminal flow

SurrealDB must be running for everything below.

*Terminal A — database (leave running):*
```bash
surreal start --user root --pass root rocksdb:./data/db
```

*Terminal B — core features, end to end:*
```bash
ragcore init                                     # schema (once, or after a DB reset)

# Ingestion + source management
ragcore ingest examples/ragcore_demo.md          # file path
ragcore ingest https://example.com/page          # URL
ragcore ingest photo.png                         # CLIP-embed an image (needs [multimodal])
ragcore ingest meeting.m4a                        # transcribe + ingest audio (needs [audio])
ragcore list
ragcore search "how does retrieval work"
ragcore search --images "a red sunset"           # cross-modal CLIP text→image
ragcore remove source:abc123

# One-shot Q&A
ragcore ask "How does ragcore do retrieval, and where are embeddings stored?"
ragcore ask --cloud "Summarize everything"       # force cloud escalation
ragcore ask --agentic "Trade-offs across all docs?"   # corrective/self-RAG loop
ragcore ask --structured "Summarise the key findings" # schema-validated JSON answer
ragcore models

# Async ingestion + worker
ragcore ingest https://example.com/big --async
ragcore jobs ; ragcore jobs --status failed
ragcore retry ingestion_job:xyz

# Multi-turn chat (history-aware)
ragcore chat                                     # new session; /exit to quit
ragcore chat --session chat_session:abc
ragcore sessions

# Web UI (FastAPI + single HTML page, localhost, no auth)
ragcore serve                                    # http://127.0.0.1:8080
```

*Terminal C (optional) — background worker for `--async` ingests:*
```bash
ragcore worker          # poll loop
ragcore worker --once   # drain currently-due jobs and exit (cron/tests)
```

### Optional extras (the full menu)

```bash
pip install -e ".[rag-upgrade]"   # EVERYTHING below in one shot
# — or pick individually —
pip install -e ".[rerank]"        # FlashRank cross-encoder
pip install -e ".[chroma]"        # ChromaDB backend
pip install -e ".[faiss]"         # FAISS backend
pip install -e ".[milvus]"        # Milvus Lite backend
pip install -e ".[qdrant]"        # in-process Qdrant backend
pip install -e ".[hybrid]"        # Qdrant + fastembed BM42 sparse (server-side RRF)
pip install -e ".[pgvector]"      # Postgres + pgvector backend
pip install -e ".[weaviate]"      # Embedded Weaviate backend (downloads a binary)
pip install -e ".[eval]"          # Ragas + TruLens + DeepEval + pyyaml
pip install -e ".[multimodal]"    # CLIP image ingest
pip install -e ".[observability]" # Arize Phoenix OTel tracing
pip install -e ".[gateway]"       # LiteLLM unified gateway
pip install -e ".[dspy]"          # DSPy strategy-prompt optimizer
pip install -e ".[structured]"    # Instructor + Outlines
pip install -e ".[docai]"         # Docling + PyMuPDF document parsing
pip install -e ".[compress]"      # LLMLingua-2 context compression
pip install -e ".[audio]"         # mlx-whisper audio ingest
npm install -g promptfoo          # Promptfoo runner is a global Node CLI, not a pip dep
ollama pull moondream             # only if using the VLM image-caption assist
```

### Vector backends — pick one in `config.toml`

```toml
[store]
vector_backend = "surreal"   # surreal(default) | chroma | faiss | milvus | qdrant | pgvector | weaviate
chroma_path    = "data/chroma"
faiss_path     = "data/faiss"
milvus_uri     = "data/milvus.db"
qdrant_path    = "data/qdrant"
qdrant_hybrid  = false                # named dense+BM42-sparse + in-Qdrant RRF
qdrant_sparse_model = "Qdrant/bm42-all-minilm-l6-v2-attentions"
pgvector_dsn   = "postgresql://localhost/ragcore"
weaviate_path  = "data/weaviate"
weaviate_version = "1.30.5"           # embedded server binary pin
collection     = "ragcore"
```

pgvector host setup (no Docker):
```bash
brew install pgvector postgresql@17 && brew services start postgresql@17
createdb ragcore && psql ragcore -c "CREATE EXTENSION vector"
```

### Retrieval / generation features

```toml
[rerank]       { enabled = true, model = "ms-marco-MiniLM-L-12-v2", top_k = 5 }
[cache]        { enabled = true, threshold = 0.95 }
[graph]        { enabled = true, hops = 1 }
[query_rewrite]{ enabled = true, strategy = "multi_query", n = 3 }   # multi_query|hyde|decompose
[agentic]      { enabled = true, max_iterations = 2, min_relevant = 2 }
[structured]   { enabled = true, backend = "instructor" }            # instructor|outlines
[docai]        { enabled = true, parser = "docling" }                # docling|pymupdf
[compression]  { enabled = true, rate = 0.5, device = "cpu" }        # LLMLingua-2 (query-agnostic)
[audio]        { enabled = true, model = "mlx-community/whisper-tiny", timestamps = false }
[multimodal]   { model = "ViT-B-32", pretrained = "laion2b_s34b_b79k", device = "mps", vlm_enabled = false }
```
*(TOML inline tables shown compactly; expand to standard `[section]` blocks in your file — see `config.example.toml`.)*

### LLMOps / cost / eval / bench

```bash
# Evaluation (Ragas + TruLens, or framework="deepeval"; default golden v1 dataset)
ragcore eval                                     # writes data/eval/report.json
ragcore eval --tag v1                            # label this run, record an eval_run
ragcore eval --dataset path/to/custom.jsonl      # override the golden default

# LLMOps lifecycle
ragcore runs                                     # list recorded eval runs
ragcore gate  --baseline v1                      # exit 1 if regression beyond [llmops] tolerance
ragcore drift --baseline v1                      # exit 1 if centroid drift > threshold
ragcore promote <run-id>                         # set deployment:current (gate required)
ragcore promote <run-id> --no-gate               # force promote
ragcore rollback                                 # revert to previous deployment

# Cost tracking
ragcore cost report                              # spend by model, cache hit-rate, right-sizing hints

# Graph-RAG (triple extraction + retrieval)
ragcore graph build                              # back-fill triples for ingested sources

# Serving benchmark (MPS; GPU/vLLM serving is design-only on this host)
ragcore bench                                    # sweep num_ctx × num_batch × concurrency

# DSPy prompt optimization
ragcore optimize                                 # compile the strategy prompt → [dspy] compiled_path
```

```toml
[eval]   { judge_model = "ollama:qwen3:8b", framework = "ragas" }   # ragas|deepeval
[llmops] { tolerance = 0.05, drift_threshold = 0.15 }
[cost]   { enabled = true, enforce = false, budget_tokens = 0, ledger_path = "data/cost.db" }
[observability] { enabled = true, backend = "phoenix", otlp_endpoint = "http://localhost:6006/v1/traces" }
[gateway]{ enabled = true }
[dspy]   { enabled = true, compiled_path = "data/dspy_compiled.json" }
[online_eval] { enabled = true, sample_rate = 0.1, metrics = ["groundedness"] }
```

> **Online eval** has no dedicated command — it samples *live chat turns* (`ragcore chat`/web) when `[online_eval] enabled`, scores them with the offline judge in a worker thread, and lands the scores on an `online_eval` OTel span exported to Phoenix. It never affects (or fails) the turn.

> **Known pin tension:** `deepeval` (<8.4.0) and `dspy` (≥8.4 ok) disagree on `click`'s upper bound; `pip check` surfaces one benign metadata warning. Both import and the suite stays green either way (see the comment above the `eval` extra in `pyproject.toml`).

> **GPU/vLLM holdout:** production inference-optimization (continuous batching, paged KV-cache, AWQ/GPTQ via vLLM/TGI) requires CUDA. `bench/serving.py::vllm_serve` raises when `nvidia-smi` is absent. The MPS Ollama bench is the runnable artifact on this machine.

### Tests / lint (dev workflow)

```bash
# Deterministic suite (~63 test modules) — no Ollama; SurrealDB tests auto-skip if binary absent
pytest tests
ruff check ragcore tests

# Live suite (~17 modules) — requires Ollama at http://localhost:11434
pytest tests/live -m live
```

`tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch/faiss (macOS OpenMP duplicate-library crash); must precede any torch/faiss import.

---

### Appendix: where the code contradicts the docs

1. **Vector index.** The local-RAG design assumed an **HNSW** index — the *default* as-built implementation is a **brute-force cosine full scan** (`db/schema.surql:40`), with HNSW/MTREE the future seam. (HNSW *is* now reachable via the pgvector/Weaviate/Qdrant backends.)
2. **SurrealDB start command.** Design said `surreal start file:./data/db` — corrected to `rocksdb:` (v3 removed `file:`).
3. **Web error mapping.** Spec says `… else 500`; the implementation (`web/app.py`) only ever returns **400 or 502** — there is no 500 branch. The code is stricter than the spec.
4. **Weaviate "needs Docker."** The RAG v3 holdout note claimed Weaviate needed Docker; superseded in v4 — Embedded Weaviate runs a downloaded Darwin binary in-process (`weaviate_store.py:3`).
5. **Eval framework count.** Earlier notes said the `framework` selector chooses among Ragas, DeepEval *and* Promptfoo. As built, `_build_judge_for_framework` (`harness.py:90`) selects only **ragas|deepeval**; Promptfoo is a separate Node-CLI runner, not part of the selector.
6. **Observability filenames.** The v2 note referenced `observability/attributes.py` and `eval/promptfoo/config.yaml`; the as-built files are `observability/otel.py` (+`spans.py`, `tracing.py`) and `eval/promptfoo/promptfooconfig.yaml`.
7. **MVP scope drift (expected, not a bug).** The original design listed web UI, async worker, and chat-history as non-goals/YAGNI. All exist via later approved specs — the seams held.
