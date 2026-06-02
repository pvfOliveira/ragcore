# ragcore — Architecture Deep Dive

*A Senior Staff Engineer's walkthrough and critical review. Everything below is grounded in code that actually exists in this repo; file:line citations point at the real thing.*

---

## 1. What it is

`ragcore` is a local-first Retrieval-Augmented Generation engine you can hold in your head. You feed it files or URLs; it extracts, chunks, embeds, and stores them in a single SurrealDB instance; then it answers questions over that corpus with a local LLM (Ollama), fusing vector and full-text search and escalating to a cloud model only when content is too large or you ask it to. It ships as three faces over one library: a Typer CLI (`ragcore.cli`), a small FastAPI + vanilla-JS web UI (`ragcore.web`), and the importable `ragcore` package itself — which the design docs rightly call "the product." It is deliberately single-user, single-machine, no-Docker, and small enough that one developer can own every line.

## 2. The big picture

Think of ragcore as a **personal research librarian working at a single desk**. When you hand over a document (`ingest`), the librarian photocopies it into index cards (chunks), files each card under two separate catalog systems — a "meaning" catalog (vector embeddings) and a "keyword" catalog (BM25 full-text) — and stores the originals in one filing cabinet (SurrealDB). When you ask a question (`ask`), the librarian first plans which catalog lookups to run, pulls cards from *both* catalogs in parallel, reconciles the two ranked piles into one shortlist (Reciprocal Rank Fusion), and writes you a cited answer. For a casual back-and-forth (`chat`), the same librarian keeps a running memory of the conversation so "what about its license?" still makes sense.

The mental model that matters: **the library is one async Python module per concept, and SurrealDB is the one cabinet that holds everything** — documents, embeddings, the job queue, and chat history all live as tables in the same database. There is no message broker, no cache tier, no second datastore. The "smart" parts (which model, local vs. cloud) are *injected*, not hardwired, so the librarian can swap pens without relearning how to file.

## 3. Architecture & data flow

```
            ┌──────────────┐   ┌─────────────────────┐   ┌──────────────────┐
  YOU  ──►  │  cli.py      │   │  web/app.py (FastAPI)│   │  worker.py loop  │
            │  (Typer)     │   │  + static/index.html │   │  (background)    │
            └──────┬───────┘   └──────────┬──────────┘   └────────┬─────────┘
                   │                       │                       │
                   └───────────┬───────────┴───────────┬───────────┘
                               ▼                       ▼
                    ┌─────────────────────── ragcore library ───────────────────────┐
                    │  ingest.py   chat.py   ask.py(LangGraph)   retrieve.py(RRF)     │
                    │  chunking.py   embedding.py   routing.py   errors.py            │
                    │                  providers.py  ──►  esperanto AIFactory         │
                    └───────┬───────────────────────────────────────────┬───────────┘
                            │ Store / JobQueue / SessionStore            │ build_chat / build_embedding
                            ▼                                            ▼
                 ┌──────────────────────┐                  ┌──────────────────────────┐
                 │      SurrealDB 3.x    │                  │  Ollama (qwen / nomic)   │
                 │  rocksdb:./data/db    │                  │  + cloud (Anthropic)     │
                 │  source, source_embedding,│              └──────────────────────────┘
                 │  ingestion_job, chat_*    │
                 │  fn::vector_search,       │
                 │  fn::text_search (BM25)   │
                 └──────────────────────────┘
```

**Walking an `ingest`** (`ragcore/ingest.py:32`):
1. `ingest_source` first asks the store `find_source_id_by_origin` (`store.py:81`). If this exact path/URL was already ingested, it short-circuits and returns `created=False` — dedup is by `origin`, by design.
2. `_extract` (`ingest.py:18`) calls `content_core.extract_content` to turn a file or URL into markdown plus a title.
3. `chunk_text` (`chunking.py:80`) splits the markdown into ~400-token chunks with 60-token overlap, recursively on paragraph → line → sentence → word boundaries, dropping fragments under `min_chunk_size` (5 tokens).
4. `store.create_source` writes the `source` row; `generate_embeddings` (`embedding.py:38`) batches the chunks through the embedding model with 3× retry; `store.add_embeddings` (`store.py:62`) writes one `source_embedding` row per chunk, each holding the chunk text *and* its vector.
5. With `--async`, none of that runs inline: the CLI instead calls `JobQueue.enqueue` (`jobs.py:38`), a row lands in `ingestion_job`, and a separate `ragcore worker` process does steps 2–4 later.

**Walking an `ask`** (`ragcore/ask.py:103`) — this is a compiled LangGraph:
1. **strategy node** (`ask.py:56`): the chat model turns your question into up to 5 focused search terms, returned as JSON. If parsing fails, it falls back to `[question]` — a nice belt-and-suspenders move.
2. **fan-out** (`ask.py:68`): `_fan_out` emits a LangGraph `Send` per term, running `retrieve_answer` in parallel.
3. **retrieve_answer node** (`ask.py:72`): each term runs `hybrid_search` (`retrieve.py:29`) — a vector search and a BM25 text search, fused by Reciprocal Rank Fusion — then the chat model drafts a partial answer over those chunks. Partial answers accumulate via `Annotated[list, operator.add]` (`ask.py:47`), LangGraph's reducer pattern.
4. **synthesize node** (`ask.py:82`): one final model call merges the partials into a cited answer; citations are the deduped `source` ids.

`chat` (`ragcore/chat.py:18`) is intentionally *not* a graph — it's a linear async function: pull history → (if any) reformulate the follow-up into a standalone query → `hybrid_search` → answer with history in the prompt → persist user + assistant turns to `chat_message`.

## 4. Codebase tour

Dependency direction is strict and one-way: **cli/web/worker → library → store → SurrealDB**, with `providers`/`routing` injected into `ingest`, `ask`, and `chat`.

| File | Job |
|---|---|
| `config.py` | Pydantic models for TOML config; `load_config` enforces that `[models.chat]` and `[models.embedding]` exist. |
| `errors.py` | Typed exception tree (`RagcoreError` → Configuration/Provider/Store); `classify_error` maps raw exceptions to friendly text; `is_transient` decides retryability. |
| `chunking.py` | tiktoken-based `token_count`, content-type detection, recursive token-bounded splitter with overlap. |
| `routing.py` | `select_model(config, role, content, force_cloud)` — the local-vs-cloud policy, the project's headline feature. |
| `providers.py` | Thin wrapper over esperanto's `AIFactory`; `_ChatAdapter` exposes a uniform `async ainvoke(prompt) -> .content`. |
| `embedding.py` | Batched embeddings with retry + mean-pooling for oversized text. |
| `store.py` | SurrealDB repository: schema init, source CRUD, `vector_search`/`text_search` via DB functions. |
| `retrieve.py` | `reciprocal_rank_fusion` + `hybrid_search`. |
| `ingest.py` | Synchronous extract→chunk→embed→store pipeline. |
| `ask.py` | The LangGraph one-shot Q&A graph. |
| `chat.py` | Linear history-aware multi-turn RAG. |
| `sessions.py` | `SessionStore`: chat sessions + messages (mirrors Store/JobQueue). |
| `jobs.py` | `JobQueue`: the home-grown SurrealDB-table queue + retry/backoff bookkeeping. |
| `worker.py` | `run_worker`: claim → `ingest_source` → mark done/retry/failed loop. |
| `cli.py` | Typer entrypoint; every command is a thin `asyncio.run(...)` over a library function. |
| `web/app.py` | FastAPI app-factory; DI seams (`get_config`/`get_store`/…) for testing; one catch-all exception handler. |
| `web/static/index.html` | The entire frontend: HTML + CSS + vanilla `fetch()` in one file, no build step. |
| `db/schema.surql` | Tables, BM25 analyzer/indexes, `fn::vector_search`/`fn::text_search`, and cascade-delete events. |

A clean reuse touch: `chat.py` imports `_build_chat` and `_clean` directly from `ask.py` (`chat.py:8`) so model provisioning and `<think>`-stripping aren't duplicated. The three repository classes (`Store`, `JobQueue`, `SessionStore`) are near-identical in shape — connection-per-operation, `_signin`, `_rows` helper, errors wrapped in `StoreError` — which is the kind of boring consistency that makes a codebase fast to read.

## 5. Tech stack & why

- **Python 3.11–3.12, async throughout.** The whole library is `async def`; the CLI bridges to it with one `asyncio.run` per command (`cli.py`). This avoids open-notebook's documented "fragile sync/async bridging" with `new_event_loop()` hacks (design §9).
- **SurrealDB 3.x (rocksdb backend), one instance.** Multi-model: it's the document store, the vector store (`vector::similarity::cosine`), the BM25 full-text engine, the job queue, *and* the chat history — all in SurrealQL. One cabinet, no glue.
- **esperanto `AIFactory`** for provider abstraction. ragcore never imports Ollama or Anthropic SDKs directly; `providers.py` is the only file that knows esperanto exists.
- **Ollama for local inference** (`qwen3:8b`/`qwen2.5:7b-instruct` chat, `nomic-embed-text` embeddings) over its OpenAI-compatible endpoint. Two resident models, not seven — unified memory on an M2 is the binding constraint (design §8).
- **LangGraph** for the `ask` graph — strategy → parallel fan-out → synthesize — mirroring open-notebook's proven shape.
- **content-core** for extraction (files/URLs → markdown), **tiktoken** for token counting, **ai-prompter/Jinja2** for editable prompt templates (`prompts/*.jinja`), **Typer** for the CLI, **FastAPI + uvicorn** for the web layer, **loguru** for worker logs, **Pydantic** for typed config and request bodies.

## 6. Architectural review (alternatives matrix)

I'll be opinionated. For each *real* decision, here are two industry-standard alternatives and the honest trade-offs.

### Category: Multi-model datastore — **chose SurrealDB**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Postgres + pgvector + tsvector** | RAG over one DB that also does vectors and FTS | Battle-tested, HNSW/IVFFlat indexes built in, huge ops ecosystem, real transactions | Heavier to install/run locally; no native graph; two query dialects (SQL + vector ops) feel bolted on |
| **SQLite + sqlite-vec + FTS5** | Truly zero-server, single-file, perfect for single-user local | Embeds in-process (no `surreal start` terminal), trivial backup, FTS5 BM25 is excellent | No graph; vector support is young; concurrency story weak; no record links / events |

**Verdict:** SurrealDB is a *learning-forward* pick. It buys a genuinely unified model (record links, `DEFINE EVENT` cascades, BM25, cosine, future graph) in one binary with one connection URL — exactly the "one cabinet" ethos. The cost is real: it's a younger product with sharper edges (see the `created`-tie and `ORDER BY`-projection traps in §7), brute-force vector scan in this schema, and you must run a separate process. For a single-user local tool that's a fair trade; for a multi-tenant service I'd reach for Postgres without hesitation.

### Category: AI provider abstraction — **chose esperanto's AIFactory**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **LangChain chat/embeddings models** | Already pulling in `langchain-core` for LangGraph | Enormous provider coverage, ecosystem integrations | Heavy, churny APIs; would couple the whole app to LangChain's abstractions |
| **Raw provider SDKs (ollama-python, anthropic)** | Maximum control, minimal deps | No abstraction tax, exact feature access | You hand-write the local↔cloud switch and re-implement it per provider; swapping providers becomes code, not config |

**Verdict:** esperanto is the right call *because* of `providers.py`'s discipline — it's the single seam, ~35 lines, and the `_ChatAdapter` deliberately uses esperanto's native `achat_complete` to avoid pulling in optional `langchain_ollama`/`langchain_anthropic` integrations (`providers.py:13`). The risk is supply-chain: ragcore's flexibility is hostage to a relatively niche library. The mitigation is exactly the thin-wrapper design — if esperanto vanished, you'd rewrite one file.

### Category: Ask orchestration — **chose LangGraph**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Plain async functions + `asyncio.gather`** | The fan-out is just "run N retrievals in parallel" | Zero framework, fully transparent, trivial to test; `chat.py` already proves it works | You re-implement state-merge/reducers by hand; lose graph visualization/checkpointing if you ever want them |
| **A heavier agent framework (CrewAI/AutoGen)** | If the strategy step grew into a true multi-agent loop | Built-in roles, tool-calling, memory | Massive overkill here; opaque control flow; fights the "understandable in one sitting" goal |

**Verdict:** LangGraph earns its keep *only* because of the parallel fan-out with an additive reducer (`Annotated[list, operator.add]`, `ask.py:47`) and `Send` (`ask.py:69`). That said, it's telling that `chat.py` consciously rejected LangGraph for a linear flow (chat design §3) — a good signal the team applies the framework only where its concurrency model pays off. If `ask` ever loses its fan-out, plain `asyncio.gather` would be simpler.

### Category: Background job queue — **chose an own SurrealDB-table queue**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Celery + Redis/RabbitMQ** | Industry default for Python task queues | Mature retries, scheduling, monitoring, scale-out | A broker + worker convention + new infra for a single-user tool — absurd weight here |
| **`surreal-commands`** (the lfnovo lib open-notebook uses) | Already SurrealDB-native, live-query push, built-in retries | No reinvention; push instead of poll | Another dependency + its worker-entrypoint conventions; less transparent than ~150 lines you own |

**Verdict:** Rolling their own (`jobs.py`, ~180 lines) is the most defensible "build vs. buy" in the repo, and the spec argues it well. The `claim_next` design (`jobs.py:62`) is genuinely thoughtful: SELECT-oldest-then-conditional-`UPDATE ... WHERE status='queued'`, treating both an empty result *and* a transaction conflict as "someone else got it." Backoff is stored, not scheduled — a retrying job is just `queued` with a future `next_attempt_at` (`jobs.py:107`), gated in the claim query. The honest cost: it polls (2s default), it's sequential by design (the local embedder is the bottleneck), and the dedup carries a benign TOCTOU window the code explicitly accepts (`jobs.py:39`).

### Category: Local inference runtime — **chose Ollama (GGUF/llama.cpp under the hood)**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **MLX (Apple's framework) directly** | Native Apple-Silicon, uses the unified-memory + Metal/ANE path | Often faster prefill/throughput on M-series; lower memory overhead; quantization tuned for Apple GPUs | Apple-only; smaller model/tooling ecosystem; no drop-in OpenAI-compatible server; you'd write more glue |
| **llama.cpp server directly (no Ollama)** | Same GGUF engine Ollama wraps | Full control over flags (context, KV cache, GPU layers), leaner | Manual model management, no friendly pull/registry, more ops per model |

**Verdict:** Ollama is the pragmatic local default — one `ollama pull`, an OpenAI-compatible endpoint esperanto already speaks, automatic model load/unload. The **MLX vs. GGUF/llama.cpp** trade-off is the interesting one on Apple Silicon: MLX can be measurably faster and lighter on memory because it's built for the unified-memory/Metal path, but it's macOS-only and lacks Ollama's ergonomics and broad model registry. GGUF via Ollama trades some raw M2 throughput for portability and a frictionless DX. For a tool whose whole pitch is "understandable and runs anywhere a dev sits," Ollama is right; an MLX provider behind `providers.py` would be a clean future optimization for Apple users — and the abstraction means it's a new provider, not a rewrite.

### Category: Web layer — **chose FastAPI + a single vanilla-JS page**

| Alternative | Why it fits | Advantages | Disadvantages |
|---|---|---|---|
| **Next.js/React SPA** | Rich chat UX, streaming, components | Best-in-class interactivity, ecosystem | A build step, node toolchain, and a second language/runtime for a localhost tool — violates "no build step" |
| **Streamlit / Gradio** | Fastest path to an ML demo UI | Almost no frontend code | Opinionated layout, heavier runtime, harder to make feel like a real app; awkward for session-switching UX |

**Verdict:** FastAPI + one `index.html` is exactly scaled to the job. The app-factory + dependency-overrides pattern (`app.py:56`) makes the HTTP layer testable with `TestClient` and zero real DB/LLM — a textbook seam. The frontend is one file of `fetch()` calls. The honest weaknesses: no streaming (answers appear all at once), synchronous ingest on `POST /api/sources` (a few seconds of blocked request), and it's unauthenticated/localhost-only by explicit design — fine for one user, dangerous if exposed.

## 7. Lessons

**Best practices it demonstrates.**
- **One seam per concern, injected not hardwired.** `routing.select_model` and `providers.build_*` are the only places that know about local-vs-cloud or which SDK. Everything downstream takes `config` and an `embedder_fn`. This is why adding MLX, or a quality-based routing rule, is a one-file change.
- **Editable prompts, never `&'static str`.** Every model-facing prompt is a Jinja template under `prompts/`, rendered via ai-prompter (`ask.py:26`). Consistent with the team's "LLM prompts must be user-editable" principle.
- **Friendly errors as a first-class layer.** `classify_error` (`errors.py:21`) turns "connection refused" into "Could not reach the model provider. Is the local server running?" — surfaced in the CLI (`cli.py:31`) and as JSON in the web UI.

**Clever bits worth stealing.**
- **Transient-vs-permanent retry classification.** `is_transient` (`errors.py:39`) is a *separate* predicate from `classify_error`, because the catch-all returns `ProviderError` and you can't infer retryability from the class — a `ValueError("No content extracted")` must fail permanently while a timeout retries. The worker uses `is_transient` to gate exponential backoff and `classify_error` only for the stored message (`worker.py:36`). That split is subtle and correct.
- **Race-safe claim without fancy locking** (`jobs.py:62`), treating empty-result and transaction-conflict identically.
- **Cascade deletes in the database, not the app.** `DEFINE EVENT ... WHEN ($after == NONE)` on both `source` and `chat_session` (`schema.surql:14,73`) means deleting a parent reaps its children server-side — the app just issues one `DELETE`.

**Real pitfalls (and how this code dodges them).**
- **The SurrealDB 3.x `created`-tie ordering trap.** Two rows created in the same instant tie on `ORDER BY created`. The store breaks the tie with a secondary key — `ORDER BY created DESC, id DESC` (`store.py:103`, `jobs.py:159`, `sessions.py:84`) — and `chat_message` orders by an explicit integer `seq` (`schema.surql:67`, written in `sessions.py:47`) rather than trusting timestamps. The `seq` carries an accepted single-writer assumption (`sessions.py:46`).
- **`ORDER BY` requires the field in the SELECT projection** on this server — note how `claim_next` selects `created` even though it only needs `id` (`jobs.py:73`). The retry spec calls this out explicitly.
- **Brute-force cosine vs. HNSW.** `fn::vector_search` does a full-table cosine scan (`schema.surql:40`), with a comment pointing at the MTREE/HNSW swap "if the corpus outgrows a full scan." Fine for a personal corpus, O(n) and a future cliff for a large one.
- **The sync/async boundary** is kept at exactly one place per entrypoint (`asyncio.run` in each CLI command); the library is uniformly async, avoiding nested-loop hazards.

**What a good engineer notices (honest weaknesses).**
- **The "500 gap" is closed, but the design doc lied about how.** The web-UI spec (§3) says non-config errors map to 502 "else 500." In reality the catch-all handler in `app.py:65` only ever returns **400 (config) or 502 (everything else)** — there is no 500 path, so a raw unhandled 500 can't leak a stack trace. The code is *better* than the spec; the spec is stale.
- **Connection-per-operation** means every store call opens a fresh WebSocket, signs in, and closes it. For one user that's fine and the design names pooling as the documented future seam — but it's real per-call latency, and `add_embeddings` issues one `CREATE` per chunk in a loop (`store.py:69`) rather than a bulk insert.
- **No streaming anywhere** — `ask`/`chat` block until the full answer is built. On a local 8B model a multi-search `ask` can take a while with nothing on screen.
- **Embedding model has no cloud fallback by design** (`routing` only escalates when a cloud model is configured for the role; embedding has none), so a down local embedder takes down both ingest and retrieval with no escape hatch.

## 8. Requirements, setup, build & run

**System prerequisites**
- **Python 3.12** (project pins `>=3.11,<3.13`), ideally via `uv`.
- **SurrealDB v3** (verified here: `3.1.2`). Start it with the **`rocksdb:`** scheme — the `file:` scheme was removed in v3, so the design doc's `surreal start file:./data/db` is wrong; the README is right.
- **Ollama** serving a chat model and an embedding model. The README suggests `qwen3:8b`; any pulled chat model works (e.g. `qwen2.5:7b-instruct`) — just set `[models.chat] local_model` to match. Embeddings use `nomic-embed-text`.
- A `config.toml` (copy from `config.example.toml`) and a `.env` with `OLLAMA_API_BASE` (and `ANTHROPIC_API_KEY` if you use `--cloud`).

**One-time setup**
```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"                                # or: uv pip install -e ".[dev]"
cp config.example.toml config.toml
cp .env.example .env                                   # set OLLAMA_API_BASE (+ ANTHROPIC_API_KEY for --cloud)
ollama pull qwen3:8b          # or qwen2.5:7b-instruct, then edit [models.chat] local_model
ollama pull nomic-embed-text
```

**The two-terminal flow.** SurrealDB must be running in its own terminal for everything below.

*Terminal A — the database (leave running):*
```bash
surreal start --user root --pass root rocksdb:./data/db
```

*Terminal B — every feature, end to end:*
```bash
# --- schema ---
ragcore init                                   # create tables, indexes, functions

# --- synchronous ingest + source management ---
ragcore ingest examples/ragcore_demo.md        # extract -> chunk -> embed -> store (blocks)
ragcore ingest https://example.com/page        # a URL works too
ragcore list                                   # id, chunk count, title, origin
ragcore search "how does retrieval work"       # hybrid (vector + BM25) chunk preview
ragcore remove source:abc123                   # delete a source (embeddings cascade away)

# --- one-shot Q&A ---
ragcore ask "How does ragcore do retrieval, and where are embeddings stored?"
ragcore ask --cloud "Summarize everything across all sources"   # force cloud escalation
ragcore models                                 # show configured roles (local + cloud)

# --- async ingestion via the job queue ---
ragcore ingest https://example.com/big --async # enqueue, prints a job id, returns immediately
ragcore jobs                                   # list all jobs, newest first (status, attempts)
ragcore jobs --status failed                   # filter by queued|running|done|failed
ragcore retry ingestion_job:xyz                # requeue a failed job (resets attempts)

# --- multi-turn chat (history-aware) ---
ragcore chat                                   # new session; prints its id
#   you> What is ragcore?
#   you> What about its license?              # follow-up resolved against history
#   you> /exit
ragcore chat --session chat_session:abc        # resume an existing session
ragcore sessions                               # list sessions (id, msg count, title)
```

*Terminal C (optional) — the background worker, for `--async` ingests:*
```bash
ragcore worker          # poll loop; processes queued jobs until Ctrl-C
ragcore worker --once   # drain currently-due jobs once and exit (good for cron/tests)
```
Note from the retry design: `--once` drains only jobs whose `next_attempt_at` is due; a job still backing off is left for a later run.

**The web UI** (a third face on the same library):
```bash
# Terminal A: surreal start ... (as above)
ragcore serve                       # http://127.0.0.1:8080  (localhost only, NO auth)
ragcore serve --host 127.0.0.1 --port 8080   # explicit host/port
```
In the browser: add sources by URL/path in the sidebar (synchronous ingest), start a chat, switch sessions, and **rename (✎) or delete (✕) sessions** — the two session controls the CLI doesn't expose. Do not put this on a network as-is.

**Tests / lint** (dev workflow):
```bash
pytest                  # unit + live-SurrealDB integration tests (asyncio_mode=auto)
ruff check ragcore tests
```

---

### Appendix: where the code contradicts the docs

1. **Vector index.** ~~The local-RAG design (§4) assumed an **HNSW** vector index~~ — *corrected 2026-06-02*: the design now reflects the as-built **brute-force cosine full scan** (`db/schema.surql`), with HNSW/MTREE noted as the future seam. (The README's worked example was never index-specific.)
2. **SurrealDB start command.** ~~Design §10 said `surreal start file:./data/db`~~ — *corrected 2026-06-02* to `rocksdb:` (SurrealDB v3 removed the `file:` scheme). README and config were already correct.
3. **Web error mapping.** The web-UI design (§3) says errors map "`ConfigurationError`→400, `StoreError`/`ProviderError`→502, else 500." The implementation (`web/app.py:69`) only ever returns **400 or 502** — there is no 500 branch, which is the point: it closes the raw-500 gap. The code is stricter than the spec.
4. **MVP scope drift (expected, not a bug).** The original design listed web UI, async worker, and chat-history as explicit non-goals/YAGNI (§1, §12). All three now exist via later approved specs — the seams held, which is the design vindicating itself rather than a contradiction.
