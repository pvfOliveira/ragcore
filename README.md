# ragcore

Local-first RAG core. Ingest documents/URLs, ask questions answered by a local
LLM over hybrid (vector + full-text) retrieval, with cloud escalation when needed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or use uv
pip install -e ".[dev]"
cp config.example.toml config.toml
cp .env.example .env   # set OLLAMA_API_BASE (and ANTHROPIC_API_KEY for cloud)

# local models (config.example.toml recommends qwen3:8b; any Ollama chat model works —
# set [models.chat] local_model to whatever you have pulled, e.g. qwen2.5:7b-instruct)
ollama pull qwen3:8b
ollama pull nomic-embed-text

# database (leave running in another terminal)
# NOTE: SurrealDB v3 uses rocksdb: storage (the file: scheme was removed)
surreal start --user root --pass root rocksdb:./data/db
```

## Use

```bash
ragcore init                              # create the SurrealDB schema
ragcore ingest path/to/document.pdf       # a file or URL (skips if already ingested)
ragcore list                              # list ingested sources (id, chunks, title, origin)
ragcore remove source:abc123              # delete a source (its embeddings go too)
ragcore search "your query"
ragcore ask "What does the document say about X?"
ragcore models                            # show configured model roles
```

Re-ingesting the same path/URL is a no-op (dedup by origin) — `remove` it first if
you want to re-index updated content.

## Worked example

With SurrealDB running and Ollama serving (see Setup), ingest the bundled sample
and ask a question about it:

```console
$ ragcore init
Schema initialized.

$ ragcore ingest examples/ragcore_demo.md
Ingested examples/ragcore_demo.md as source:0bvnjzqbqvi0u2y8fl7k

$ ragcore search "how does retrieval work"
[source:0bvnjzqbqvi0u2y8fl7k] # ragcore architecture
ragcore is a local-first Retrieval-Augmented Generation system. Ingestion extracts...

$ ragcore ask "How does ragcore do retrieval, and where are embeddings stored?"
Ragcore retrieves information through a combination of methods: it first runs a
vector similarity search and a BM25 full-text search independently. After obtaining
their respective rankings, these results are then fused using Reciprocal Rank
Fusion [source:1]. The embeddings used in this process are stored in SurrealDB
along with the context chunks from which they were derived [source:2].

Sources: source:0bvnjzqbqvi0u2y8fl7k
```

The answer is grounded in the ingested document and cites its sources — fully
local, no cloud call. Use `ragcore ask --cloud "..."` to force cloud escalation
(requires a `cloud_model` in `config.toml` and the provider's API key).

## Production RAG upgrade

Optional, config-selectable upgrades layered on the default SurrealDB hybrid
retrieval. Install the whole bundle with `pip install -e ".[rag-upgrade]"`, or
pick individual extras as shown below.

### Swappable vector backends

The vector index is pluggable behind a small async `VectorStore` protocol. Pick
a backend in `config.toml`; SurrealDB stays the source of truth and non-surreal
backends receive a mirrored write on ingest.

```toml
[store]
vector_backend = "chroma"     # surreal (default) | chroma | faiss | milvus
chroma_path    = "data/chroma"
faiss_path     = "data/faiss"
milvus_uri     = "data/milvus.db"
collection     = "ragcore"
```

```bash
pip install -e ".[chroma]"    # or ".[faiss]" / ".[milvus]"
```

### Reranking (FlashRank)

Cross-encoder reranking of the fused candidates, fully local:

```toml
[rerank]
enabled = true
model   = "ms-marco-MiniLM-L-12-v2"
top_k   = 5
```

```bash
pip install -e ".[rerank]"
```

### Semantic caching

A local sqlite cache keyed by query embedding; a near-duplicate query within
`threshold` cosine similarity reuses the prior answer:

```toml
[cache]
enabled   = true
threshold = 0.95
```

### Evaluation (`ragcore eval`)

Retrieval+answer quality scored with **Ragas** (faithfulness, answer_relevancy,
context_precision) and **TruLens** (groundedness, context_relevance,
answer_relevance), judged by a **local Ollama** model via litellm — no cloud
keys. The judge model is `[eval] judge_model` (default `ollama:qwen3:8b`; a
non-reasoning instruct model such as `ollama:qwen2.5:7b-instruct` is much faster
per metric).

```bash
pip install -e ".[eval]"
surreal start --user root --pass root rocksdb:./data/db   # in another terminal
ragcore ingest examples/ragcore_demo.md
ragcore eval                                              # writes data/eval/report.json
```

### Test tiers

- **Deterministic** (`pytest tests`): the full suite with stubbed judges and
  fixed vectors; no Ollama required. SurrealDB-backed tests auto-skip if the
  `surreal` binary is absent.
- **Live** (`pytest tests/live -m live`): exercises the real Ollama path —
  pluggable backends with real `nomic-embed-text` embeddings and the real
  Ragas+TruLens judge. Auto-skipped unless `http://localhost:11434` is reachable
  (gate in `tests/conftest.py`).

### Skills demonstrated

`file:line` artifacts for each capability. Live-verified entries were exercised
against a local Ollama (qwen3:8b / qwen2.5:7b-instruct + nomic-embed-text) by the
`tests/live` tier; structural-only entries are covered by the deterministic suite.

| Skill | Artifact | Verification |
| --- | --- | --- |
| Chroma backend | `ragcore/vectorstores/chroma_store.py:10` (`ChromaStore`); factory `ragcore/vectorstores/base.py:63` | Live — real-embedding roundtrip, `tests/live/test_live_rag_upgrade.py:75` |
| FAISS backend | `ragcore/vectorstores/faiss_store.py:18` (`FaissStore`); factory `ragcore/vectorstores/base.py:72` | Live — real-embedding roundtrip, `tests/live/test_live_rag_upgrade.py:79` |
| Milvus backend | `ragcore/vectorstores/milvus_store.py:20` (`MilvusStore`); factory `ragcore/vectorstores/base.py:77` | Live — real-embedding roundtrip, `tests/live/test_live_rag_upgrade.py:83` |
| Reranking (FlashRank) | `ragcore/rerank.py:7` (`rerank`); config `ragcore/config.py:57` | Structural — `tests/test_rerank.py` |
| Semantic caching | `ragcore/cache.py:22` (`SemanticCache`); config `ragcore/config.py:63` | Structural — `tests/test_cache.py` |
| Model evaluation | `ragcore/eval/harness.py:45` (`compute_metrics`); CLI `ragcore/cli.py:278` | Live — full 6-metric report from the local Ollama judge, all floats in [0,1] (`tests/live/test_live_rag_upgrade.py:88`, passing) |
| Ragas | `ragcore/eval/harness.py:161` (`_score_ragas`) | Live (qwen2.5:7b-instruct) — faithfulness=1.0, answer_relevancy≈0.85, context_precision≈0.50 on a grounded record |
| TruLens | `ragcore/eval/harness.py:173` (`_score_trulens`); litellm shim `ragcore/eval/harness.py:62` | Live (qwen2.5:7b-instruct) — groundedness=1.0, context_relevance=0.5, answer_relevance=1.0 on a grounded record |

> Live-eval notes (Task 10): first real exercise of the `judge=None` path
> surfaced three fixable mismatches, all fixed in `ragcore/eval/harness.py`:
> (1) trulens-vs-litellm instrumentation crash on `litellm.CallTypes`
> (`_patch_trulens_litellm_instrumentation`); (2) Ollama rejecting ragas'
> `n`-sampling param (`litellm.drop_params = True`); (3) `answer_relevancy`
> needing an embeddings model (wired from `[models.embedding]`). The
> per-metric `LLM` judge calls are slow under a reasoning model (qwen3:8b);
> use a non-reasoning instruct model for `ragcore eval`. The end-to-end
> `ragcore eval` (which also runs the LangGraph answer step) can hit esperanto's
> default Ollama HTTP read-timeout under qwen3:8b — independent of the metrics,
> which are verified above.

## LLM platform

Optional capabilities installed with `pip install -e ".[platform]"` (or the
individual extras below). All are opt-in via `config.toml` and do not affect
the default ingest/ask/eval paths unless explicitly enabled.

### LLMOps lifecycle

An `eval_run` registry (SurrealDB) records every evaluation with its metrics,
config snapshot, and an optional tag. A `deployment:current` pointer tracks
which run is live, and a history list supports one-level rollback.

```toml
[llmops]
tolerance       = 0.05   # max allowed metric regression vs baseline
drift_threshold = 0.15   # cosine-distance drift that fails `drift`
```

```bash
ragcore eval --tag v2            # run eval and record a named run
ragcore runs                     # list all recorded eval runs
ragcore gate  --baseline v1      # exit nonzero if any metric regressed vs baseline
ragcore drift --baseline v1      # report embedding centroid drift vs baseline
ragcore promote <run-id>         # set deployment:current (gate required unless --no-gate)
ragcore rollback                 # revert deployment:current to the previous run
```

### Cost optimization

Per-request token accounting at the LLM call boundary, recorded in a
local SQLite ledger. `ragcore cost report` aggregates spend by model,
cache hit-rate and tokens avoided, plus right-sizing hints.

```toml
[cost]
enabled       = false    # record usage into the ledger
enforce       = false    # block requests over budget (else warn only)
budget_tokens = 0        # 0 = no budget
ledger_path   = "data/cost.db"

[cost.rates]
"anthropic:claude-3-5-sonnet-20241022" = 3.0   # USD per 1k tokens (local = omit)
```

```bash
ragcore cost report              # spend by model, cache hit-rate, right-sizing hints
```

### Serving benchmark (inference-optimization)

Sweeps the cartesian product of `num_ctx × num_batch × concurrency` against a
local Ollama server on MPS, reporting latency (TTFT + total) and aggregate
throughput.

```toml
[bench]
num_ctx     = [2048, 4096]
num_batch   = [128, 256]
concurrency = [1, 2, 4]
keep_alive  = "5m"
prompt      = "Summarize the theory of relativity in three sentences."
```

```bash
ragcore bench                    # sweep and print latency/throughput report
```

> **GPU/vLLM holdout:** production inference-optimization (continuous batching,
> paged KV-cache, AWQ/GPTQ quantization via vLLM or TGI) requires CUDA and is
> a documented design-only holdout on this MPS host.
> `ragcore/bench/serving.py::vllm_serve` raises at runtime when `nvidia-smi`
> is absent. The MPS Ollama bench (`ragcore bench`) is the runnable artifact
> on this machine.

### Graph-RAG

LLM triple extraction at ingest time populates a SurrealDB entity/relation
graph. At query time, entities mentioned in the query seed a graph traversal
and the connected chunks are injected as a third RRF signal alongside vector
and BM25 results.

```toml
[graph]
enabled = false   # enable graph-RAG traversal at query time
hops    = 1       # traversal depth
```

```bash
ragcore graph build              # back-fill the graph for already-ingested sources
```

Enable `[graph] enabled = true` before ingesting to extract triples
automatically; or run `ragcore graph build` to back-fill an existing corpus.

### Multimodal (CLIP)

CLIP-embeds images at ingest time and stores them in a `image_embedding`
SurrealDB table. `ragcore search --images` runs a cross-modal text→image query
using a SurrealDB-side cosine function.

```bash
pip install -e ".[multimodal]"   # adds open_clip, torch, Pillow

ragcore ingest photo.png         # CLIP-embed an image and store it
ragcore search --images "a red sunset"   # cross-modal text → image retrieval
```

Configure the CLIP model and device in `config.toml`:

```toml
[multimodal]
model      = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
device     = "mps"   # mps | cuda | cpu
```

### Test tiers (platform)

- **Deterministic** (`pytest tests`): 162 tests — stubs all LLM calls and
  SurrealDB interactions; no Ollama required. SurrealDB-backed tests
  auto-skip if the `surreal` binary is absent.
- **Live** (`pytest tests/live -m live`): 5 end-to-end platform tests
  (llmops, cost, bench, graph, multimodal) over real Ollama + SurrealDB +
  CLIP. Auto-skipped unless `http://localhost:11434` is reachable.

### Skills demonstrated

`file:line` artifacts for the platform layer. Each `file:line` was verified
by reading the file.

| Skill | Artifact | Status |
| --- | --- | --- |
| llmops | `ragcore/llmops/registry.py:27` (`RunRegistry.record`); `ragcore/llmops/gates.py:19` (`check_gate`), `ragcore/llmops/gates.py:44` (`check_drift`); `ragcore/llmops/deploy.py:43` (`DeploymentStore.promote`) | proven (deterministic + live) |
| ai-cost-optimization | `ragcore/cost/ledger.py:7` (`CostLedger`); `ragcore/cost/report.py:5` (`build_report`) | proven (deterministic + live) |
| graph-rag | `ragcore/graph.py:33` (`extract_triples`), `ragcore/graph.py:89` (`GraphStore`), `ragcore/graph.py:273` (`graph_context`); `ragcore/retrieve.py:29` (`fuse_with_graph`) | proven (deterministic + live) |
| multimodal-ai | `ragcore/multimodal.py:58` (`ClipEmbedder`), `ragcore/multimodal.py:33` (`cross_modal_rank`), `ragcore/multimodal.py:154` (`ImageStore`) | proven (deterministic + live) |
| inference-optimization | `ragcore/bench/harness.py:30` (`run_bench`); `ragcore/bench/serving.py:39` (`vllm_serve` — CUDA-gated design stub) | MPS bench proven; GPU serving aspirational (holdout) |

## RAG v3 (opt-in)

Five advanced retrieval capabilities, all disabled by default. Enable each
independently in `config.toml`; the default ingest/ask paths are unchanged.

### Query rewriting

Rewrites the user query before retrieval using one of three strategies:
`multi_query` (generate N paraphrases and union the results), `hyde`
(hallucinate a hypothetical answer and embed it), or `decompose` (split a
compound question into sub-queries and merge results).

```toml
[query_rewrite]
enabled  = true
strategy = "multi_query"   # multi_query | hyde | decompose
n        = 3
```

### Agentic RAG

Iterative retrieval loop: the LLM decides whether the retrieved context is
sufficient or whether to re-query before producing a final answer. Useful for
multi-hop questions.

```bash
ragcore ask --agentic "What are the trade-offs discussed across all documents?"
```

```toml
[agentic]
enabled        = true
max_iterations = 2
min_relevant   = 2
```

### Structured generation

Forces the LLM to return a validated Pydantic schema via **Instructor** (JSON
mode) or **Outlines** (constrained token sampling). Activated with the
`--structured` flag:

```bash
ragcore ask --structured "Summarise the key findings"
```

```toml
[structured]
enabled = true
backend = "instructor"   # instructor | outlines
```

```bash
uv pip install -e '.[structured]'   # pulls instructor + outlines
```

### Qdrant vector backend

In-process Qdrant (no server required) as an alternative to the default
SurrealDB vector index. All existing backends (chroma, faiss, milvus) remain
available.

```toml
[store]
vector_backend = "qdrant"
qdrant_path    = "data/qdrant"
```

```bash
uv pip install -e '.[qdrant]'
```

> **Weaviate holdout:** Weaviate requires a running Docker daemon and is a
> documented design holdout — the adapter interface is defined but not wired
> to a live instance on this host.

### Document AI / OCR + VLM captions

`ragcore ingest` gains a richer extraction path: **Docling** (layout-aware PDF
parsing) or **PyMuPDF** for OCR, plus a small Ollama vision model
(moondream) to caption embedded images and inject them as searchable text.

```toml
[docai]
enabled = true
parser  = "docling"   # docling | pymupdf

[multimodal]
vlm_enabled = true
vlm_model   = "moondream"
```

```bash
uv pip install -e '.[docai,structured]'
ollama pull moondream
```

## Web UI

A thin local chat UI (FastAPI + one HTML page, no build step):

~~~bash
surreal start --user root --pass root rocksdb:./data/db   # in another terminal
ragcore serve            # http://127.0.0.1:8080  (localhost only, no auth)
~~~

Open the URL: add sources (URL/path) in the sidebar, start a chat, switch sessions.
It is unauthenticated and bound to localhost — do not expose it to a network as-is.
