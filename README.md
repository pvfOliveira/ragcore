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

## Web UI

A thin local chat UI (FastAPI + one HTML page, no build step):

~~~bash
surreal start --user root --pass root rocksdb:./data/db   # in another terminal
ragcore serve            # http://127.0.0.1:8080  (localhost only, no auth)
~~~

Open the URL: add sources (URL/path) in the sidebar, start a chat, switch sessions.
It is unauthenticated and bound to localhost — do not expose it to a network as-is.
