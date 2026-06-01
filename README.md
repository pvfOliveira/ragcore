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
ragcore ingest path/to/document.pdf       # or a URL
ragcore search "your query"
ragcore ask "What does the document say about X?"
ragcore models                            # show configured model roles
```
