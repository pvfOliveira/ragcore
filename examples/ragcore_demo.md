# ragcore architecture

ragcore is a local-first Retrieval-Augmented Generation system.
Ingestion extracts a document, splits it into token-based chunks, embeds each
chunk with a local Ollama model, and stores both the text and the vectors in
SurrealDB. Retrieval is hybrid: it runs a vector similarity search and a BM25
full-text search, then fuses the two rankings with Reciprocal Rank Fusion.
Answering uses a LangGraph workflow that plans search terms, retrieves context,
and synthesizes a cited answer. Models are pluggable via the esperanto library,
so local models are used by default and cloud models are used only when needed.
