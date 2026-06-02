"""Typer CLI: init, ingest, search, ask, models."""
from __future__ import annotations

import asyncio
from typing import Optional

import typer

from ragcore.config import load_config
from ragcore.embedding import generate_embedding
from ragcore.ingest import ingest_source
from ragcore.retrieve import hybrid_search
from ragcore.store import Store

app = typer.Typer(help="ragcore — local-first RAG")
_state = {"config_path": "config.toml"}


@app.callback()
def _main(config: str = typer.Option("config.toml", "--config")):
    from dotenv import load_dotenv
    load_dotenv()
    _state["config_path"] = config


def _load():
    cfg = load_config(_state["config_path"])
    return cfg, Store(cfg.surreal)


def _fail(exc: Exception):
    from ragcore.errors import classify_error
    _, message = classify_error(exc)
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _embedder(cfg):
    async def fn(query: str):
        return await generate_embedding(query, cfg, chunk_size=cfg.chunking.chunk_size)
    return fn


@app.command()
def init():
    """Create the SurrealDB schema."""
    try:
        cfg, store = _load()
        asyncio.run(store.init_schema())
        typer.echo("Schema initialized.")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def ingest(source: str):
    """Ingest a file path or URL (skips if the same origin was already ingested)."""
    try:
        cfg, store = _load()
        result = asyncio.run(ingest_source(source, store, cfg, chunk_size=cfg.chunking.chunk_size))
        if result.created:
            typer.echo(f"Ingested {source} as {result.source_id}")
        else:
            typer.echo(f"Skipped {source} (already ingested as {result.source_id})")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command("list")
def list_sources():
    """List ingested sources."""
    try:
        cfg, store = _load()
        sources = asyncio.run(store.list_sources())
        if not sources:
            typer.echo("No sources ingested.")
            return
        for s in sources:
            typer.echo(
                f"{s['id']}  [{s['chunks']} chunks]  "
                f"{s.get('title') or '(untitled)'}  <{s.get('origin') or ''}>"
            )
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def remove(source_id: str):
    """Remove an ingested source by id (its embeddings are deleted too)."""
    try:
        cfg, store = _load()
        deleted = asyncio.run(store.delete_source(source_id))
        if deleted:
            typer.echo(f"Removed {source_id}")
        else:
            typer.echo(f"No such source: {source_id}")
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def search(query: str, k: int = 5):
    """Hybrid search; print matching chunks."""
    try:
        cfg, store = _load()
        results = asyncio.run(hybrid_search(store, _embedder(cfg), query, k=k))
        for r in results:
            typer.echo(f"[{r['source']}] {r['content'][:120]}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def ask(question: str, cloud: bool = typer.Option(False, "--cloud", help="Force cloud escalation")):
    """Ask a question over the ingested corpus."""
    try:
        from ragcore.ask import answer_question
        cfg, store = _load()
        result = asyncio.run(answer_question(question, store, cfg, _embedder(cfg), force_cloud=cloud))
        typer.echo(result["answer"])
        if result["citations"]:
            typer.echo("\nSources: " + ", ".join(result["citations"]))
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def models():
    """Show configured model roles."""
    cfg = load_config(_state["config_path"])
    for role, spec in cfg.models.items():
        line = f"{role}: local={spec.local_provider}/{spec.local_model}"
        if spec.cloud_model:
            line += f"  cloud={spec.cloud_provider}/{spec.cloud_model}"
        typer.echo(line)


if __name__ == "__main__":
    app()
