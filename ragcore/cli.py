"""Typer CLI: init, ingest, search, ask, models."""
from __future__ import annotations

import asyncio
from typing import Optional

import typer

from ragcore.config import load_config
from ragcore.embedding import generate_embedding
from ragcore.ingest import ingest_source
from ragcore.retrieve import hybrid_search, vector_store_for
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
def ingest(
    source: str,
    is_async: bool = typer.Option(False, "--async", help="Queue as a background job instead of ingesting now"),
):
    """Ingest a file path or URL (skips if the same origin was already ingested)."""
    try:
        cfg, store = _load()
        if is_async:
            from ragcore.jobs import JobQueue
            res = asyncio.run(JobQueue(cfg.surreal).enqueue(source))
            if res.status == "queued":
                typer.echo(f"Queued {source} as job {res.job_id}")
            elif res.status == "exists":
                typer.echo(f"Already queued {source} (job {res.job_id})")
            else:  # already_ingested
                typer.echo(f"Skipped {source} (already ingested)")
            return
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
def worker(once: bool = typer.Option(False, "--once", help="Drain the queue and exit")):
    """Run the background ingestion worker (Ctrl-C to stop)."""
    try:
        from ragcore.worker import run_worker
        cfg, _ = _load()
        try:
            asyncio.run(run_worker(cfg, once=once))
        except KeyboardInterrupt:
            typer.echo("Worker stopped.")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def jobs(status: Optional[str] = typer.Option(None, "--status", help="Filter by status: queued|running|done|failed")):
    """List ingestion jobs (newest first)."""
    try:
        from ragcore.jobs import JobQueue
        cfg, _ = _load()
        rows = asyncio.run(JobQueue(cfg.surreal).list_jobs(status=status))
        if not rows:
            typer.echo("No jobs.")
            return
        for j in rows:
            extra = j["source_id"] or j["error"] or ""
            typer.echo(f"{j['id']}  {j['status']:<8}  attempts={j['attempts']}  {j['origin']}  {extra}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def retry(job_id: str):
    """Requeue a failed ingestion job."""
    try:
        from ragcore.jobs import JobQueue
        cfg, _ = _load()
        ok = asyncio.run(JobQueue(cfg.surreal).requeue(job_id))
        if ok:
            typer.echo(f"Requeued {job_id}")
        else:
            typer.echo(f"No failed job {job_id}")
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
        results = asyncio.run(
            hybrid_search(store, _embedder(cfg), query, k=k, vector_store=vector_store_for(cfg))
        )
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
def chat(session: Optional[str] = typer.Option(None, "--session", help="Continue an existing session id")):
    """Interactive chat over your corpus (history-aware). /exit or Ctrl-D to quit."""
    try:
        from ragcore.chat import chat_turn
        from ragcore.sessions import SessionStore
        cfg, store = _load()
        sess = SessionStore(cfg.surreal)
        session_id = session or asyncio.run(sess.create_session())
        typer.echo(f"Session {session_id}  (/exit to quit)")
        embed = _embedder(cfg)
        while True:
            try:
                message = input("you> ")
            except EOFError:
                break
            if message.strip() in ("/exit", "/quit"):
                break
            if not message.strip():
                continue
            try:
                result = asyncio.run(chat_turn(sess, store, cfg, session_id, message, embed))
                typer.echo(result["answer"])
                if result["citations"]:
                    typer.echo("Sources: " + ", ".join(result["citations"]))
            except Exception as e:  # one bad turn must not end the session
                from ragcore.errors import classify_error
                _, m = classify_error(e)
                typer.echo(f"[error] {m}", err=True)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def sessions():
    """List chat sessions."""
    try:
        from ragcore.sessions import SessionStore
        cfg, _ = _load()
        rows = asyncio.run(SessionStore(cfg.surreal).list_sessions())
        if not rows:
            typer.echo("No chat sessions.")
            return
        for s in rows:
            typer.echo(f"{s['id']}  [{s['messages']} msgs]  {s.get('title') or '(untitled)'}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command()
def serve(host: str = typer.Option("127.0.0.1", "--host"),
          port: int = typer.Option(8080, "--port")):
    """Serve the local web UI (localhost only, no auth)."""
    try:
        import uvicorn

        from ragcore.web.app import create_app
        cfg = load_config(_state["config_path"])
        typer.echo(f"ragcore web UI on http://{host}:{port}  (local only, no auth)")
        uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


def _config_snapshot(cfg) -> dict:
    return {"models": {k: v.local_model for k, v in cfg.models.items()},
            "routing": cfg.routing.model_dump(),
            "rerank": cfg.rerank.enabled, "cache": cfg.cache.enabled,
            "vector_backend": cfg.store.vector_backend}


@app.command("eval")
def eval_cmd(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Path to a JSONL eval dataset"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Label this run for later reference"),
):
    """Evaluate retrieval+answer quality (Ragas + TruLens) with a local Ollama judge."""
    try:
        from ragcore.eval.harness import run_eval
        from ragcore.llmops.registry import RunRegistry
        cfg = load_config(_state["config_path"])
        report = run_eval(cfg, dataset)
        run_id = asyncio.run(RunRegistry(cfg.surreal).record(
            metrics=report,
            config_snapshot=_config_snapshot(cfg),
            dataset=dataset,
            tag=tag,
        ))
        typer.echo(f"Run recorded: {run_id}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command(name="runs")
def runs_cmd():
    """List recorded eval runs (id, tag, created, key metrics)."""
    try:
        cfg, _ = _load()
        from ragcore.llmops.registry import RunRegistry
        rows = asyncio.run(RunRegistry(cfg.surreal).list_runs())
        for r in rows:
            families = ", ".join(sorted(r["metrics"].keys())) if r["metrics"] else "-"
            typer.echo(f"{r['id']}  tag={r['tag']}  {r['created']}  metrics=[{families}]")
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
