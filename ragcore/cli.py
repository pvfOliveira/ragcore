"""Typer CLI: init, ingest, search, ask, models."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _is_image_path(source: str) -> bool:
    return Path(source).suffix.lower() in _IMAGE_EXTENSIONS


@app.command()
def ingest(
    source: str,
    is_async: bool = typer.Option(False, "--async", help="Queue as a background job instead of ingesting now"),
):
    """Ingest a file path or URL (skips if the same origin was already ingested)."""
    try:
        cfg, store = _load()
        if _is_image_path(source):
            from ragcore.multimodal import ingest_image
            image_id, created = asyncio.run(ingest_image(source, cfg))
            if created:
                typer.echo(f"Ingested image {source} as {image_id}")
            else:
                typer.echo(f"Skipped {source} (already ingested as {image_id})")
            return
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


async def _image_search(query: str, k: int, store, embedder) -> list[dict]:
    """CLIP text-encode the query and retrieve the nearest images."""
    qv = embedder.embed_text(query)
    return await store.search(qv, k)


@app.command()
def search(
    query: str,
    k: int = 5,
    images: bool = typer.Option(False, "--images", help="Cross-modal image search (CLIP text->image)"),
):
    """Hybrid search; print matching chunks."""
    try:
        cfg, store = _load()
        if images:
            from ragcore.multimodal import ClipEmbedder, ImageStore
            embedder = ClipEmbedder(model_name=cfg.multimodal.model,
                                    pretrained=cfg.multimodal.pretrained,
                                    device=cfg.multimodal.device)
            img_store = ImageStore(cfg.surreal)
            hits = asyncio.run(_image_search(query, k, img_store, embedder))
            if not hits:
                typer.echo("No images found.")
            for h in hits:
                typer.echo(f"{h['path']}  ({h['similarity']:.3f})")
            return
        results = asyncio.run(
            hybrid_search(store, _embedder(cfg), query, k=k, vector_store=vector_store_for(cfg), config=cfg)
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


async def _record_eval_run(cfg, report, dataset, tag) -> str:
    snapshot = _config_snapshot(cfg)
    snapshot["centroid"] = await _dataset_centroid(cfg, dataset)
    from ragcore.llmops.registry import RunRegistry
    return await RunRegistry(cfg.surreal).record(
        metrics=report, config_snapshot=snapshot, dataset=dataset, tag=tag)


def _config_snapshot(cfg) -> dict:
    return {"models": {k: v.local_model for k, v in cfg.models.items()},
            "routing": cfg.routing.model_dump(),
            "rerank": cfg.rerank.enabled, "cache": cfg.cache.enabled,
            "vector_backend": cfg.store.vector_backend}


async def _dataset_centroid(cfg, dataset_path):
    from ragcore.eval.harness import _DEFAULT_DATASET, _load_dataset

    items = _load_dataset(Path(dataset_path) if dataset_path else _DEFAULT_DATASET)
    embed = _embedder(cfg)
    vecs = [await embed(it["question"]) for it in items]
    if not vecs:
        return []
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


@app.command("eval")
def eval_cmd(
    dataset: Optional[str] = typer.Option(None, "--dataset", help="Path to a JSONL eval dataset"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Label this run for later reference"),
):
    """Evaluate retrieval+answer quality (Ragas + TruLens) with a local Ollama judge."""
    try:
        from ragcore.eval.harness import run_eval
        cfg = load_config(_state["config_path"])
        report = run_eval(cfg, dataset)
        run_id = asyncio.run(_record_eval_run(cfg, report, dataset, tag))
        typer.echo(f"Run recorded: {run_id}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command(name="gate")
def gate_cmd(
    baseline: str = typer.Option(..., "--baseline", help="Baseline run id or tag"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
):
    """Run eval; exit 1 if any metric regresses beyond [llmops] tolerance vs the baseline."""
    try:
        cfg, _ = _load()
        from ragcore.eval.harness import run_eval
        from ragcore.llmops.gates import check_gate
        from ragcore.llmops.registry import RunRegistry
        base = asyncio.run(RunRegistry(cfg.surreal).resolve(baseline))
        if base is None:
            typer.echo(f"No baseline run for '{baseline}'", err=True)
            raise typer.Exit(2)
        report = run_eval(cfg, dataset)
        ok, regressions = check_gate(report, base["metrics"], cfg.llmops.tolerance)
        for r in regressions:
            typer.echo(f"REGRESSION {r['metric']}: {r['baseline']:.3f} -> {r['current']:.3f}", err=True)
        typer.echo("GATE PASS" if ok else "GATE FAIL")
        raise typer.Exit(0 if ok else 1)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command(name="drift")
def drift_cmd(
    baseline: str = typer.Option(..., "--baseline", help="Baseline run id or tag"),
    dataset: Optional[str] = typer.Option(None, "--dataset"),
):
    """Check embedding-space drift vs a baseline run's stored centroid; exit 1 if drifted."""
    try:
        cfg, _ = _load()
        from ragcore.llmops.gates import check_drift
        from ragcore.llmops.registry import RunRegistry
        base = asyncio.run(RunRegistry(cfg.surreal).resolve(baseline))
        if base is None:
            typer.echo(f"No baseline run for '{baseline}'", err=True)
            raise typer.Exit(2)
        base_centroid = base["config_snapshot"].get("centroid")
        if not base_centroid:
            typer.echo(
                f"Baseline run '{baseline}' has no stored centroid; "
                "re-run eval to record one.", err=True
            )
            raise typer.Exit(2)
        current_centroid = asyncio.run(_dataset_centroid(cfg, dataset))
        drifted, distance = check_drift(base_centroid, current_centroid, cfg.llmops.drift_threshold)
        typer.echo(f"Centroid distance: {distance:.4f}  (threshold={cfg.llmops.drift_threshold})")
        typer.echo("DRIFT DETECTED" if drifted else "DRIFT OK")
        raise typer.Exit(1 if drifted else 0)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command(name="promote")
def promote_cmd(
    run_id: str,
    no_gate: bool = typer.Option(False, "--no-gate", help="Promote even if the run's gate did not pass"),
):
    """Point production at an eval run (refuses a run that failed its gate unless --no-gate)."""
    try:
        cfg, _ = _load()
        from ragcore.llmops.deploy import DeploymentStore
        asyncio.run(DeploymentStore(cfg.surreal).promote(run_id, require_gate=not no_gate))
        typer.echo(f"Promoted {run_id}")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


@app.command(name="rollback")
def rollback_cmd():
    """Revert production to the previously promoted run."""
    try:
        cfg, _ = _load()
        from ragcore.llmops.deploy import DeploymentStore
        prev = asyncio.run(DeploymentStore(cfg.surreal).rollback())
        typer.echo(f"Rolled back to {prev}" if prev else "No previous deployment to roll back to.")
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
def bench():
    """Sweep Ollama serving params and report latency/throughput (MPS benchmark)."""
    try:
        import json

        from ragcore.bench.harness import OllamaBenchClient, run_bench

        cfg, _ = _load()
        client = OllamaBenchClient(model=cfg.models["chat"].local_model)
        report = run_bench(client, **cfg.bench.model_dump())

        # Print table header
        typer.echo(
            f"{'num_ctx':>8}  {'num_batch':>9}  {'conc':>4}  "
            f"{'ttft_s':>7}  {'latency_s':>9}  {'tok/s':>8}"
        )
        typer.echo("-" * 58)
        for run in report["runs"]:
            typer.echo(
                f"{run['num_ctx']:>8}  {run['num_batch']:>9}  {run['concurrency']:>4}  "
                f"{run['ttft_s']:>7.3f}  {run['total_latency_s']:>9.3f}  {run['tok_per_s']:>8.1f}"
            )

        bt = report["best_throughput"]
        bl = report["best_latency"]
        typer.echo("")
        typer.echo(
            f"Best throughput: ctx={bt['num_ctx']} batch={bt['num_batch']} "
            f"conc={bt['concurrency']} → {bt['tok_per_s']:.1f} tok/s"
        )
        typer.echo(
            f"Best latency:    ctx={bl['num_ctx']} batch={bl['num_batch']} "
            f"conc={bl['concurrency']} → {bl['total_latency_s']:.3f}s total / {bl['ttft_s']:.3f}s ttft"
        )

        # Write report JSON
        report_path = Path("data/bench/report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"\nReport written to {report_path}")
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


graph_app = typer.Typer(help="Knowledge-graph commands.")
app.add_typer(graph_app, name="graph")


@graph_app.command(name="build")
def graph_build():
    """Back-fill the knowledge graph over already-ingested sources (skips already-graphed sources)."""
    try:
        from ragcore.graph import GraphStore, _chat_fn, extract_triples

        cfg, store = _load()
        sources = asyncio.run(store.list_sources())
        if not sources:
            typer.echo("No sources ingested — nothing to graph.")
            return
        gs = GraphStore(cfg.surreal)
        chat = _chat_fn(cfg)
        for s in sources:
            sid = s["id"]
            already = asyncio.run(gs.has_source(sid))
            if already:
                typer.echo(f"skipped {sid} (already graphed)")
                continue
            chunks = asyncio.run(store.get_chunks(sid))
            if not chunks:
                typer.echo(f"skipped {sid} (no chunks)")
                continue
            all_triples = []
            for chunk in chunks:
                all_triples.extend(asyncio.run(extract_triples(chunk, chat)))
            if all_triples:
                asyncio.run(gs.upsert_triples(sid, all_triples))
                typer.echo(f"graphed {sid} ({len(all_triples)} triples)")
            else:
                typer.echo(f"graphed {sid} (no triples)")
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


cost_app = typer.Typer(help="Cost ledger and spend reporting.")
app.add_typer(cost_app, name="cost")


@cost_app.command(name="report")
def cost_report_cmd():
    """Print spend by model, cache savings, and right-sizing hints."""
    try:
        from ragcore.cost.ledger import CostLedger
        from ragcore.cost.report import build_report

        cfg, _ = _load()
        agg = CostLedger(cfg.cost.ledger_path).aggregate()
        bench_path = Path("data/bench/report.json")
        bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else None
        rep = build_report(agg, cfg.cost.rates, bench)

        typer.echo("=== Cost Report ===")
        typer.echo(f"Total spend:  ${rep['total_spend_usd']:.4f}")
        typer.echo("")

        typer.echo("Spend by model:")
        if rep["spend_usd"]:
            for key, usd in rep["spend_usd"].items():
                typer.echo(f"  {key:<40}  ${usd:.4f}")
        else:
            typer.echo("  (no usage recorded)")
        typer.echo("")

        cache = rep["cache"]
        typer.echo(
            f"Cache:  hit_rate={cache['hit_rate'] * 100:.1f}%"
            f"  tokens_avoided_estimate={cache['tokens_avoided_estimate']:.0f}"
        )
        typer.echo("")

        if rep["rightsizing"]:
            typer.echo("Right-sizing hints:")
            for hint in rep["rightsizing"]:
                typer.echo(f"  • {hint}")
        else:
            typer.echo("Right-sizing hints: none")

        if rep.get("bench_summary"):
            bs = rep["bench_summary"]
            typer.echo("")
            best_tok = bs["best_tok_per_s"]
            bench_line = f"Bench:  best_tok_per_s={best_tok:.1f} tok/s" if best_tok is not None else "Bench:  best_tok_per_s=n/a"
            if bs["best_ttft_s"] is not None:
                bench_line += f"  best_ttft={bs['best_ttft_s']:.3f} s"
            if bs["best_total_latency_s"] is not None:
                bench_line += f"  best_total_latency={bs['best_total_latency_s']:.3f} s"
            typer.echo(bench_line)
    except typer.Exit:
        raise
    except Exception as e:
        _fail(e)


if __name__ == "__main__":
    app()
