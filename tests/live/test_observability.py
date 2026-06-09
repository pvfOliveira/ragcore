"""Live observability proof: a real Ollama ask traced through OTel to an
in-process Phoenix collector produces a root ``"ask"`` span with child
``gen_ai`` stage spans carrying the stable ``gen_ai.*`` attributes.

Phoenix API used (arize-phoenix 17.2.0):
  * ``px.launch_app()`` starts an in-process server + OTLP/HTTP collector,
    returning a ``Session`` whose ``.url`` is the base URL (default
    ``http://localhost:6006/``). The OTLP traces endpoint is
    ``{base}/v1/traces``.
  * Captured spans are read back via the typed client:
    ``phoenix.client.Client(base_url=...).spans.get_spans_dataframe(
    project_name="default")`` — a pandas DataFrame with a ``name`` column and
    grouped attribute columns (``attributes.gen_ai`` is a nested dict
    ``{system, request.model, operation.name, usage.*}``).
"""
from __future__ import annotations

import copy
import time
from urllib.parse import urlsplit

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def _flatten(prefix, value, out):
    """Flatten a nested attribute dict (e.g. {'request': {'model': x}}) into
    dotted keys ('request.model' -> x) so we can match STABLE_GEN_AI_ATTRS."""
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def test_phoenix_otel_trace_captures_ask(surreal_url):
    px = pytest.importorskip("phoenix")
    import asyncio

    from phoenix.client import Client

    from ragcore.ask import answer_question
    from ragcore.embedding import generate_embedding
    from ragcore.ingest import ingest_source
    from ragcore.llmops.registry import RunRegistry
    from ragcore.observability.otel import STABLE_GEN_AI_ATTRS
    from ragcore.observability.tracing import _set_provider_for_test, get_tracer
    from ragcore.store import Store

    # --- launch in-process Phoenix (server + OTLP/HTTP collector) ---
    session = px.launch_app()
    base_url = session.url.rstrip("/")
    parts = urlsplit(base_url)
    # Normalise 0.0.0.0 -> localhost for a routable client/exporter target.
    host = "localhost" if parts.hostname in ("0.0.0.0", "") else parts.hostname
    base_url = f"{parts.scheme}://{host}:{parts.port or 6006}"
    otlp_endpoint = f"{base_url}/v1/traces"

    cfg = copy.deepcopy(load_config())
    cfg.models["chat"].local_model = "qwen2.5:7b-instruct"
    cfg.surreal = SurrealConfig(
        url=surreal_url, namespace="ragcore_obs", database="ragcore_obs"
    )
    cfg.store.vector_backend = "surreal"
    cfg.observability.enabled = True
    cfg.observability.otlp_endpoint = otlp_endpoint
    _set_provider_for_test(None)  # force rebuild against the now-running Phoenix
    assert get_tracer(cfg) is not None

    # --- optional in-memory mirror on the SAME provider for a deterministic
    #     secondary check (the PRIMARY proof remains Phoenix capture below) ---
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from ragcore.observability import tracing as _tracing

    mem_exporter = InMemorySpanExporter()
    _tracing._provider.add_span_processor(SimpleSpanProcessor(mem_exporter))

    store = Store(cfg.surreal)
    asyncio.run(store.init_schema())

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as fh:
        fh.write("The Eiffel Tower is in Paris and was completed in 1889.")
        doc = fh.name

    async def _embedder(q):
        return await generate_embedding(q, cfg, chunk_size=cfg.chunking.chunk_size)

    asyncio.run(ingest_source(doc, store, cfg))
    result = asyncio.run(
        answer_question("Where is the Eiffel Tower?", store, cfg, _embedder)
    )
    assert result["answer"]

    # --- in-memory mirror: deterministic confirmation the spans were produced ---
    mem_spans = mem_exporter.get_finished_spans()
    mem_names = {s.name for s in mem_spans}
    assert "ask" in mem_names, f"no root 'ask' span produced; got {mem_names}"
    stage_spans = [
        s
        for s in mem_spans
        if s.name.startswith(("strategy", "retrieve_answer", "synthesize"))
    ]
    assert stage_spans, f"no gen_ai stage span produced; got {mem_names}"
    # The stage span must carry the stable gen_ai.* attributes.
    sample = stage_spans[0]
    for key in ("gen_ai.system", "gen_ai.operation.name", "gen_ai.request.model"):
        assert key in sample.attributes, (
            f"stage span {sample.name!r} missing {key}; "
            f"has {dict(sample.attributes)}"
        )
    assert sample.attributes["gen_ai.operation.name"] == "chat"
    assert sample.attributes["ragcore.pipeline.stage"]

    # --- PRIMARY PROOF: read the trace back from Phoenix ---
    client = Client(base_url=base_url)
    spans_df = None
    for _ in range(20):  # poll up to ~10s for the OTLP export to land + index
        try:
            df = client.spans.get_spans_dataframe(project_name="default")
        except Exception:
            df = None
        if df is not None and len(df) > 0 and "ask" in set(df["name"].tolist()):
            spans_df = df
            break
        time.sleep(0.5)

    assert spans_df is not None and len(spans_df) > 0, (
        "Phoenix captured no spans"
    )
    names = set(spans_df["name"].tolist())
    assert "ask" in names, f"Phoenix missing root 'ask' span; got {names}"
    assert any(
        n.startswith(("strategy", "retrieve_answer", "synthesize")) for n in names
    ), f"Phoenix missing gen_ai stage span; got {names}"

    # The captured stage span must carry the stable gen_ai.* attributes.
    # In the dataframe they live in the nested 'attributes.gen_ai' dict.
    stage_rows = spans_df[
        spans_df["name"].str.startswith(
            ("strategy", "retrieve_answer", "synthesize")
        )
    ]
    assert len(stage_rows) > 0
    gen_ai_obj = stage_rows.iloc[0]["attributes.gen_ai"]
    flat = {}
    _flatten("gen_ai", gen_ai_obj, flat)
    for key in ("gen_ai.system", "gen_ai.operation.name", "gen_ai.request.model"):
        assert key in flat, (
            f"Phoenix stage span missing {key}; flattened gen_ai = {flat}"
        )
    assert flat["gen_ai.operation.name"] == "chat"

    # --- record the proof run in the registry ---
    run_id = asyncio.run(
        RunRegistry(cfg.surreal).record(
            metrics={
                "observability": {"spans_captured": float(len(spans_df))}
            },
            config_snapshot={
                "backend": "phoenix",
                "gen_ai_conventions": list(STABLE_GEN_AI_ATTRS),
            },
            tag="observability-v2",
            gate_passed=True,
        )
    )
    assert run_id

    print(
        f"\n[observability] phoenix={base_url} spans_captured={len(spans_df)} "
        f"names={sorted(names)} run_id={run_id}"
    )

    _set_provider_for_test(None)
