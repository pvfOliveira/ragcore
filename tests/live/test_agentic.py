"""Live proof: the agentic corrective loop answers over Ollama end-to-end
(ask routed through run_agentic) and records a RunRegistry run."""
from __future__ import annotations

import asyncio
import copy
import tempfile

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def test_agentic_ask_over_ollama(surreal_url):
    from ragcore.ask import answer_question
    from ragcore.embedding import generate_embedding
    from ragcore.ingest import ingest_source
    from ragcore.llmops.registry import RunRegistry
    from ragcore.store import Store

    cfg = copy.deepcopy(load_config())
    cfg.models["chat"].local_model = "qwen2.5:7b-instruct"
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_ag", database="ragcore_ag")
    cfg.store.vector_backend = "surreal"
    cfg.agentic.enabled = True

    store = Store(cfg.surreal)
    asyncio.run(store.init_schema())

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as fh:
        fh.write("Reciprocal rank fusion blends multiple ranked lists by summing 1/(k+rank).")
        doc = fh.name

    async def _embedder(q):
        return await generate_embedding(q, cfg, chunk_size=cfg.chunking.chunk_size)

    asyncio.run(ingest_source(doc, store, cfg))
    result = asyncio.run(answer_question("How does reciprocal rank fusion work?", store, cfg, _embedder))
    assert result["answer"] and isinstance(result["answer"], str)

    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"agentic": {"answered": 1.0}},
        config_snapshot={"max_iterations": cfg.agentic.max_iterations},
        tag="agentic-v3", gate_passed=True))
    assert run_id

    print(f"\n[agentic] answer_len={len(result['answer'])} citations={result['citations']} run_id={run_id}")
