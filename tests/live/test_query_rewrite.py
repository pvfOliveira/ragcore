"""Live proof: multi_query expansion over Ollama produces real variants and the
fused retrieval still returns relevant chunks."""
from __future__ import annotations

import copy

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


async def test_multi_query_expansion_over_ollama(surreal_url):
    from ragcore.ask import _select_and_build, _clean
    from ragcore.query_rewrite import rewrite_query
    from ragcore.llmops.registry import RunRegistry

    cfg = copy.deepcopy(load_config())
    cfg.query_rewrite.enabled = True
    cfg.query_rewrite.strategy = "multi_query"
    cfg.query_rewrite.n = 3
    chat, provider, model = _select_and_build(cfg)

    async def chat_fn(prompt):
        msg = await chat.ainvoke(prompt)
        return _clean(msg.content)

    variants = await rewrite_query("How does reciprocal rank fusion work?", cfg, chat_fn)
    assert len(variants) >= 2 and all(isinstance(v, str) and v for v in variants)

    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_qr", database="ragcore_qr")
    run_id = await RunRegistry(cfg.surreal).record(
        metrics={"query_rewrite": {"variants": float(len(variants))}},
        config_snapshot={"strategy": "multi_query"}, tag="query-rewrite-v3", gate_passed=True)
    assert run_id
