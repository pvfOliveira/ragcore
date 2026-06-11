"""LIVE: Wave-6 eval evidence on this host.

1. The real ``run_eval`` machinery over golden v1 (10 provenance-noted items
   citing tests/live/fixtures/grid_storage.md) with the real Ollama
   Ragas + TruLens judge, then a self-comparison ``check_gate`` — RunRegistry
   tag ``golden-v4``. This is the slowest live test in the repo (judge LLM x
   10 items x 6 metrics).
2. A real sampled chat turn (``chat_turn`` over Ollama + SurrealDB sessions)
   scored through ``online.maybe_score_turn`` — the "evals in prod" path,
   where prod is honestly ragcore's live local chat — RunRegistry tag
   ``online-eval-v4``. Span export is covered deterministically in Task 13;
   observability stays default-off here so the test proves the score, not
   Phoenix connectivity.
"""
from __future__ import annotations

import asyncio
import copy
import random
from pathlib import Path

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live

_FIXTURE = Path(__file__).parent / "fixtures" / "grid_storage.md"


def _cfg(surreal_url: str, ns: str):
    cfg = copy.deepcopy(load_config())
    # Non-reasoning model for answerer + judge (house live-eval override:
    # avoids qwen3 CoT overhead across 60 judge calls).
    cfg.models["chat"].local_model = "qwen2.5:7b-instruct"
    cfg.eval.judge_model = "ollama:qwen2.5:7b-instruct"
    cfg.eval.framework = "ragas"
    cfg.surreal = SurrealConfig(url=surreal_url, namespace=ns, database=ns)
    cfg.store.vector_backend = "surreal"
    return cfg


def _ingest_corpus(cfg):
    """Schema-init the ephemeral store and ingest the golden corpus (idempotent)."""
    from ragcore.ingest import ingest_source
    from ragcore.store import Store

    store = Store(cfg.surreal)
    asyncio.run(store.init_schema())
    result = asyncio.run(ingest_source(str(_FIXTURE), store, cfg))
    assert result.source_id, "golden corpus ingest returned no source_id"
    return store


# ---------------------------------------------------------------------------
# 1. golden v1 -> run_eval (real judge) -> gate -> registry (golden-v4)
# ---------------------------------------------------------------------------

def test_golden_v1_run_eval_and_gate(surreal_url):
    import ragcore.config as config_mod
    from ragcore.eval.golden import GOLDEN_DIR, load_golden
    from ragcore.eval.harness import RAGAS_METRICS, TRULENS_METRICS, run_eval
    from ragcore.llmops.gates import check_gate
    from ragcore.llmops.registry import RunRegistry

    cfg = _cfg(surreal_url, "ragcore_golden_v4")
    _ingest_corpus(cfg)

    items = load_golden("v1")
    assert len(items) >= 10, f"golden v1 should have >=10 items, got {len(items)}"
    dataset_path = GOLDEN_DIR / "v1.jsonl"

    # run_eval answers each question against cfg.surreal (our ephemeral store),
    # but its judge is built inside compute_metrics(judge=None), which calls
    # load_config() AGAIN — i.e. the default config.toml, not this test's cfg.
    # Rebind load_config to return our cfg for the duration so the judge uses
    # the same overrides as every other live eval test, then restore.
    original_load_config = config_mod.load_config
    config_mod.load_config = lambda *a, **k: cfg
    try:
        report = run_eval(cfg, dataset_path)
    finally:
        config_mod.load_config = original_load_config

    # Standard report shape: both families, every metric a real score in [0,1].
    assert set(report) == {"ragas", "trulens"}, f"unexpected families: {set(report)}"
    assert set(report["ragas"]) == set(RAGAS_METRICS)
    assert set(report["trulens"]) == set(TRULENS_METRICS)
    for family, scores in report.items():
        for metric, value in scores.items():
            assert isinstance(value, float), f"{family}.{metric} not float: {value!r}"
            assert 0.0 <= value <= 1.0, f"{family}.{metric} out of [0,1]: {value}"

    # Self-comparison must pass the gate with zero regressions.
    gate_ok, regressions = check_gate(report, report, cfg.llmops.tolerance)
    assert gate_ok, f"self-comparison gate failed: {regressions}"
    assert regressions == []

    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics=report,
        config_snapshot={"dataset": "golden/v1.jsonl", "items": len(items),
                         "judge": cfg.eval.judge_model},
        dataset=str(dataset_path), tag="golden-v4", gate_passed=gate_ok))
    assert run_id and ":" in run_id

    print(f"\n[golden-v4] run_id={run_id} report={report}")


# ---------------------------------------------------------------------------
# 2. real sampled chat turn -> online_eval groundedness (online-eval-v4)
# ---------------------------------------------------------------------------

def test_online_eval_scores_real_chat_turn(surreal_url):
    from ragcore.chat import chat_turn
    from ragcore.embedding import generate_embedding
    from ragcore.eval import online
    from ragcore.llmops.registry import RunRegistry
    from ragcore.sessions import SessionStore

    cfg = _cfg(surreal_url, "ragcore_online_v4")
    store = _ingest_corpus(cfg)

    cfg.online_eval.enabled = True
    cfg.online_eval.sample_rate = 1.0  # this turn MUST be sampled

    sessions = SessionStore(cfg.surreal)
    session_id = asyncio.run(sessions.create_session("online-eval-v4"))

    async def embedder(q: str):
        return await generate_embedding(q, cfg, chunk_size=cfg.chunking.chunk_size)

    # chat_turn resolves online.maybe_score_turn at call time, so a capturing
    # wrapper on the module observes the real scores; delegate to the real
    # function with a seeded rng, restore in finally.
    captured: list = []
    real_maybe_score_turn = online.maybe_score_turn

    async def _capturing(question, answer, chunks, config, rng=None):
        scores = await real_maybe_score_turn(
            question, answer, chunks, config, rng=random.Random(42))
        captured.append(scores)
        return scores

    online.maybe_score_turn = _capturing
    try:
        result = asyncio.run(chat_turn(
            sessions, store, cfg, session_id,
            "Why is grid storage needed?", embedder))
    finally:
        online.maybe_score_turn = real_maybe_score_turn

    assert result.get("answer"), "chat_turn returned no answer"
    assert len(captured) == 1, "online eval was not invoked exactly once"
    scores = captured[0]
    assert scores is not None, "turn not scored despite enabled + sample_rate=1.0"
    assert "groundedness" in scores, f"no groundedness in {scores}"
    assert 0.0 <= scores["groundedness"] <= 1.0, f"groundedness out of [0,1]: {scores}"

    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"groundedness": scores["groundedness"],
                 "sample_rate": cfg.online_eval.sample_rate},
        config_snapshot={"path": "live local chat (the honest 'prod')",
                         "judge": cfg.eval.judge_model},
        tag="online-eval-v4", gate_passed=None))
    assert run_id and ":" in run_id

    print(f"\n[online-eval-v4] run_id={run_id} scores={scores}")
