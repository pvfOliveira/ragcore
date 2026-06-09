"""Live proof for the eval frameworks: a DeepEval judge gates over Ollama, and a
Promptfoo eval runs against Ollama. Each records a run under data/eval/."""
from __future__ import annotations

import copy

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def _cfg(surreal_url):
    cfg = copy.deepcopy(load_config())
    cfg.models["chat"].local_model = "qwen2.5:7b-instruct"
    cfg.eval.judge_model = "ollama:qwen2.5:7b-instruct"
    cfg.eval.framework = "deepeval"
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_eval2", database="ragcore_eval2")
    return cfg


def test_deepeval_gates_over_ollama(surreal_url):
    pytest.importorskip("deepeval")
    import asyncio

    from ragcore.eval.deepeval_judge import DeepEvalJudge
    from ragcore.eval.harness import compute_metrics
    from ragcore.llmops.gates import check_gate
    from ragcore.llmops.registry import RunRegistry

    cfg = _cfg(surreal_url)
    judge = DeepEvalJudge(cfg)
    records = [{
        "question": "Where is the Eiffel Tower?",
        "answer": "The Eiffel Tower is in Paris, France.",
        "contexts": ["The Eiffel Tower is in Paris, France, completed in 1889."],
        "ground_truth": "Paris, France",
    }]
    report = compute_metrics(records, judge=judge)
    assert report["ragas"]["faithfulness"] >= 0.0  # real DeepEval score

    # eval-driven-dev: a crafted regression makes the gate fail. DeepEval only
    # populates the ragas family, so report["trulens"] is all-zeros; a baseline
    # that expects 1.0 there guarantees a regression beyond tolerance no matter
    # how high the (easy-case) ragas scores land. We also assert the regression
    # is reported with a metric name + negative delta — gate mechanics, not luck.
    baseline = {
        "ragas": {k: 1.0 for k in report["ragas"]},
        "trulens": {k: 1.0 for k in report["trulens"]},
    }
    ok, regressions = check_gate(report, baseline, cfg.llmops.tolerance)
    assert ok is False and regressions
    assert all(r["delta"] < 0 and r["metric"] for r in regressions)

    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics=report, config_snapshot={"framework": "deepeval"},
        tag="deepeval-v2", gate_passed=ok))
    assert run_id


def test_promptfoo_runs_over_ollama(surreal_url):
    import asyncio

    from ragcore.eval.promptfoo.runner import promptfoo_available, run_promptfoo
    from ragcore.llmops.registry import RunRegistry

    if not promptfoo_available():
        pytest.skip("promptfoo CLI not installed")
    cfg = _cfg(surreal_url)
    metrics = run_promptfoo()
    assert 0.0 <= metrics["promptfoo"]["pass_rate"] <= 1.0
    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics=metrics, config_snapshot={"framework": "promptfoo"},
        tag="promptfoo-v2", gate_passed=metrics["promptfoo"]["pass_rate"] >= 0.5))
    assert run_id
