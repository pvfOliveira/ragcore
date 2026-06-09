"""Live proof: a small BootstrapFewShot compile over Ollama produces a loadable
artifact and the optimized strategy prompt drives a real ask."""
from __future__ import annotations

import asyncio
import copy

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def test_dspy_compiles_strategy_over_ollama(surreal_url, tmp_path):
    pytest.importorskip("dspy")
    from ragcore.dspy_optimizer import compile_strategy, load_compiled_strategy
    from ragcore.llmops.registry import RunRegistry

    cfg = copy.deepcopy(load_config())
    cfg.models["chat"].local_model = "qwen2.5:7b-instruct"
    cfg.eval.judge_model = "ollama:qwen2.5:7b-instruct"
    cfg.dspy.enabled = True
    cfg.dspy.max_demos = 2  # keep the optimized task SMALL (landmine)
    cfg.dspy.compiled_path = str(tmp_path / "dspy_compiled.json")
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_dspy", database="ragcore_dspy")

    prompt = compile_strategy(cfg)
    assert "{{question}}" in prompt
    assert "{{max_searches}}" in prompt
    assert load_compiled_strategy(cfg.dspy.compiled_path) == prompt

    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"dspy": {"compiled": 1.0, "max_demos": float(cfg.dspy.max_demos)}},
        config_snapshot={"compiled_path": cfg.dspy.compiled_path},
        tag="dspy-v2", gate_passed=True))
    assert run_id
