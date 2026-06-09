"""A DeepEval-backed judge mirroring the harness ``judge.score(metric, record)``
protocol. Judge LLM is the local Ollama model (via litellm), so this runs fully
local. Heavy imports happen in __init__ to stay out of the offline path. DeepEval
populates only the ragas metric family; trulens metric names return 0.0."""
from __future__ import annotations

from typing import Any

from ragcore.eval.harness import _litellm_model


class DeepEvalJudge:
    def __init__(self, config: Any):
        from deepeval.metrics import (
            AnswerRelevancyMetric, ContextualPrecisionMetric, FaithfulnessMetric,
        )
        from deepeval.models import LiteLLMModel  # NOTE: verify exact class name when deepeval is installed (Task 8)

        model_id = _litellm_model(config.eval.judge_model)   # "ollama/<model>"
        judge_model = LiteLLMModel(model=model_id)
        self._metrics = {
            "faithfulness": FaithfulnessMetric(model=judge_model),
            "answer_relevancy": AnswerRelevancyMetric(model=judge_model),
            "context_precision": ContextualPrecisionMetric(model=judge_model),
        }

    def score(self, metric: str, record: dict) -> float:
        if metric not in self._metrics:
            return 0.0

        try:
            from deepeval.test_case import LLMTestCase
            test_case = LLMTestCase(
                input=record["question"],
                actual_output=record["answer"],
                retrieval_context=list(record["contexts"]),
                expected_output=record.get("ground_truth", ""),
            )
        except ModuleNotFoundError:
            # deepeval not installed — construct a simple namespace that
            # satisfies injected fake metrics in deterministic tests.
            import types
            test_case = types.SimpleNamespace(
                input=record["question"],
                actual_output=record["answer"],
                retrieval_context=list(record["contexts"]),
                expected_output=record.get("ground_truth", ""),
            )

        m = self._metrics[metric]
        m.measure(test_case)
        return float(m.score)
