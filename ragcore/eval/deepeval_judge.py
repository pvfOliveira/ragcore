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

    def _make_test_case(self, record: dict):
        """Build a DeepEval test case (lazy import). Overridden in deterministic
        tests so the score() path can be exercised without deepeval installed."""
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=record["question"],
            actual_output=record["answer"],
            retrieval_context=list(record["contexts"]),
            expected_output=record.get("ground_truth", ""),
        )

    def score(self, metric: str, record: dict) -> float:
        if metric not in self._metrics:
            return 0.0
        m = self._metrics[metric]
        m.measure(self._make_test_case(record))
        return float(m.score)
