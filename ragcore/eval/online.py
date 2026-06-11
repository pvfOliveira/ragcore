"""Sampled online evaluation of live chat traffic ("evals in prod").

HONEST LABEL: "prod" here is ragcore's live local chat path (CLI/web) — the
real serving path of this local-first system, and the evidence says so.

A sampled turn is scored with the SAME judge plumbing as offline eval
(``_build_judge_for_framework`` → TruLens/Ragas groundedness over Ollama) and
the scores land on an ``online_eval`` OTel span exported to Phoenix. Scoring
runs in a worker thread (the judge is sync + slow) and NEVER raises into the
user's turn."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

logger = logging.getLogger(__name__)

_rng = random.Random()


def should_sample(rate: float, rng: random.Random) -> bool:
    return rng.random() < rate


_judge_cache: dict[str, Any] = {}


def _build_judge(config: Any):
    """Judge instance per (framework, judge_model) — heavy Ragas/TruLens init."""
    key = f"{config.eval.framework}:{config.eval.judge_model}"
    if key not in _judge_cache:
        from ragcore.eval.harness import _build_judge_for_framework
        _judge_cache[key] = _build_judge_for_framework(config)
    return _judge_cache[key]


async def maybe_score_turn(question: str, answer: str, chunks: list[dict],
                           config: Any, rng: Optional[random.Random] = None
                           ) -> Optional[dict]:
    """Score this turn if sampled. Returns scores or None. Never raises."""
    oe = config.online_eval
    if not oe.enabled or not should_sample(oe.sample_rate, rng or _rng):
        return None
    try:
        record = {"question": question, "answer": answer,
                  "contexts": [c.get("content", "") for c in chunks],
                  "ground_truth": ""}

        def _score() -> dict:
            judge = _build_judge(config)
            return {m: float(judge.score(m, record)) for m in oe.metrics}

        scores = await asyncio.to_thread(_score)
    except Exception as e:
        logger.warning("online eval failed (turn unaffected): %s", e)
        return None
    _emit_span(scores, config)
    return scores


def _emit_span(scores: dict, config: Any) -> None:
    try:
        from ragcore.observability.spans import traced_span
        from ragcore.observability.tracing import get_tracer
        tracer = get_tracer(config)
        with traced_span(tracer, "online_eval") as span:
            if span is not None:
                span.set_attribute("ragcore.eval.sampled", True)
                for metric, score in scores.items():
                    span.set_attribute(f"ragcore.eval.{metric}", score)
    except Exception as e:  # tracing must never break the turn either
        logger.warning("online eval span emit failed: %s", e)
