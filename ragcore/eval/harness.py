"""Offline-testable evaluation harness for ragcore: Ragas + TruLens metrics.

The report shape is fixed::

    {
      "ragas":   {"faithfulness", "answer_relevancy", "context_precision"},
      "trulens": {"groundedness", "context_relevance", "answer_relevance"},
    }

Aggregation is identical whether the per-record scores come from an injected
stub judge (deterministic, offline tests) or from the real Ollama-backed judge
(``judge is None``). The judge interface is a single method::

    judge.score(metric: str, record: dict) -> float

where ``record`` is ``{question, answer, contexts, ground_truth}``. This keeps
the heavy ragas/trulens engines lazily imported behind the real judge and lets
``compute_metrics`` exercise the report assembly without a live LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")
TRULENS_METRICS = ("groundedness", "context_relevance", "answer_relevance")

_DEFAULT_DATASET = Path(__file__).parent / "dataset.jsonl"
_REPORT_PATH = Path(__file__).parent.parent.parent / "data" / "eval" / "report.json"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate(metrics: tuple[str, ...], records: list[dict], judge: Any) -> dict[str, float]:
    """Mean of the judge's per-record score for each metric."""
    out: dict[str, float] = {}
    for metric in metrics:
        out[metric] = _mean([float(judge.score(metric, r)) for r in records])
    return out


def compute_metrics(records: list[dict], *, judge: Any = None) -> dict[str, dict[str, float]]:
    """Compute the Ragas + TruLens report over ``records``.

    Each record is ``{question, answer, contexts, ground_truth}``. When ``judge``
    is injected (tests) the fixed per-record scores are aggregated directly; when
    ``judge is None`` a local Ollama-backed judge is built lazily.
    """
    if judge is None:
        from ragcore.config import load_config

        judge = _OllamaJudge(load_config())
    return {
        "ragas": _aggregate(RAGAS_METRICS, records, judge),
        "trulens": _aggregate(TRULENS_METRICS, records, judge),
    }


def _litellm_model(judge_model: str) -> str:
    """Map ``config.eval.judge_model`` (``ollama:qwen3:8b``) to a litellm id
    (``ollama/qwen3:8b``). The first ``:`` separates provider from model; any
    further ``:`` (the Ollama tag) is kept."""
    provider, _, model = judge_model.partition(":")
    return f"{provider}/{model}" if model else judge_model


class _OllamaJudge:
    """Real LLM judge backed by a local Ollama model via litellm.

    Ragas metrics run through a ``LangchainLLMWrapper`` over ``ChatLiteLLM``;
    TruLens RAG-triad feedback runs through the ``LiteLLM`` provider. Both point
    at ``ollama/<model>``. Heavy imports happen in ``__init__`` so they stay out
    of the offline (stubbed) path. Validated live in Task 10.
    """

    def __init__(self, config: Any):
        import os

        from langchain_community.chat_models import ChatLiteLLM
        from ragas.llms import LangchainLLMWrapper
        from trulens.providers.litellm import LiteLLM

        model = _litellm_model(config.eval.judge_model)
        api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

        self._ragas_llm = LangchainLLMWrapper(ChatLiteLLM(model=model, api_base=api_base))
        self._trulens = LiteLLM(model_engine=model, api_base=api_base)

    # --- ragas -------------------------------------------------------------
    def _ragas_metric(self, name: str):
        from ragas.metrics import (  # lazy: avoid import at module load
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        metric = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }[name]
        metric.llm = self._ragas_llm
        return metric

    def _score_ragas(self, name: str, record: dict) -> float:
        from ragas import SingleTurnSample

        sample = SingleTurnSample(
            user_input=record["question"],
            response=record["answer"],
            retrieved_contexts=list(record["contexts"]),
            reference=record.get("ground_truth"),
        )
        return float(self._ragas_metric(name).single_turn_score(sample))

    # --- trulens -----------------------------------------------------------
    def _score_trulens(self, name: str, record: dict) -> float:
        question = record["question"]
        answer = record["answer"]
        contexts = list(record["contexts"])
        context_text = "\n".join(contexts)
        p = self._trulens
        if name == "groundedness":
            score, _ = p.groundedness_measure_with_cot_reasons(context_text, answer)
        elif name == "context_relevance":
            score = _mean(
                [p.context_relevance_with_cot_reasons(question, c)[0] for c in contexts]
            )
        elif name == "answer_relevance":
            score, _ = p.relevance_with_cot_reasons(question, answer)
        else:  # pragma: no cover - guarded by metric tuples
            raise KeyError(name)
        return float(score)

    def score(self, metric: str, record: dict) -> float:
        if metric in RAGAS_METRICS:
            return self._score_ragas(metric, record)
        return self._score_trulens(metric, record)


def _load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def _contexts_for(question: str, store, config, embedder_fn) -> list[str]:
    """Retrieve the chunk texts ragcore would ground an answer on (mirrors the
    ask graph's retrieval) so eval metrics see the same contexts."""
    from ragcore.retrieve import hybrid_search, vector_store_for

    vector_store = vector_store_for(config)
    extra = {"vector_store": vector_store} if vector_store is not None else {}
    chunks = await hybrid_search(store, embedder_fn, question, k=10, **extra)
    return [c["content"] for c in chunks]


def run_eval(config: Any, dataset_path: str | Path | None = None) -> dict:
    """Live evaluation: answer each dataset question with ragcore, retrieve its
    contexts, compute Ragas + TruLens metrics with the local Ollama judge, write
    ``data/eval/report.json`` and print a small table. Exercised in Task 10."""
    import asyncio

    from ragcore.ask import answer_question
    from ragcore.embedding import generate_embedding
    from ragcore.store import Store

    path = Path(dataset_path) if dataset_path else _DEFAULT_DATASET
    dataset = _load_dataset(path)
    store = Store(config.surreal)

    async def embedder_fn(query: str):
        return await generate_embedding(query, config, chunk_size=config.chunking.chunk_size)

    async def _build_records() -> list[dict]:
        records = []
        for item in dataset:
            question = item["question"]
            result = await answer_question(question, store, config, embedder_fn)
            contexts = await _contexts_for(question, store, config, embedder_fn)
            records.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "contexts": contexts,
                    "ground_truth": item.get("ground_truth", ""),
                }
            )
        return records

    records = asyncio.run(_build_records())
    report = compute_metrics(records)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    for family, scores in report.items():
        print(f"\n{family}:")
        for metric, value in scores.items():
            print(f"  {metric:<20} {value:.3f}")
