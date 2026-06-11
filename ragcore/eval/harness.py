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

# Golden v1 (versioned, provenance-noted; see golden/MANIFEST.md). The pre-v4
# dataset.jsonl stays on disk for run-history comparability but is no longer
# the default; --dataset still overrides.
_DEFAULT_DATASET = Path(__file__).parent / "golden" / "v1.jsonl"
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
    ``judge is None`` a local Ollama-backed judge is built lazily, selected by
    ``config.eval.framework`` (default ``"ragas"``; ``"deepeval"`` uses
    :class:`~ragcore.eval.deepeval_judge.DeepEvalJudge`).
    """
    if judge is None:
        from ragcore.config import load_config

        judge = _build_judge_for_framework(load_config())
    return {
        "ragas": _aggregate(RAGAS_METRICS, records, judge),
        "trulens": _aggregate(TRULENS_METRICS, records, judge),
    }


def _build_judge_for_framework(config: Any):
    """Select the judge implementation from config.eval.framework."""
    framework = getattr(config.eval, "framework", "ragas")
    if framework == "deepeval":
        from ragcore.eval.deepeval_judge import DeepEvalJudge
        return DeepEvalJudge(config)
    return _OllamaJudge(config)


def _patch_trulens_litellm_instrumentation() -> None:
    """Work around a trulens-vs-litellm version incompatibility.

    TruLens' ``LiteLLM`` provider instruments every class in the ``litellm``
    module that exposes a ``completion`` attribute. In litellm>=1.87 the module
    also exposes ``CallTypes`` — an ``Enum`` whose ``completion`` member is *not*
    a function, so trulens' ``Endpoint._instrument_class`` blows up on
    ``func.__name__``. We wrap that method to skip any member whose resolved
    attribute is not a real named function. Idempotent; a no-op if trulens isn't
    importable or is already patched.
    """
    try:
        from trulens.core.feedback.endpoint import Endpoint
    except Exception:  # pragma: no cover - trulens optional
        return
    if getattr(Endpoint, "_ragcore_instrument_patch", False):
        return

    original = Endpoint._instrument_class

    def _safe_instrument_class(self, cls, method_name):
        func = getattr(cls, method_name, None)
        # Real instrumentable targets are functions/methods carrying __name__;
        # enum members and other descriptors don't, and must be skipped.
        if func is not None and not hasattr(func, "__name__"):
            return None
        return original(self, cls, method_name)

    Endpoint._instrument_class = _safe_instrument_class
    Endpoint._ragcore_instrument_patch = True


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

        import litellm
        from langchain_community.chat_models import ChatLiteLLM
        from langchain_community.embeddings import OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        # Ragas' answer_relevancy asks the judge for several question variants
        # via the OpenAI-style ``n`` sampling param, which Ollama rejects
        # (``UnsupportedParamsError: ollama does not support parameters ['n']``).
        # Let litellm silently drop params the backend can't honour.
        litellm.drop_params = True

        _patch_trulens_litellm_instrumentation()
        from trulens.providers.litellm import LiteLLM

        model = _litellm_model(config.eval.judge_model)
        api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")

        self._ragas_llm = LangchainLLMWrapper(ChatLiteLLM(model=model, api_base=api_base))
        # answer_relevancy embeds the generated questions to compare against the
        # original, so the ragas metric needs an embeddings model too. Reuse the
        # configured Ollama embedding model (e.g. nomic-embed-text).
        embed_role = config.models["embedding"]
        self._ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=embed_role.local_model, base_url=api_base)
        )
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
        # answer_relevancy additionally needs an embeddings model.
        if hasattr(metric, "embeddings"):
            metric.embeddings = self._ragas_embeddings
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
