"""Live (Ollama-backed) tier for the production RAG upgrade.

Every test here is ``@pytest.mark.live`` and is skipped by ``conftest.py``'s
``pytest_collection_modifyitems`` unless a local Ollama answers on
``http://localhost:11434``. These exercise the paths the deterministic suite
cannot: real ``nomic-embed-text`` embeddings flowing through the pluggable
vector backends, and the real Ragas + TruLens judge running against a local
Ollama chat model. Assertions are outcome/structure based (relevant chunk wins,
metric keys present, scores are floats in ``[0, 1]``) so they tolerate model
variance.
"""
from __future__ import annotations

import copy

import pytest

from ragcore.config import load_config
from ragcore.embedding import generate_embedding
from ragcore.eval.harness import (
    RAGAS_METRICS,
    TRULENS_METRICS,
    compute_metrics,
)
from ragcore.vectorstores.base import make_vector_store

pytestmark = pytest.mark.live


# A tiny corpus where one chunk clearly answers the probe and the others are
# semantically distant, so a real embedding model must rank the right one first.
_CORPUS = [
    ("c1", "Photosynthesis lets plants turn sunlight into chemical energy."),
    ("c2", "The mitochondrion is the powerhouse of the cell."),
    ("c3", "Quarterly revenue grew on strong cloud subscription sales."),
]
_PROBE = "How do plants make energy from light?"
_EXPECTED_ID = "c1"


def _config_for(backend: str, tmp_path):
    """Load the real config.toml (Ollama models) and point *backend* at tmp."""
    cfg = copy.deepcopy(load_config())
    cfg.store.vector_backend = backend
    cfg.store.chroma_path = str(tmp_path / "chroma")
    cfg.store.faiss_path = str(tmp_path / "faiss")
    cfg.store.milvus_uri = str(tmp_path / "milvus.db")
    return cfg


async def _embed(text: str, cfg) -> list[float]:
    return await generate_embedding(text, cfg, chunk_size=cfg.chunking.chunk_size)


async def _roundtrip(backend: str, tmp_path):
    cfg = _config_for(backend, tmp_path)
    vs = make_vector_store(cfg)
    chunks = [
        {"id": cid, "text": text, "embedding": await _embed(text, cfg)}
        for cid, text in _CORPUS
    ]
    await vs.add_embeddings("src-live", chunks)
    query_vec = await _embed(_PROBE, cfg)
    results = await vs.vector_search(query_vec, k=3)
    assert results, f"{backend}: vector_search returned no hits"
    assert results[0]["id"] == _EXPECTED_ID, (
        f"{backend}: expected {_EXPECTED_ID} first, got "
        f"{[r['id'] for r in results]}"
    )
    # The winning hit carries its text and a numeric score.
    assert results[0]["text"] == dict(_CORPUS)[_EXPECTED_ID]
    assert isinstance(results[0]["score"], float)


async def test_chroma_real_embeddings_roundtrip(tmp_path):
    await _roundtrip("chroma", tmp_path)


async def test_faiss_real_embeddings_roundtrip(tmp_path):
    await _roundtrip("faiss", tmp_path)


async def test_milvus_real_embeddings_roundtrip(tmp_path):
    pytest.importorskip("pymilvus")
    await _roundtrip("milvus", tmp_path)


def test_eval_real_judge_report_shape():
    """Honesty bar: the real Ragas + TruLens judge against a local Ollama model
    produces all six metrics as floats in [0, 1] over a canned grounded record.

    This is the same code path as ``compute_metrics(records, judge=None)`` — that
    branch builds ``_OllamaJudge(load_config())`` using ``config.toml``'s
    ``[eval] judge_model``. Here we build the judge explicitly so the test can
    pick a faster non-reasoning instruct model (qwen3's chain-of-thought makes
    each ragas/trulens call take minutes) and keep the live run tractable; the
    judge object and aggregation are identical.
    """
    from ragcore.config import load_config as _load
    from ragcore.eval.harness import _OllamaJudge

    cfg = _load()
    # A non-reasoning instruct model keeps per-metric latency reasonable.
    cfg.eval.judge_model = "ollama:qwen2.5:7b-instruct"
    judge = _OllamaJudge(cfg)

    records = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "contexts": [
                "France is a country in Western Europe.",
                "Paris is the capital and largest city of France.",
            ],
            "ground_truth": "Paris",
        },
    ]

    report = compute_metrics(records, judge=judge)  # real local Ollama judge

    assert set(report) == {"ragas", "trulens"}
    assert set(report["ragas"]) == set(RAGAS_METRICS)
    assert set(report["trulens"]) == set(TRULENS_METRICS)
    for family in ("ragas", "trulens"):
        for metric, value in report[family].items():
            assert isinstance(value, float), f"{family}.{metric} not a float: {value!r}"
            assert 0.0 <= value <= 1.0, f"{family}.{metric} out of [0,1]: {value}"
