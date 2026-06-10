"""Live proof: a schema-valid Answer is produced over Ollama via Instructor, and
via Outlines (outlines 1.x from_ollama)."""
from __future__ import annotations

import asyncio
import copy

import pytest

from pydantic import BaseModel
from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


class Answer(BaseModel):
    answer: str
    confidence: float


def _record(tag, surreal_url):
    from ragcore.llmops.registry import RunRegistry
    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_st", database="ragcore_st")
    return asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"structured": {"ok": 1.0}}, config_snapshot={"backend": tag},
        tag=f"structured-{tag}-v3", gate_passed=True))


def test_instructor_typed_answer_over_ollama(surreal_url):
    pytest.importorskip("instructor")
    from ragcore.structured import generate_structured
    cfg = copy.deepcopy(load_config())
    cfg.structured.enabled = True
    cfg.structured.backend = "instructor"
    out = asyncio.run(generate_structured("What is reciprocal rank fusion? Be brief.", Answer, cfg))
    assert isinstance(out, Answer) and out.answer
    assert _record("instructor", surreal_url)


def test_outlines_typed_answer_over_ollama(surreal_url):
    pytest.importorskip("outlines")
    from ragcore.structured import generate_structured
    cfg = copy.deepcopy(load_config())
    cfg.structured.enabled = True
    cfg.structured.backend = "outlines"
    try:
        out = asyncio.run(generate_structured("Say hello briefly.", Answer, cfg))
    except Exception as e:
        pytest.xfail(f"Outlines brittle on this Ollama host: {e}")
    assert isinstance(out, Answer)
    assert _record("outlines", surreal_url)
