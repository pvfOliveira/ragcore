import pytest
from pydantic import BaseModel
from ragcore.config import StructuredConfig
from ragcore.structured import generate_structured


class Answer(BaseModel):
    answer: str
    confidence: float


class _Cfg:
    def __init__(self, **kw): self.structured = StructuredConfig(**kw)


async def test_instructor_backend_returns_validated_model(monkeypatch):
    cfg = _Cfg(enabled=True, backend="instructor")
    monkeypatch.setattr("ragcore.structured._instructor_client",
                        lambda config: (object(), "qwen2.5:7b-instruct"))
    monkeypatch.setattr("ragcore.structured._instructor_create",
                        lambda client, model, prompt, schema: Answer(answer="rrf fuses ranks", confidence=0.9))
    out = await generate_structured("what is rrf?", Answer, cfg)
    assert isinstance(out, Answer) and out.confidence == 0.9


async def test_disabled_raises():
    cfg = _Cfg(enabled=False)
    with pytest.raises(Exception):
        await generate_structured("q", Answer, cfg)
