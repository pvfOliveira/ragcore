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


async def test_outlines_backend_returns_validated_model(monkeypatch):
    cfg = _Cfg(enabled=True, backend="outlines")
    monkeypatch.setattr("ragcore.structured._outlines_generate",
                        lambda config, prompt, schema: schema(answer="ok", confidence=0.5))
    out = await generate_structured("q", Answer, cfg)
    assert isinstance(out, Answer) and out.answer == "ok"


async def test_unknown_backend_raises():
    cfg = _Cfg(enabled=True, backend="bogus")
    with pytest.raises(ValueError):
        await generate_structured("q", Answer, cfg)


async def test_answer_structured_retrieves_then_generates(monkeypatch):
    import ragcore.ask as ask

    async def fake_hybrid(store, emb, q, k=10, config=None, **kw):
        return [{"id": "1", "content": "rrf fuses ranks", "source": "s1"}]
    monkeypatch.setattr("ragcore.ask.hybrid_search", fake_hybrid)
    monkeypatch.setattr("ragcore.ask.vector_store_for", lambda c: None)

    async def fake_gen(prompt, schema, config):
        assert "rrf fuses ranks" in prompt   # the retrieved context reached the prompt
        return schema(answer="A", citations=["s1"], confidence=0.7)
    monkeypatch.setattr("ragcore.structured.generate_structured", fake_gen)

    class Cfg:
        structured = StructuredConfig(enabled=True, backend="instructor")
        store = type("S", (), {"vector_backend": "surreal"})()
    out = await ask.answer_structured("what is rrf?", store=object(), config=Cfg(),
                                      embedder_fn=lambda x: [0.0])
    assert out.answer == "A" and out.confidence == 0.7
