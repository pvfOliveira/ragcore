def test_set_gen_ai_attributes_sets_stable_keys():
    from ragcore.observability.otel import set_gen_ai_attributes

    class FakeSpan:
        def __init__(self):
            self.attrs = {}
        def set_attribute(self, key, value):
            self.attrs[key] = value

    span = FakeSpan()
    set_gen_ai_attributes(
        span, system="ollama", operation="chat",
        model="qwen2.5:7b-instruct", input_tokens=12, output_tokens=34,
    )
    assert span.attrs["gen_ai.system"] == "ollama"
    assert span.attrs["gen_ai.operation.name"] == "chat"
    assert span.attrs["gen_ai.request.model"] == "qwen2.5:7b-instruct"
    assert span.attrs["gen_ai.usage.input_tokens"] == 12
    assert span.attrs["gen_ai.usage.output_tokens"] == 34


def test_set_gen_ai_attributes_omits_none_token_counts():
    from ragcore.observability.otel import set_gen_ai_attributes

    class FakeSpan:
        def __init__(self):
            self.attrs = {}
        def set_attribute(self, key, value):
            self.attrs[key] = value

    span = FakeSpan()
    set_gen_ai_attributes(span, system="ollama", operation="chat", model="m")
    assert "gen_ai.usage.input_tokens" not in span.attrs
    assert "gen_ai.usage.output_tokens" not in span.attrs


import pytest


def _inmemory_tracer():
    """Build a real OTel tracer backed by an in-memory exporter (no Phoenix)."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_get_tracer_returns_none_when_disabled():
    from ragcore.config import ObservabilityConfig
    from ragcore.observability.tracing import get_tracer

    class Cfg:
        observability = ObservabilityConfig(enabled=False)

    assert get_tracer(Cfg()) is None


def test_traced_span_records_when_enabled_and_noop_when_disabled():
    from ragcore.observability.spans import traced_span
    from ragcore.observability.otel import set_gen_ai_attributes
    from ragcore.observability.tracing import _set_provider_for_test

    provider, exporter = _inmemory_tracer()
    _set_provider_for_test(provider)
    tracer = provider.get_tracer("ragcore")

    with traced_span(tracer, "chat qwen") as span:
        set_gen_ai_attributes(span, system="ollama", operation="chat", model="qwen")

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["chat qwen"]
    assert spans[0].attributes["gen_ai.system"] == "ollama"

    # None tracer → no-op, span is None, nothing raised
    with traced_span(None, "noop") as span:
        assert span is None

    _set_provider_for_test(None)  # reset global


def test_ask_emits_root_and_gen_ai_spans(monkeypatch):
    import asyncio
    import ragcore.ask as ask
    from ragcore.config import (
        Config, ModelRole, ObservabilityConfig, CacheConfig, CostConfig,
    )
    from ragcore.observability.tracing import _set_provider_for_test

    provider, exporter = _inmemory_tracer()
    _set_provider_for_test(provider)

    class _Msg:
        def __init__(self, content): self.content = content
    class _FakeChat:
        async def ainvoke(self, prompt):
            if "searches" in prompt or "strategy" in prompt.lower():
                return _Msg('{"searches": ["q"]}')
            return _Msg("an answer")
    monkeypatch.setattr(ask, "build_chat_model", lambda provider, model, **k: _FakeChat())

    async def _embedder(q): return [0.1, 0.2, 0.3]
    async def _fake_hybrid(*a, **k): return [{"source": "s1", "content": "ctx"}]
    monkeypatch.setattr(ask, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(ask, "vector_store_for", lambda cfg: None)

    cfg = Config(
        models={"chat": ModelRole(local_provider="ollama", local_model="qwen2.5:7b-instruct"),
                "embedding": ModelRole(local_provider="ollama", local_model="nomic-embed-text")},
        observability=ObservabilityConfig(enabled=True),
        cache=CacheConfig(enabled=False), cost=CostConfig(enabled=False),
    )

    result = asyncio.run(ask.answer_question("what is q?", store=None, config=cfg, embedder_fn=_embedder))
    assert result["answer"] == "an answer"

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert any(n == "ask" for n in names), names                       # root
    gen_ai = [s for s in spans if s.attributes.get("gen_ai.system") == "ollama"]
    assert gen_ai, "expected at least one gen_ai span with stable attributes"
    assert gen_ai[0].attributes["gen_ai.request.model"] == "qwen2.5:7b-instruct"
    assert gen_ai[0].attributes["gen_ai.operation.name"] == "chat"
    stages = {s.attributes.get("ragcore.pipeline.stage") for s in gen_ai}
    assert {"strategy", "retrieve_answer", "synthesize"} <= stages

    _set_provider_for_test(None)


def test_ask_is_byte_identical_when_observability_disabled(monkeypatch):
    """Disabled tracer must not change the result and must emit no spans."""
    import asyncio
    import ragcore.ask as ask
    from ragcore.config import Config, ModelRole, ObservabilityConfig, CacheConfig, CostConfig
    from ragcore.observability.tracing import _set_provider_for_test

    provider, exporter = _inmemory_tracer()
    _set_provider_for_test(provider)

    class _Msg:
        def __init__(self, content): self.content = content
    class _FakeChat:
        async def ainvoke(self, prompt):
            return _Msg('{"searches": ["q"]}' if "strategy" in prompt.lower() else "an answer")
    monkeypatch.setattr(ask, "build_chat_model", lambda p, m, **k: _FakeChat())
    monkeypatch.setattr(ask, "hybrid_search", lambda *a, **k: _async_list())
    monkeypatch.setattr(ask, "vector_store_for", lambda cfg: None)

    cfg = Config(
        models={"chat": ModelRole(local_provider="ollama", local_model="m"),
                "embedding": ModelRole(local_provider="ollama", local_model="e")},
        observability=ObservabilityConfig(enabled=False),
        cache=CacheConfig(enabled=False), cost=CostConfig(enabled=False),
    )
    async def _embedder(q): return [0.1]
    result = asyncio.run(ask.answer_question("q?", None, cfg, _embedder))
    assert result["answer"] == "an answer"
    assert exporter.get_finished_spans() == ()   # nothing emitted when disabled
    _set_provider_for_test(None)


async def _async_list():
    return [{"source": "s1", "content": "ctx"}]
