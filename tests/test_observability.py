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
