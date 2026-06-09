"""Tracer provider wiring. Phoenix (in-process) is the live-proven backend;
Langfuse is a wired OTLP code path that does not need to run. ``get_tracer``
returns ``None`` (zero overhead) whenever observability is disabled."""
from __future__ import annotations

from typing import Any, Optional

_provider: Any = None  # cached OTel TracerProvider once built


def _build_provider(config: Any):
    """Build a TracerProvider exporting to Phoenix's OTLP collector. Optionally
    also export to Langfuse if its keys are configured (code path only)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider()
    obs = config.observability
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=obs.otlp_endpoint))
    )
    if obs.langfuse_host and obs.langfuse_public_key and obs.langfuse_secret_key:
        import base64
        token = base64.b64encode(
            f"{obs.langfuse_public_key}:{obs.langfuse_secret_key}".encode()
        ).decode()
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(
            endpoint=f"{obs.langfuse_host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {token}"},
        )))
    return provider


def get_tracer(config: Any):
    """Return an OTel tracer, or ``None`` when observability is disabled."""
    global _provider
    obs = getattr(config, "observability", None)
    if obs is None or not obs.enabled:
        return None
    if _provider is None:
        _provider = _build_provider(config)
    return _provider.get_tracer("ragcore")


def _set_provider_for_test(provider: Optional[Any]) -> None:
    """Test hook: inject (or reset with ``None``) the cached provider."""
    global _provider
    _provider = provider
