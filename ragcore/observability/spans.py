"""A no-op-safe span context manager. ``tracer is None`` → yields ``None`` and
does nothing, so instrumentation is zero-cost when observability is disabled."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def traced_span(tracer: Optional[Any], name: str):
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        yield span
