"""OpenTelemetry GenAI semantic-convention helpers.

The GenAI conventions are still marked *experimental* upstream. We emit ONLY the
stable, widely-adopted subset and skip volatile attributes (events/messages,
tool-call details). Emitted keys:

    gen_ai.system            — provider id (e.g. "ollama")
    gen_ai.operation.name    — "chat"
    gen_ai.request.model     — model name
    gen_ai.usage.input_tokens / gen_ai.usage.output_tokens — when known
"""
from __future__ import annotations

from typing import Any, Optional

STABLE_GEN_AI_ATTRS = (
    "gen_ai.system",
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
)


def set_gen_ai_attributes(
    span: Any, *, system: str, operation: str, model: str,
    input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
) -> None:
    span.set_attribute("gen_ai.system", system)
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("gen_ai.request.model", model)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
