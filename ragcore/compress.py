"""LLMLingua-2 context compression for retrieved chunks (pre-prompt).

One implementation, two honest claims: RAG-side *context compression* and
cost-side *prompt compression* — the measured ratio is the cost evidence.
The LLMLingua-2 encoder (a small token-classification transformer) runs
locally on CPU/MPS; lazy-initialized, cached at module level.
Compression is query-agnostic: LLMLingua-2's token-classification path does
not accept a question parameter (only LLMLingua-1's coarse mode does)."""
from __future__ import annotations

from typing import Any

_compressor: Any = None  # cached PromptCompressor (heavy init)


def _get_compressor(config) -> Any:
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor  # lazy — optional extra
        _compressor = PromptCompressor(
            model_name=config.compression.model,
            use_llmlingua2=True,
            device_map=config.compression.device,
        )
    return _compressor


_NOOP_STATS = {"enabled": False, "orig_chars": 0, "compressed_chars": 0, "ratio": 1.0}


def compress_context(chunks: list[dict], query: str, config,
                     compressor: Any = None) -> tuple[list[dict], dict]:
    """Compress each chunk's ``content`` per ``config.compression``.

    Returns ``(chunks, stats)``. Disabled or empty input → the original list
    unchanged (identity) and no-op stats. Ids/sources are preserved so
    citations keep resolving. ``compressor`` is injectable for tests.

    ``query`` is accepted for API stability but NOT used by the LLMLingua-2
    backend — its compress path is query-agnostic (the question-aware path
    exists only in LLMLingua-1 coarse mode; verified against llmlingua 0.2.2)."""
    cc = config.compression
    if not cc.enabled or not chunks:
        return chunks, dict(_NOOP_STATS)
    comp = compressor if compressor is not None else _get_compressor(config)
    out: list[dict] = []
    orig = compressed = 0
    for c in chunks:
        text = c.get("content", "")
        orig += len(text)
        if text:
            res = comp.compress_prompt(text, rate=cc.rate)
            new_text = res["compressed_prompt"]
        else:
            new_text = text
        compressed += len(new_text)
        merged = dict(c)
        merged["content"] = new_text
        out.append(merged)
    ratio = (compressed / orig) if orig else 1.0
    stats = {"enabled": True, "orig_chars": orig,
             "compressed_chars": compressed, "ratio": ratio}
    _record_span_attrs(stats)
    return out, stats


def _record_span_attrs(stats: dict) -> None:
    """Attach compression stats to the current OTel span, if tracing is on."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        span.set_attribute("ragcore.compression.ratio", stats["ratio"])
        span.set_attribute("ragcore.compression.orig_chars", stats["orig_chars"])
        span.set_attribute("ragcore.compression.compressed_chars", stats["compressed_chars"])
    except Exception:
        pass
