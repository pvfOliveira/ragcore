"""Deterministic tests for LLMLingua-2 context compression.

Disabled → identity (byte-identical default). Enabled → chunk text is
compressed via an injected compressor; ids/sources preserved so citations
still resolve; stats report the ratio. One impl, two honest claims:
context-compression (RAG) and prompt-compression (cost side)."""
from ragcore.compress import compress_context
from ragcore.config import CompressionConfig


class Cfg:
    def __init__(self, **kw):
        self.compression = CompressionConfig(**kw)


class FakeCompressor:
    """Halves the text; records calls."""
    def __init__(self):
        self.calls = []

    def compress_prompt(self, text, rate=0.5):
        self.calls.append({"rate": rate})
        return {"compressed_prompt": text[: max(1, len(text) // 2)]}


CHUNKS = [
    {"id": "c1", "content": "alpha " * 20, "source": "s1", "score": 0.9},
    {"id": "c2", "content": "beta " * 20, "source": "s2", "score": 0.8},
]


def test_disabled_is_identity():
    out, stats = compress_context(CHUNKS, "q", Cfg(enabled=False))
    assert out is CHUNKS                      # the very same object — zero cost
    assert stats["enabled"] is False and stats["ratio"] == 1.0


def test_enabled_compresses_and_preserves_metadata():
    comp = FakeCompressor()
    out, stats = compress_context(CHUNKS, "why alpha?", Cfg(enabled=True, rate=0.4),
                                  compressor=comp)
    assert [c["id"] for c in out] == ["c1", "c2"]
    assert [c["source"] for c in out] == ["s1", "s2"]
    assert all(len(o["content"]) < len(c["content"]) for o, c in zip(out, CHUNKS))
    assert 0 < stats["ratio"] < 1
    assert stats["orig_chars"] > stats["compressed_chars"] > 0
    assert comp.calls[0] == {"rate": 0.4}


def test_original_chunks_not_mutated():
    before = [dict(c) for c in CHUNKS]
    compress_context(CHUNKS, "q", Cfg(enabled=True), compressor=FakeCompressor())
    assert CHUNKS == before


def test_empty_chunks_noop():
    out, stats = compress_context([], "q", Cfg(enabled=True), compressor=FakeCompressor())
    assert out == [] and stats["ratio"] == 1.0


def test_config_defaults_off():
    assert CompressionConfig().enabled is False


def test_empty_content_chunk_skips_compressor():
    """A chunk with content="" must not trigger a compressor call."""
    comp = FakeCompressor()
    chunks = [
        {"id": "c1", "content": "hello world", "source": "s1", "score": 0.9},
        {"id": "c2", "content": "", "source": "s2", "score": 0.8},
    ]
    out, stats = compress_context(chunks, "q", Cfg(enabled=True, rate=0.5),
                                  compressor=comp)
    # only the non-empty chunk triggers the compressor
    assert len(comp.calls) == 1
    # empty chunk passes through unchanged
    assert out[1]["content"] == ""
    # stats are still coherent: orig_chars == len("hello world"), ratio > 0
    assert stats["orig_chars"] == len("hello world")
    assert stats["compressed_chars"] > 0
    assert 0 < stats["ratio"] <= 1.0
