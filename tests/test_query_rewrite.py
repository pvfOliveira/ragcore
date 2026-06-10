from ragcore.config import QueryRewriteConfig
from ragcore.query_rewrite import rewrite_query


class _Cfg:
    def __init__(self, **kw): self.query_rewrite = QueryRewriteConfig(**kw)


async def test_disabled_returns_query_unchanged():
    cfg = _Cfg(enabled=False)
    async def chat(_): return "should not be called"
    out = await rewrite_query("what is rrf?", cfg, chat)
    assert out == ["what is rrf?"]


async def test_multi_query_parses_numbered_list():
    cfg = _Cfg(enabled=True, strategy="multi_query", n=3)
    async def chat(_): return "1. define rrf\n2. reciprocal rank fusion meaning\n3. rrf formula"
    out = await rewrite_query("what is rrf?", cfg, chat)
    assert out == ["define rrf", "reciprocal rank fusion meaning", "rrf formula"]


async def test_hyde_uses_hypothetical_doc_as_query():
    cfg = _Cfg(enabled=True, strategy="hyde")
    async def chat(_): return "Reciprocal Rank Fusion combines ranked lists by summing 1/(k+rank)."
    out = await rewrite_query("what is rrf?", cfg, chat)
    assert len(out) == 1 and "Reciprocal Rank Fusion" in out[0]


async def test_decompose_returns_subquestions():
    cfg = _Cfg(enabled=True, strategy="decompose", n=2)
    async def chat(_): return "1. what is a ranked list?\n2. how does rrf weight ranks?"
    out = await rewrite_query("explain rrf", cfg, chat)
    assert out == ["what is a ranked list?", "how does rrf weight ranks?"]


async def test_unknown_strategy_raises():
    import pytest
    cfg = _Cfg(enabled=True, strategy="bogus")
    async def chat(_): return ""
    with pytest.raises(ValueError):
        await rewrite_query("q", cfg, chat)


async def test_expand_and_fuse_runs_search_per_variant():
    from ragcore import query_rewrite as qr
    calls = []
    async def fake_search(term):
        calls.append(term)
        return [{"id": f"{term}-1", "content": term, "source": "s", "score": 1.0}]
    cfg = _Cfg(enabled=True, strategy="multi_query", n=2)
    async def chat(_): return "1. a\n2. b"
    fused = await qr.expand_and_fuse("orig", cfg, chat, fake_search, k=10)
    assert calls == ["a", "b"]
    assert {c["id"] for c in fused} == {"a-1", "b-1"}
