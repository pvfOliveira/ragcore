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
