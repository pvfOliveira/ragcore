"""Sampled online scoring of the live chat path ("prod" = the local chat
path, labeled as such). Sampling is injectable-RNG-deterministic; judge
failures NEVER break the user's turn; scores land on an online_eval OTel span."""
import random

import pytest

import ragcore.eval.online as online
from ragcore.config import OnlineEvalConfig


class Cfg:
    def __init__(self, **kw):
        self.online_eval = OnlineEvalConfig(**kw)
        from ragcore.config import ObservabilityConfig
        self.observability = ObservabilityConfig()


class FixedJudge:
    def __init__(self, score=0.8, fail=False):
        self._score, self._fail = score, fail
        self.calls = []

    def score(self, metric, record):
        if self._fail:
            raise RuntimeError("judge exploded")
        self.calls.append((metric, record["question"]))
        return self._score


def test_should_sample_rate_bounds():
    rng = random.Random(7)
    assert not any(online.should_sample(0.0, rng) for _ in range(50))
    assert all(online.should_sample(1.0, rng) for _ in range(50))


async def test_enabled_but_not_sampled_returns_none(monkeypatch):
    def boom(cfg):
        raise AssertionError("judge must not be built when not sampled")
    monkeypatch.setattr(online, "_build_judge", boom)
    out = await online.maybe_score_turn("q", "a", [{"content": "c"}],
                                        Cfg(enabled=True, sample_rate=0.0),
                                        rng=random.Random(0))
    assert out is None


async def test_disabled_returns_none_without_judging(monkeypatch):
    def boom(cfg):
        raise AssertionError("judge must not be built when disabled")
    monkeypatch.setattr(online, "_build_judge", boom)
    out = await online.maybe_score_turn("q", "a", [{"content": "ctx"}],
                                        Cfg(enabled=False), rng=random.Random(0))
    assert out is None


async def test_sampled_turn_scores_with_existing_judge(monkeypatch):
    judge = FixedJudge(score=0.75)
    monkeypatch.setattr(online, "_build_judge", lambda cfg: judge)
    out = await online.maybe_score_turn(
        "why?", "because.", [{"content": "ctx one"}, {"content": "ctx two"}],
        Cfg(enabled=True, sample_rate=1.0), rng=random.Random(0))
    assert out == {"groundedness": 0.75}
    metric, q = judge.calls[0]
    assert metric == "groundedness" and q == "why?"


async def test_judge_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(online, "_build_judge", lambda cfg: FixedJudge(fail=True))
    out = await online.maybe_score_turn("q", "a", [{"content": "c"}],
                                        Cfg(enabled=True, sample_rate=1.0),
                                        rng=random.Random(0))
    assert out is None                       # logged, never raised


async def test_span_attributes_via_inmemory_exporter(monkeypatch):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from ragcore.observability.tracing import _set_provider_for_test

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _set_provider_for_test(provider)
    try:
        cfg = Cfg(enabled=True, sample_rate=1.0)
        cfg.observability.enabled = True
        monkeypatch.setattr(online, "_build_judge", lambda c: FixedJudge(score=0.9))
        await online.maybe_score_turn("q", "a", [{"content": "c"}], cfg,
                                      rng=random.Random(0))
        spans = exporter.get_finished_spans()
        ev = [s for s in spans if s.name == "online_eval"]
        assert ev, "no online_eval span emitted"
        attrs = dict(ev[0].attributes)
        assert attrs["ragcore.eval.sampled"] is True
        assert attrs["ragcore.eval.groundedness"] == pytest.approx(0.9)
    finally:
        _set_provider_for_test(None)


async def test_chat_turn_invokes_online_eval_when_enabled(monkeypatch):
    """chat_turn wires through maybe_score_turn (and not when disabled)."""
    import ragcore.chat as chat_mod

    calls = []

    async def fake_maybe(question, answer, chunks, config, rng=None):
        calls.append(answer)
        return {"groundedness": 1.0}

    monkeypatch.setattr("ragcore.eval.online.maybe_score_turn", fake_maybe)

    class FakeSessions:
        async def get_history(self, sid, limit):
            return []

        async def add_message(self, *a, **k):
            pass

    class FakeChatModel:
        async def ainvoke(self, prompt):
            class M:
                content = "an answer"
            return M()

    async def fake_hybrid(*a, **k):
        return [{"id": "c1", "content": "ctx", "source": "s1", "score": 1.0}]

    monkeypatch.setattr(chat_mod, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(chat_mod, "vector_store_for", lambda cfg: None)
    monkeypatch.setattr(chat_mod, "_build_chat", lambda cfg, content: FakeChatModel())

    from ragcore.config import (
        ChatConfig,
        CompressionConfig,
        GraphConfig,
        ObservabilityConfig,
        OnlineEvalConfig,
    )

    class C:
        chat = ChatConfig()
        graph = GraphConfig()
        compression = CompressionConfig()
        online_eval = OnlineEvalConfig(enabled=True, sample_rate=1.0)
        observability = ObservabilityConfig()

    async def embedder(q):
        return [0.0]

    result = await chat_mod.chat_turn(FakeSessions(), object(), C(), "sid",
                                      "hello", embedder)
    assert result["answer"] == "an answer"
    assert calls == ["an answer"]

    calls.clear()
    C.online_eval = OnlineEvalConfig(enabled=False)
    await chat_mod.chat_turn(FakeSessions(), object(), C(), "sid", "hello", embedder)
    assert calls == []
