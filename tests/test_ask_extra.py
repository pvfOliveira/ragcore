"""Tests for ask._clean thinking-tag stripping and fan-out answer accumulation."""
from ragcore import ask as ask_mod
from ragcore.ask import _build_graph, _clean


def test_clean_strips_closed_think_block():
    assert _clean("before<think>private reasoning</think>after") == "beforeafter"


def test_clean_strips_dangling_unclosed_think():
    # A reasoning model truncated mid-thought leaks an unclosed <think>.
    assert _clean("the answer<think>truncated reasoning that never closes") == "the answer"


def test_clean_trims_surrounding_whitespace():
    assert _clean("  <think>x</think>  hello  ") == "hello"


class _FakeChat:
    def __init__(self, responses):
        self._responses = list(responses)

    async def ainvoke(self, prompt):
        class _Msg:
            content = self._responses.pop(0)

        return _Msg()


async def test_fanout_accumulates_one_answer_per_search(monkeypatch):
    # strategy -> 3 searches, so exactly 3 retrieve_answer nodes must run and
    # the operator.add reducer must accumulate exactly 3 answers.
    chat = _FakeChat([
        '{"searches": ["alpha", "beta", "gamma"]}',  # strategy
        "ans-1", "ans-2", "ans-3",                    # one per retrieve_answer
        "final synthesized answer",                   # synthesize
    ])
    monkeypatch.setattr(ask_mod, "_build_chat",
                        lambda config, content="", force_cloud=False: chat)
    monkeypatch.setattr(ask_mod, "_select_and_build",
                        lambda config, content="", force_cloud=False: (chat, "test", "test-model"))

    async def fake_hybrid(store, embedder_fn, query, k=10, **kwargs):
        return [{"id": "source_embedding:1", "source": "source:0", "content": "ctx"}]

    monkeypatch.setattr(ask_mod, "hybrid_search", fake_hybrid)

    graph = _build_graph()
    result = await graph.ainvoke({
        "question": "q", "answers": [],
        "_store": object(), "_config": None, "_embedder_fn": None, "_force_cloud": False,
    })

    assert len(result["answers"]) == 3
    assert result["answer"] == "final synthesized answer"
