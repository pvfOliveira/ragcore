import pytest

from ragcore import ask as ask_mod
from ragcore.ask import answer_question


class FakeChat:
    def __init__(self, responses):
        self._responses = list(responses)

    async def ainvoke(self, prompt):
        class _Msg:
            content = self._responses.pop(0)
        return _Msg()


@pytest.fixture(autouse=True)
def patch_ask(monkeypatch):
    chat = FakeChat([
        '{"searches": ["alpha", "beta"]}',  # strategy
        "alpha partial answer",              # per-term answer 1
        "beta partial answer",               # per-term answer 2
        "FINAL synthesized answer [source:source:0]",  # final
    ])
    monkeypatch.setattr(ask_mod, "_build_chat", lambda config, content="", force_cloud=False: chat)
    monkeypatch.setattr(ask_mod, "_select_and_build",
                        lambda config, content="", force_cloud=False: (chat, "test", "test-model"))

    async def fake_hybrid(store, embedder_fn, query, k=10, **kwargs):
        return [{"id": "source_embedding:1", "source": "source:0", "content": f"ctx for {query}"}]
    monkeypatch.setattr(ask_mod, "hybrid_search", fake_hybrid)


async def test_answer_question_end_to_end():
    result = await answer_question(
        "What is alpha?", store=object(), config=None, embedder_fn=None,
    )
    assert "FINAL synthesized answer" in result["answer"]
    assert "source:0" in result["citations"]


async def test_strategy_handles_non_object_json(monkeypatch):
    # A local model returning a bare JSON array must NOT crash; it should fall
    # back to using the question itself as the single search term.
    chat = FakeChat([
        '["alpha", "beta"]',                 # strategy: valid JSON but not an object
        "partial answer for fallback",       # one retrieve_answer (1 search = the question)
        "FINAL answer about the topic",      # synthesize
    ])
    monkeypatch.setattr(ask_mod, "_build_chat",
                        lambda config, content="", force_cloud=False: chat)
    monkeypatch.setattr(ask_mod, "_select_and_build",
                        lambda config, content="", force_cloud=False: (chat, "test", "test-model"))

    async def fake_hybrid(store, embedder_fn, query, k=10, **kwargs):
        return [{"id": "source_embedding:1", "source": "source:0", "content": "ctx"}]
    monkeypatch.setattr(ask_mod, "hybrid_search", fake_hybrid)

    result = await answer_question("What is the topic?", store=object(), config=None, embedder_fn=None)
    assert "FINAL answer" in result["answer"]
