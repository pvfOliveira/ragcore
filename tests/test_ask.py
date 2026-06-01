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

    async def fake_hybrid(store, embedder_fn, query, k=10):
        return [{"id": "source_embedding:1", "source": "source:0", "content": f"ctx for {query}"}]
    monkeypatch.setattr(ask_mod, "hybrid_search", fake_hybrid)


async def test_answer_question_end_to_end():
    result = await answer_question(
        "What is alpha?", store=object(), config=None, embedder_fn=None,
    )
    assert "FINAL synthesized answer" in result["answer"]
    assert "source:0" in result["citations"]
