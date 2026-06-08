import pytest

from ragcore import chat as chat_mod
from ragcore.chat import chat_turn
from ragcore.config import ChatConfig, Config, ModelRole


def _config():
    return Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="m"),
            "embedding": ModelRole(local_provider="ollama", local_model="e"),
        },
        chat=ChatConfig(history_window=10),
    )


class FakeChat:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1

        class _Msg:
            content = self._responses.pop(0)

        return _Msg()


class FakeSessionStore:
    def __init__(self, history):
        self._history = list(history)
        self.added = []

    async def get_history(self, session_id, limit=None):
        # Mirror SessionStore.get_history exactly (limit=0 -> []).
        if limit is None:
            return self._history
        return self._history[-limit:] if limit > 0 else []

    async def add_message(self, session_id, role, content, citations=None):
        self.added.append((role, content, citations))


@pytest.fixture(autouse=True)
def patch_hybrid(monkeypatch):
    async def fake_hybrid(store, embedder_fn, query, k=10, **kwargs):
        return [{"id": "source_embedding:1", "source": "source:0", "content": "ctx"}]
    monkeypatch.setattr(chat_mod, "hybrid_search", fake_hybrid)


async def test_first_turn_skips_reformulation_and_persists(monkeypatch):
    chat = FakeChat(["The answer about X [source:0]"])  # answer only
    monkeypatch.setattr(chat_mod, "_build_chat",
                        lambda config, content="", force_cloud=False: chat)
    ss = FakeSessionStore(history=[])
    result = await chat_turn(ss, object(), _config(), "chat_session:1", "What is X?", None)

    assert chat.calls == 1  # no reformulation on the first turn
    assert "answer about X" in result["answer"]
    assert result["citations"] == ["source:0"]
    assert [(r, c) for r, c, _ in ss.added] == [
        ("user", "What is X?"), ("assistant", result["answer"])]
    assert ss.added[1][2] == ["source:0"]


async def test_followup_turn_reformulates(monkeypatch):
    chat = FakeChat(["X license", "It is MIT [source:0]"])  # reformulate, then answer
    monkeypatch.setattr(chat_mod, "_build_chat",
                        lambda config, content="", force_cloud=False: chat)
    ss = FakeSessionStore(history=[
        {"role": "user", "content": "What is X?", "citations": None, "created": "t"},
        {"role": "assistant", "content": "X is a database", "citations": [], "created": "t"},
    ])
    result = await chat_turn(ss, object(), _config(), "chat_session:1",
                             "what about its license?", None)

    assert chat.calls == 2  # reformulation + answer
    assert "MIT" in result["answer"]
    assert ("user", "what about its license?") in [(r, c) for r, c, _ in ss.added]


async def test_blank_reformulation_falls_back_to_raw_message(monkeypatch):
    # If the reformulation model returns blank, the search query falls back to the
    # raw user message (chat.py: `_clean(...) or message`).
    chat = FakeChat(["   ", "answer [source:0]"])  # blank reformulation, then answer
    monkeypatch.setattr(chat_mod, "_build_chat",
                        lambda config, content="", force_cloud=False: chat)
    captured = {}

    async def cap_hybrid(store, embedder_fn, query, k=10, **kwargs):
        captured["query"] = query
        return [{"id": "e:1", "source": "source:0", "content": "ctx"}]

    monkeypatch.setattr(chat_mod, "hybrid_search", cap_hybrid)
    ss = FakeSessionStore(history=[
        {"role": "user", "content": "prior", "citations": None, "created": "t"}])

    await chat_turn(ss, object(), _config(), "chat_session:1", "the real message", None)
    assert captured["query"] == "the real message"
