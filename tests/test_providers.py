"""Tests for the _ChatAdapter that wraps esperanto's native achat_complete.

This adapter exists because esperanto's .to_langchain() requires the optional
langchain_ollama integration; the adapter is the contract ask.py relies on
(`await chat.ainvoke(prompt)` -> object with `.content`).
"""
from ragcore.providers import _ChatAdapter


class _FakeChatCompletion:
    """Mimics esperanto's ChatCompletion, which exposes `.content`."""

    def __init__(self, content: str):
        self.content = content


class _FakeLanguageModel:
    def __init__(self):
        self.received = None

    async def achat_complete(self, messages):
        self.received = messages
        return _FakeChatCompletion("the model reply")


async def test_chatadapter_ainvoke_returns_content():
    model = _FakeLanguageModel()
    adapter = _ChatAdapter(model)

    result = await adapter.ainvoke("hello there")

    assert result.content == "the model reply"


async def test_chatadapter_wraps_prompt_as_single_user_message():
    model = _FakeLanguageModel()
    adapter = _ChatAdapter(model)

    await adapter.ainvoke("hello there")

    assert model.received == [{"role": "user", "content": "hello there"}]
