"""Thin wrappers over esperanto's AIFactory.

esperanto exposes a unified factory; for local models we target Ollama's
OpenAI-compatible endpoint (set OLLAMA_API_BASE in the environment).
"""
from __future__ import annotations

from typing import Any

from esperanto import AIFactory


class _ChatAdapter:
    """Minimal chat wrapper exposing ``async ainvoke(prompt) -> obj.content``.

    Backed by esperanto's native ``achat_complete`` so we don't depend on the
    optional ``langchain_ollama`` / ``langchain_anthropic`` integrations. The
    returned esperanto ``ChatCompletion`` already exposes ``.content``.
    """

    def __init__(self, model: Any):
        self._model = model

    async def ainvoke(self, prompt: str):
        return await self._model.achat_complete([{"role": "user", "content": prompt}])


def build_chat_model(provider: str, model: str, **cfg: Any):
    """Return a chat model exposing ``async ainvoke(prompt) -> obj.content``."""
    llm = AIFactory.create_language(model_name=model, provider=provider, config=cfg)
    return _ChatAdapter(llm)


def build_embedding_model(provider: str, model: str, **cfg: Any):
    """Return an esperanto embedding model (exposes async .aembed(list[str]))."""
    return AIFactory.create_embedding(model_name=model, provider=provider, config=cfg)
