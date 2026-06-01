"""Thin wrappers over esperanto's AIFactory.

esperanto exposes a unified factory; for local models we target Ollama's
OpenAI-compatible endpoint (set OLLAMA_API_BASE in the environment).
"""
from __future__ import annotations

from typing import Any

from esperanto import AIFactory


def build_chat_model(provider: str, model: str, **cfg: Any):
    """Return a LangChain-compatible chat model."""
    return AIFactory.create_language(model_name=model, provider=provider, config=cfg).to_langchain()


def build_embedding_model(provider: str, model: str, **cfg: Any):
    """Return an esperanto embedding model (exposes async .aembed(list[str]))."""
    return AIFactory.create_embedding(model_name=model, provider=provider, config=cfg)
