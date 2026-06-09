"""Live proof: LiteLLM routes a real call to Ollama, and a forced-primary-failure
falls back down the chain."""
from __future__ import annotations

import asyncio
import copy

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live


def test_litellm_routes_real_call_to_ollama(surreal_url):
    pytest.importorskip("litellm")
    from ragcore.gateway import LiteLLMGateway
    from ragcore.llmops.registry import RunRegistry

    gateway = LiteLLMGateway(primary="ollama/qwen2.5:7b-instruct", fallback_chain=[])
    chat = gateway.chat_for("ollama", "qwen2.5:7b-instruct")
    msg = asyncio.run(chat.ainvoke("Reply with the single word: ready"))
    assert msg.content and isinstance(msg.content, str)

    # Fallback: a bogus primary falls back to the working Ollama model.
    fb = LiteLLMGateway(primary="ollama/does-not-exist",
                        fallback_chain=["ollama/qwen2.5:7b-instruct"])
    fb_chat = fb.chat_for("ollama", "does-not-exist")
    fb_msg = asyncio.run(fb_chat.ainvoke("Reply with the single word: ok"))
    assert fb_msg.content

    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_gw", database="ragcore_gw")
    run_id = asyncio.run(RunRegistry(cfg.surreal).record(
        metrics={"gateway": {"routed": 1.0, "fallback": 1.0}},
        config_snapshot={"primary": "ollama/qwen2.5:7b-instruct"},
        tag="gateway-v2", gate_passed=True))
    assert run_id
