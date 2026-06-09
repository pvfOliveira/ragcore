import asyncio


def test_gateway_returns_content_via_adapter(monkeypatch):
    """LiteLLMGateway exposes the same ainvoke(prompt).content adapter shape as
    the esperanto chat adapter, so it is drop-in for _build_chat."""
    import ragcore.gateway as gw

    async def _fake_acompletion(**kwargs):
        class _Choice:
            message = type("M", (), {"content": "routed answer"})()
        return type("R", (), {"choices": [_Choice()]})()

    monkeypatch.setattr(gw, "_acompletion", _fake_acompletion)
    gateway = gw.LiteLLMGateway(primary="ollama/qwen2.5:7b-instruct", fallback_chain=[])
    chat = gateway.chat_for("ollama", "qwen2.5:7b-instruct")
    msg = asyncio.run(chat.ainvoke("hello"))
    assert msg.content == "routed answer"


def test_gateway_passes_fallbacks(monkeypatch):
    """The configured fallback chain is forwarded to litellm."""
    import ragcore.gateway as gw
    seen = {}

    async def _fake_acompletion(**kwargs):
        seen.update(kwargs)
        class _Choice:
            message = type("M", (), {"content": "ok"})()
        return type("R", (), {"choices": [_Choice()]})()

    monkeypatch.setattr(gw, "_acompletion", _fake_acompletion)
    gateway = gw.LiteLLMGateway(primary="ollama/m", fallback_chain=["ollama/backup"])
    asyncio.run(gateway.chat_for("ollama", "m").ainvoke("x"))
    assert seen["model"] == "ollama/m"
    assert seen["fallbacks"] == ["ollama/backup"]


def test_build_chat_uses_gateway_only_when_enabled(monkeypatch):
    """[gateway] enabled=false keeps the existing esperanto path."""
    import ragcore.ask as ask
    from ragcore.config import Config, GatewayConfig, ModelRole

    calls = {"esperanto": 0, "gateway": 0}
    monkeypatch.setattr(ask, "build_chat_model",
                        lambda p, m, **k: calls.__setitem__("esperanto", calls["esperanto"] + 1) or object())

    cfg_off = Config(
        models={"chat": ModelRole(local_provider="ollama", local_model="m"),
                "embedding": ModelRole(local_provider="ollama", local_model="e")},
        gateway=GatewayConfig(enabled=False))
    ask._build_chat(cfg_off, content="q")
    assert calls["esperanto"] == 1 and calls["gateway"] == 0
