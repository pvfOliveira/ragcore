"""A thin LiteLLM gateway. It slots BEHIND routing.select_model: the local/cloud
decision is unchanged; when [gateway] enabled, the resolved (provider, model) is
routed through litellm with a fallback chain. Exposes the same
``ainvoke(prompt) -> obj.content`` adapter as ragcore.providers so it is drop-in
for _build_chat. Default OFF — esperanto path untouched."""
from __future__ import annotations

from typing import Any


async def _acompletion(**kwargs):
    """Indirection so tests can monkeypatch litellm without importing it."""
    import litellm
    return await litellm.acompletion(**kwargs)


def _litellm_id(provider: str, model: str) -> str:
    return f"{provider}/{model}"


class _GatewayChat:
    def __init__(self, model_id: str, fallback_chain: list[str]):
        self._model_id = model_id
        self._fallbacks = fallback_chain

    async def ainvoke(self, prompt: str):
        resp = await _acompletion(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            fallbacks=self._fallbacks or None,
        )
        content = resp.choices[0].message.content
        return type("_Msg", (), {"content": content})()


class LiteLLMGateway:
    def __init__(self, *, primary: str, fallback_chain: list[str]):
        self.primary = primary
        self.fallback_chain = list(fallback_chain)

    def chat_for(self, provider: str, model: str) -> _GatewayChat:
        return _GatewayChat(_litellm_id(provider, model), self.fallback_chain)


_gateway: LiteLLMGateway | None = None


def get_gateway(config: Any) -> LiteLLMGateway | None:
    """Return a cached gateway when [gateway] enabled, else None."""
    global _gateway
    gw = getattr(config, "gateway", None)
    if gw is None or not gw.enabled:
        return None
    if _gateway is None:
        primary = gw.fallback_chain[0] if gw.fallback_chain else ""
        _gateway = LiteLLMGateway(primary=primary, fallback_chain=gw.fallback_chain[1:])
    return _gateway
