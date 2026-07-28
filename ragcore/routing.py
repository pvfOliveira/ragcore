"""Local-vs-cloud model selection policy.

Generalizes open-notebook's single 105k-token rule into a per-role policy:
pick local unless content exceeds the role budget or cloud is forced — and
only if a cloud model is actually configured for that role.
"""
from __future__ import annotations

from ragcore.chunking import token_count
from ragcore.config import Config
from ragcore.errors import ConfigurationError


def select_model(
    config: Config, role: str, content: str = "", force_cloud: bool = False
) -> tuple[str, str]:
    """Return (provider, model) for the given role."""
    if role not in config.models:
        raise ConfigurationError(f"No model role '{role}' configured.")
    spec = config.models[role]
    has_cloud = bool(spec.cloud_provider and spec.cloud_model)
    wants_cloud = force_cloud or token_count(content) > config.routing.escalate_over_tokens
    if wants_cloud and has_cloud:
        return spec.cloud_provider, spec.cloud_model  # type: ignore[return-value]
    return spec.local_provider, spec.local_model
