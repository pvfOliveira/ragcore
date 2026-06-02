"""Typed exception hierarchy + error classifier (pattern ported from open-notebook)."""
from __future__ import annotations


class RagcoreError(Exception):
    """Base for all ragcore errors."""


class ConfigurationError(RagcoreError):
    """Missing/invalid configuration or credentials."""


class ProviderError(RagcoreError):
    """An AI provider call failed (network, rate limit, server)."""


class StoreError(RagcoreError):
    """A database/storage operation failed."""


def classify_error(exc: Exception) -> tuple[type[RagcoreError], str]:
    """Map a raw exception to a typed error class + user-friendly message."""
    text = str(exc).lower()
    if "401" in text or "unauthorized" in text or "api key" in text:
        return ConfigurationError, "Invalid or missing API key. Check your credentials."
    if "429" in text or "rate limit" in text:
        return ProviderError, "Provider rate limit hit. Wait and retry."
    if "connection" in text or "timeout" in text or "refused" in text:
        return ProviderError, "Could not reach the model provider. Is the local server running?"
    return ProviderError, str(exc)


_TRANSIENT_MARKERS = (
    "connection", "timeout", "timed out", "refused", "reset",
    "temporarily", "unavailable", "rate limit", "429", "502", "503",
)


def is_transient(exc: Exception) -> bool:
    """True only for retryable transient failures (network / provider availability)."""
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)
