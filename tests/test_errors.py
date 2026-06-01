from ragcore.errors import (
    RagcoreError, ConfigurationError, ProviderError, classify_error,
)


def test_hierarchy():
    assert issubclass(ConfigurationError, RagcoreError)
    assert issubclass(ProviderError, RagcoreError)


def test_classify_rate_limit():
    exc_class, msg = classify_error(Exception("HTTP 429 rate limit exceeded"))
    assert exc_class is ProviderError
    assert "rate limit" in msg.lower()


def test_classify_auth():
    exc_class, msg = classify_error(Exception("401 Unauthorized"))
    assert exc_class is ConfigurationError
    assert "api key" in msg.lower()


def test_classify_unknown_passthrough():
    exc_class, msg = classify_error(Exception("some weird error"))
    assert exc_class is ProviderError
    assert "some weird error" in msg
