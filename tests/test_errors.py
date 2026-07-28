from ragcore.errors import (
    ConfigurationError,
    ProviderError,
    RagcoreError,
    classify_error,
    is_transient,
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


def test_is_transient_true_for_network_and_rate_limit():
    assert is_transient(Exception("Connection refused"))
    assert is_transient(Exception("request timed out"))
    assert is_transient(Exception("HTTP 503 Service Unavailable"))
    assert is_transient(Exception("429 rate limit exceeded"))


def test_is_transient_false_for_permanent():
    assert not is_transient(ValueError("No content extracted from /tmp/x"))
    assert not is_transient(Exception("401 invalid api key"))
    assert not is_transient(Exception("something totally unknown"))
