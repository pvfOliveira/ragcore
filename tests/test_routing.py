from ragcore.config import Config, ModelRole, RoutingConfig
from ragcore.routing import select_model


def _cfg():
    return Config(
        models={
            "chat": ModelRole(
                local_provider="ollama", local_model="qwen3:8b",
                cloud_provider="anthropic", cloud_model="claude-sonnet-4-6",
            ),
            "embedding": ModelRole(local_provider="ollama", local_model="nomic-embed-text"),
        },
        routing=RoutingConfig(escalate_over_tokens=100),
    )


def test_local_by_default():
    assert select_model(_cfg(), "chat", content="short") == ("ollama", "qwen3:8b")


def test_escalate_on_large_content():
    big = "word " * 500  # > 100 tokens
    assert select_model(_cfg(), "chat", content=big) == ("anthropic", "claude-sonnet-4-6")


def test_force_cloud():
    assert select_model(_cfg(), "chat", content="short", force_cloud=True) == (
        "anthropic", "claude-sonnet-4-6",
    )


def test_no_cloud_configured_stays_local():
    assert select_model(_cfg(), "embedding", content="x" * 10_000, force_cloud=True) == (
        "ollama", "nomic-embed-text",
    )
