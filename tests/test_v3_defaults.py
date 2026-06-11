"""Guard: every RAG v3/v4 capability defaults OFF, so the core paths are unchanged."""
from ragcore.config import (
    AgenticConfig,
    AudioConfig,
    CompressionConfig,
    DocaiConfig,
    MultimodalConfig,
    OnlineEvalConfig,
    QueryRewriteConfig,
    StoreConfig,
    StructuredConfig,
)


def test_all_v3_flags_default_off():
    assert QueryRewriteConfig().enabled is False
    assert AgenticConfig().enabled is False
    assert StructuredConfig().enabled is False
    assert DocaiConfig().enabled is False
    assert MultimodalConfig().vlm_enabled is False
    # default vector backend stays surreal (no adapter routed)
    assert StoreConfig().vector_backend == "surreal"


def test_rag_v4_defaults_are_off():
    """RAG v4 contract: with default config, every v4 feature is inert."""
    assert StoreConfig().vector_backend == "surreal"      # pg/weaviate opt-in
    assert StoreConfig().qdrant_hybrid is False
    assert CompressionConfig().enabled is False
    assert AudioConfig().enabled is False
    assert OnlineEvalConfig().enabled is False
