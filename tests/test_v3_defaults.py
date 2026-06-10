"""Guard: every RAG v3 capability defaults OFF, so the core paths are unchanged."""
from ragcore.config import (QueryRewriteConfig, AgenticConfig, StructuredConfig,
                            DocaiConfig, StoreConfig, MultimodalConfig)


def test_all_v3_flags_default_off():
    assert QueryRewriteConfig().enabled is False
    assert AgenticConfig().enabled is False
    assert StructuredConfig().enabled is False
    assert DocaiConfig().enabled is False
    assert MultimodalConfig().vlm_enabled is False
    # default vector backend stays surreal (no adapter routed)
    assert StoreConfig().vector_backend == "surreal"
