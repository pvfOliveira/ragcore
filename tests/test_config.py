from pathlib import Path

from ragcore.config import load_config


def test_load_config(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
[models.chat]
local_provider = "ollama"
local_model = "qwen3:8b"
cloud_provider = "anthropic"
cloud_model = "claude-sonnet-4-6"

[models.embedding]
local_provider = "ollama"
local_model = "nomic-embed-text"

[routing]
escalate_over_tokens = 100000

[chunking]
chunk_size = 400
chunk_overlap = 60
min_chunk_size = 5

[surreal]
url = "ws://localhost:8000/rpc"
namespace = "ragcore"
database = "ragcore"
user = "root"
password = "root"
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.models["chat"].local_model == "qwen3:8b"
    assert cfg.models["chat"].cloud_provider == "anthropic"
    assert cfg.models["embedding"].cloud_model is None
    assert cfg.routing.escalate_over_tokens == 100000
    assert cfg.chunking.chunk_size == 400
    assert cfg.surreal.namespace == "ragcore"


def test_worker_config_defaults_and_override(tmp_path):
    base = (
        '[models.chat]\nlocal_provider="ollama"\nlocal_model="m"\n'
        '[models.embedding]\nlocal_provider="ollama"\nlocal_model="e"\n'
    )
    f1 = tmp_path / "c1.toml"
    f1.write_text(base)
    cfg1 = load_config(f1)
    assert cfg1.worker.max_attempts == 3
    assert cfg1.worker.retry_base_seconds == 2

    f2 = tmp_path / "c2.toml"
    f2.write_text(base + "[worker]\nmax_attempts=5\nretry_base_seconds=1.5\n")
    cfg2 = load_config(f2)
    assert cfg2.worker.max_attempts == 5
    assert cfg2.worker.retry_base_seconds == 1.5


def test_chat_config_default_and_override(tmp_path):
    base = (
        '[models.chat]\nlocal_provider="ollama"\nlocal_model="m"\n'
        '[models.embedding]\nlocal_provider="ollama"\nlocal_model="e"\n'
    )
    f1 = tmp_path / "c1.toml"
    f1.write_text(base)
    assert load_config(f1).chat.history_window == 10
    f2 = tmp_path / "c2.toml"
    f2.write_text(base + "[chat]\nhistory_window=4\n")
    assert load_config(f2).chat.history_window == 4


def test_new_v2_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[models.chat]\nlocal_provider='ollama'\nlocal_model='qwen2.5:7b-instruct'\n"
        "[models.embedding]\nlocal_provider='ollama'\nlocal_model='nomic-embed-text'\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.observability.enabled is False
    assert cfg.observability.backend == "phoenix"
    assert cfg.gateway.enabled is False
    assert cfg.dspy.enabled is False
    assert cfg.eval.framework == "ragas"


def test_new_v2_config_overrides(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[models.chat]\nlocal_provider='ollama'\nlocal_model='qwen2.5:7b-instruct'\n"
        "[models.embedding]\nlocal_provider='ollama'\nlocal_model='nomic-embed-text'\n"
        "[observability]\nenabled=true\nbackend='langfuse'\n"
        "[gateway]\nenabled=true\nfallback_chain=['ollama/qwen2.5:7b-instruct']\n"
        "[dspy]\nenabled=true\n"
        "[eval]\nframework='deepeval'\n"
    )
    cfg = load_config(cfg_file)
    assert cfg.observability.enabled is True
    assert cfg.observability.backend == "langfuse"
    assert cfg.gateway.fallback_chain == ["ollama/qwen2.5:7b-instruct"]
    assert cfg.dspy.enabled is True
    assert cfg.eval.framework == "deepeval"
