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
