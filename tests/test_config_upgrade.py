from pathlib import Path

from ragcore.config import load_config

_MODELS = (
    '[models.chat]\nlocal_provider="ollama"\nlocal_model="qwen3:8b"\n'
    '[models.embedding]\nlocal_provider="ollama"\nlocal_model="nomic-embed-text"\n'
)


def test_upgrade_sections_parsed(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        _MODELS
        + """
[store]
vector_backend = "chroma"

[rerank]
enabled = true
model = "ms-marco-MiniLM-L-12-v2"
top_k = 8

[cache]
enabled = true
threshold = 0.88

[eval]
judge_model = "ollama:llama3.1:8b"
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.store.vector_backend == "chroma"
    assert cfg.rerank.enabled is True
    assert cfg.rerank.top_k == 8
    assert cfg.cache.enabled is True
    assert cfg.cache.threshold == 0.88
    assert cfg.eval.judge_model == "ollama:llama3.1:8b"


def test_upgrade_sections_default_when_absent(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(_MODELS)
    cfg = load_config(cfg_file)
    # Absent sections must load with behavior-preserving defaults.
    assert cfg.store.vector_backend == "surreal"
    assert cfg.rerank.enabled is False
    assert cfg.rerank.model == "ms-marco-MiniLM-L-12-v2"
    assert cfg.rerank.top_k == 5
    assert cfg.cache.enabled is False
    assert cfg.cache.threshold == 0.95
    assert cfg.eval.judge_model == "ollama:qwen3:8b"
