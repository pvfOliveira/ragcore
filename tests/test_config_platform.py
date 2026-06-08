from ragcore.config import load_config


def test_platform_sections_parse(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[models.chat]\nlocal_provider="ollama"\nlocal_model="qwen2.5:7b-instruct"\n'
        '[models.embedding]\nlocal_provider="ollama"\nlocal_model="nomic-embed-text"\n'
        '[llmops]\ntolerance=0.1\ndrift_threshold=0.2\n'
        '[cost]\nenabled=true\nenforce=true\nbudget_tokens=4000\n'
        '[cost.rates]\n"anthropic:claude"=3.0\n'
        '[bench]\nconcurrency=[1,2]\n'
        '[graph]\nenabled=true\nhops=2\n'
        '[multimodal]\nmodel="ViT-B-32"\npretrained="laion2b_s34b_b79k"\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.llmops.tolerance == 0.1
    assert cfg.llmops.drift_threshold == 0.2
    assert cfg.cost.enabled
    assert cfg.cost.enforce
    assert cfg.cost.budget_tokens == 4000
    assert cfg.cost.rates["anthropic:claude"] == 3.0
    assert cfg.bench.concurrency == [1, 2]
    assert cfg.graph.enabled
    assert cfg.graph.hops == 2
    assert cfg.multimodal.model == "ViT-B-32"


def test_platform_sections_default(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[models.chat]\nlocal_provider="ollama"\nlocal_model="qwen2.5:7b-instruct"\n'
        '[models.embedding]\nlocal_provider="ollama"\nlocal_model="nomic-embed-text"\n'
    )
    cfg = load_config(cfg_file)
    assert cfg.cost.enabled is False
    assert cfg.cost.enforce is False
    assert cfg.graph.enabled is False
    assert cfg.llmops.tolerance == 0.05
    assert cfg.bench.num_ctx == [2048, 4096]
    assert cfg.bench.concurrency == [1, 2, 4]
    assert cfg.multimodal.device == "mps"
    assert cfg.multimodal.model == "ViT-B-32"
