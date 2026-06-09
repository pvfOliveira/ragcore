import json


def test_load_compiled_strategy_returns_none_when_absent(tmp_path):
    from ragcore.dspy_optimizer import load_compiled_strategy
    assert load_compiled_strategy(str(tmp_path / "missing.json")) is None


def test_load_compiled_strategy_reads_prompt(tmp_path):
    from ragcore.dspy_optimizer import load_compiled_strategy
    art = tmp_path / "dspy_compiled.json"
    art.write_text(json.dumps({"strategy_prompt": "OPTIMIZED: list searches for {{question}}"}))
    assert load_compiled_strategy(str(art)) == "OPTIMIZED: list searches for {{question}}"


def test_render_strategy_prefers_compiled_when_enabled(tmp_path):
    """When [dspy] enabled and an artifact exists, _render_strategy uses the
    compiled prompt; otherwise it falls back to the Jinja template."""
    import ragcore.ask as ask
    from ragcore.config import Config, DspyConfig, ModelRole

    art = tmp_path / "dspy_compiled.json"
    art.write_text(json.dumps({"strategy_prompt": "COMPILED for: {{question}}"}))
    cfg_on = Config(
        models={"chat": ModelRole(local_provider="ollama", local_model="m"),
                "embedding": ModelRole(local_provider="ollama", local_model="e")},
        dspy=DspyConfig(enabled=True, compiled_path=str(art)))
    rendered = ask._render_strategy(cfg_on, {"question": "what is q?", "max_searches": 5})
    assert "COMPILED" in rendered
    assert "what is q?" in rendered  # the {{question}} was rendered

    cfg_off = Config(
        models={"chat": ModelRole(local_provider="ollama", local_model="m"),
                "embedding": ModelRole(local_provider="ollama", local_model="e")},
        dspy=DspyConfig(enabled=False, compiled_path=str(art)))
    rendered_off = ask._render_strategy(cfg_off, {"question": "what is q?", "max_searches": 5})
    assert "COMPILED" not in rendered_off  # Jinja template instead


def test_save_and_load_roundtrip(tmp_path):
    from ragcore.dspy_optimizer import load_compiled_strategy, save_compiled_strategy
    p = str(tmp_path / "sub" / "dspy_compiled.json")
    save_compiled_strategy(p, "PROMPT {{question}}")
    assert load_compiled_strategy(p) == "PROMPT {{question}}"
