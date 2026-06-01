from typer.testing import CliRunner

from ragcore import cli as cli_mod

runner = CliRunner()


def test_models_command(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[models.chat]\nlocal_provider="ollama"\nlocal_model="qwen3:8b"\n'
        '[models.embedding]\nlocal_provider="ollama"\nlocal_model="nomic-embed-text"\n'
    )
    result = runner.invoke(cli_mod.app, ["--config", str(cfg_file), "models"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.stdout
    assert "nomic-embed-text" in result.stdout
