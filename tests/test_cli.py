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


def test_list_command(monkeypatch):
    class FakeStore:
        async def list_sources(self):
            return [{"id": "source:1", "title": "Doc One",
                     "origin": "/tmp/a.txt", "created": "2026-06-02", "chunks": 3}]
    monkeypatch.setattr(cli_mod, "_load", lambda: (None, FakeStore()))
    result = runner.invoke(cli_mod.app, ["list"])
    assert result.exit_code == 0
    assert "source:1" in result.stdout
    assert "Doc One" in result.stdout


def test_list_command_empty(monkeypatch):
    class FakeStore:
        async def list_sources(self):
            return []
    monkeypatch.setattr(cli_mod, "_load", lambda: (None, FakeStore()))
    result = runner.invoke(cli_mod.app, ["list"])
    assert result.exit_code == 0
    assert "No sources ingested" in result.stdout


def test_remove_command(monkeypatch):
    class FakeStore:
        async def delete_source(self, source_id):
            return True
    monkeypatch.setattr(cli_mod, "_load", lambda: (None, FakeStore()))
    result = runner.invoke(cli_mod.app, ["remove", "source:1"])
    assert result.exit_code == 0
    assert "Removed source:1" in result.stdout


def test_remove_command_missing(monkeypatch):
    class FakeStore:
        async def delete_source(self, source_id):
            return False
    monkeypatch.setattr(cli_mod, "_load", lambda: (None, FakeStore()))
    result = runner.invoke(cli_mod.app, ["remove", "source:nope"])
    assert result.exit_code == 1
    assert "No such source" in result.stdout


def test_ingest_async_enqueues(monkeypatch):
    from ragcore.jobs import EnqueueResult

    class FakeQueue:
        def __init__(self, surreal):
            pass

        async def enqueue(self, origin):
            return EnqueueResult(job_id="ingestion_job:1", status="queued")

    class Cfg:
        surreal = None

    monkeypatch.setattr("ragcore.jobs.JobQueue", FakeQueue)
    monkeypatch.setattr(cli_mod, "_load", lambda: (Cfg(), None))
    result = runner.invoke(cli_mod.app, ["ingest", "doc.txt", "--async"])
    assert result.exit_code == 0
    assert "Queued doc.txt as job ingestion_job:1" in result.stdout


def test_jobs_command_lists(monkeypatch):
    class FakeQueue:
        def __init__(self, surreal):
            pass

        async def list_jobs(self, status=None):
            return [{"id": "ingestion_job:1", "origin": "/tmp/a.txt",
                     "status": "done", "source_id": "source:1",
                     "error": None, "created": "2026-06-02"}]

    class Cfg:
        surreal = None

    monkeypatch.setattr("ragcore.jobs.JobQueue", FakeQueue)
    monkeypatch.setattr(cli_mod, "_load", lambda: (Cfg(), None))
    result = runner.invoke(cli_mod.app, ["jobs"])
    assert result.exit_code == 0
    assert "ingestion_job:1" in result.stdout
    assert "done" in result.stdout


def test_worker_once_runs(monkeypatch):
    calls = {}

    async def fake_run_worker(config, *, once=False, poll_interval=2.0, queue=None, store=None):
        calls["once"] = once

    class Cfg:
        surreal = None

    monkeypatch.setattr("ragcore.worker.run_worker", fake_run_worker)
    monkeypatch.setattr(cli_mod, "_load", lambda: (Cfg(), None))
    result = runner.invoke(cli_mod.app, ["worker", "--once"])
    assert result.exit_code == 0
    assert calls.get("once") is True
