import pytest

from ragcore import worker as worker_mod
from ragcore.config import ChunkingConfig, Config, ModelRole, SurrealConfig
from ragcore.ingest import IngestResult
from ragcore.worker import run_worker


def _config():
    return Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="m"),
            "embedding": ModelRole(local_provider="ollama", local_model="e"),
        },
        chunking=ChunkingConfig(),
        surreal=SurrealConfig(),
    )


class FakeQueue:
    def __init__(self, jobs):
        self._jobs = list(jobs)
        self.done = []
        self.failed = []

    async def claim_next(self):
        return self._jobs.pop(0) if self._jobs else None

    async def mark_done(self, job_id, source_id):
        self.done.append((job_id, source_id))

    async def mark_failed(self, job_id, error):
        self.failed.append((job_id, error))


async def test_worker_processes_job_to_done(monkeypatch):
    async def fake_ingest(origin, store, config, chunk_size=400):
        return IngestResult(source_id="source:1", created=True)
    monkeypatch.setattr(worker_mod, "ingest_source", fake_ingest)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt"}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.done == [("ingestion_job:1", "source:1")]
    assert queue.failed == []


async def test_worker_marks_failed_on_error(monkeypatch):
    async def boom(origin, store, config, chunk_size=400):
        raise ValueError("No content extracted from /tmp/a.txt")
    monkeypatch.setattr(worker_mod, "ingest_source", boom)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt"}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.done == []
    assert len(queue.failed) == 1
    job_id, message = queue.failed[0]
    assert job_id == "ingestion_job:1"
    assert "No content extracted" in message


async def test_worker_once_returns_on_empty_queue():
    queue = FakeQueue([])
    await run_worker(_config(), once=True, queue=queue, store=object())
    assert queue.done == [] and queue.failed == []
