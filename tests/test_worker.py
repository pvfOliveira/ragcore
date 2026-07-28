
from ragcore import worker as worker_mod
from ragcore.config import (
    ChunkingConfig,
    Config,
    ModelRole,
    SurrealConfig,
    WorkerConfig,
)
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
        worker=WorkerConfig(max_attempts=3, retry_base_seconds=2),
    )


class FakeQueue:
    def __init__(self, jobs):
        self._jobs = list(jobs)
        self.done = []
        self.failed = []
        self.retried = []

    async def claim_next(self):
        return self._jobs.pop(0) if self._jobs else None

    async def mark_done(self, job_id, source_id):
        self.done.append((job_id, source_id))

    async def mark_failed(self, job_id, error, attempts=None):
        self.failed.append((job_id, error, attempts))

    async def mark_retry(self, job_id, attempts, delay_seconds, error):
        self.retried.append((job_id, attempts, delay_seconds, error))


async def test_worker_processes_job_to_done(monkeypatch):
    async def fake_ingest(origin, store, config, chunk_size=400):
        return IngestResult(source_id="source:1", created=True)
    monkeypatch.setattr(worker_mod, "ingest_source", fake_ingest)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt", "attempts": 0}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.done == [("ingestion_job:1", "source:1")]
    assert queue.failed == [] and queue.retried == []


async def test_worker_transient_failure_retries(monkeypatch):
    async def boom(origin, store, config, chunk_size=400):
        raise ConnectionError("Connection refused")
    monkeypatch.setattr(worker_mod, "ingest_source", boom)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt", "attempts": 0}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.failed == []
    assert len(queue.retried) == 1
    job_id, attempts, delay, message = queue.retried[0]
    assert job_id == "ingestion_job:1"
    assert attempts == 1
    assert delay == 2          # base 2 * 2**(1-1)
    assert "provider" in message.lower()


async def test_worker_transient_at_cap_marks_failed(monkeypatch):
    async def boom(origin, store, config, chunk_size=400):
        raise ConnectionError("Connection refused")
    monkeypatch.setattr(worker_mod, "ingest_source", boom)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt", "attempts": 2}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.retried == []
    assert len(queue.failed) == 1
    assert queue.failed[0][0] == "ingestion_job:1" and queue.failed[0][2] == 3


async def test_worker_permanent_failure_marks_failed_immediately(monkeypatch):
    async def boom(origin, store, config, chunk_size=400):
        raise ValueError("No content extracted from /tmp/a.txt")
    monkeypatch.setattr(worker_mod, "ingest_source", boom)

    queue = FakeQueue([{"id": "ingestion_job:1", "origin": "/tmp/a.txt", "attempts": 0}])
    await run_worker(_config(), once=True, queue=queue, store=object())

    assert queue.retried == []
    assert len(queue.failed) == 1
    assert queue.failed[0][2] == 1


async def test_worker_once_returns_on_empty_queue():
    queue = FakeQueue([])
    await run_worker(_config(), once=True, queue=queue, store=object())
    assert queue.done == [] and queue.failed == [] and queue.retried == []
