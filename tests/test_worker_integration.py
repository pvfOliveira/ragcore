"""Live end-to-end: the worker's retry path against a real SurrealDB JobQueue.

Closes the seam between worker.py (fake-queue unit tests) and jobs.py (live queue
tests): a transient failure must drive a real mark_retry, leaving the job queued
with a bumped attempts count and a future next_attempt_at. Requires the `surreal`
binary (the `surreal_url` fixture skips if absent).
"""
import pytest

from ragcore import worker as worker_mod
from ragcore.config import ChunkingConfig, Config, ModelRole, SurrealConfig, WorkerConfig
from ragcore.jobs import JobQueue
from ragcore.store import Store
from ragcore.worker import run_worker


@pytest.fixture()
async def setup(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    store = Store(cfg)
    await store.init_schema()
    queue = JobQueue(cfg)
    config = Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="m"),
            "embedding": ModelRole(local_provider="ollama", local_model="e"),
        },
        chunking=ChunkingConfig(),
        surreal=cfg,
        worker=WorkerConfig(max_attempts=3, retry_base_seconds=2),
    )
    return config, queue, store


async def test_worker_transient_failure_requeues_in_live_db(setup, monkeypatch):
    config, queue, store = setup
    await queue.enqueue("/tmp/a.txt")

    async def boom(origin, store, config, chunk_size=400):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(worker_mod, "ingest_source", boom)

    await run_worker(config, once=True, queue=queue, store=store)

    # The transient failure requeued the job for retry: attempts bumped to 1, and a
    # future next_attempt_at means it is not immediately claimable again.
    queued = await queue.list_jobs(status="queued")
    assert len(queued) == 1 and queued[0]["attempts"] == 1
    assert await queue.claim_next() is None  # backing off, not yet due
