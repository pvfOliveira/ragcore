import asyncio

import pytest

from ragcore.config import SurrealConfig
from ragcore.jobs import JobQueue
from ragcore.store import Store


@pytest.fixture()
async def setup(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    store = Store(cfg)
    await store.init_schema()  # also creates ingestion_job
    queue = JobQueue(cfg)
    return store, queue


async def test_enqueue_creates_queued_job(setup):
    store, queue = setup
    res = await queue.enqueue("/tmp/a.txt")
    assert res.status == "queued"
    assert res.job_id and res.job_id.startswith("ingestion_job:")
    jobs = await queue.list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["origin"] == "/tmp/a.txt"
    assert jobs[0]["status"] == "queued"


async def test_enqueue_skips_duplicate_pending_job(setup):
    store, queue = setup
    first = await queue.enqueue("/tmp/a.txt")
    second = await queue.enqueue("/tmp/a.txt")
    assert second.status == "exists"
    assert second.job_id == first.job_id
    assert len(await queue.list_jobs()) == 1


async def test_enqueue_skips_already_ingested_origin(setup):
    store, queue = setup
    await store.create_source(title="T", full_text="x", origin="/tmp/done.txt")
    res = await queue.enqueue("/tmp/done.txt")
    assert res.status == "already_ingested"
    assert res.job_id is None
    assert await queue.list_jobs() == []


async def test_claim_next_transitions_and_empties(setup):
    store, queue = setup
    await queue.enqueue("/tmp/a.txt")
    claimed = await queue.claim_next()
    assert claimed is not None
    assert claimed["origin"] == "/tmp/a.txt"
    running = await queue.list_jobs(status="running")
    assert len(running) == 1 and running[0]["id"] == claimed["id"]
    assert await queue.claim_next() is None


async def test_mark_done_and_failed(setup):
    store, queue = setup
    j1 = (await queue.enqueue("/tmp/a.txt")).job_id
    await queue.claim_next()
    await queue.mark_done(j1, "source:abc")
    done = await queue.list_jobs(status="done")
    assert len(done) == 1 and done[0]["source_id"] == "source:abc"

    j2 = (await queue.enqueue("/tmp/b.txt")).job_id
    await queue.claim_next()
    await queue.mark_failed(j2, "boom")
    failed = await queue.list_jobs(status="failed")
    assert len(failed) == 1 and failed[0]["error"] == "boom"


async def test_claim_next_is_race_safe(setup):
    # The increment's central invariant: two workers claiming concurrently must yield
    # exactly one claim (the guarded UPDATE makes the race loser return None, not steal).
    store, queue = setup
    await queue.enqueue("/tmp/a.txt")
    a, b = await asyncio.gather(queue.claim_next(), queue.claim_next())
    claimed = [r for r in (a, b) if r is not None]
    assert len(claimed) == 1
    assert claimed[0]["origin"] == "/tmp/a.txt"
    assert await queue.claim_next() is None


async def test_claim_next_skips_not_yet_due(setup):
    store, queue = setup
    jid = (await queue.enqueue("/tmp/a.txt")).job_id
    await queue.claim_next()  # -> running
    await queue.mark_retry(jid, attempts=1, delay_seconds=300, error="boom")  # due in 5 min
    assert await queue.claim_next() is None
    q = await queue.list_jobs(status="queued")
    assert len(q) == 1 and q[0]["attempts"] == 1


async def test_claim_next_returns_due_retry_with_attempts(setup):
    store, queue = setup
    jid = (await queue.enqueue("/tmp/a.txt")).job_id
    await queue.claim_next()
    await queue.mark_retry(jid, attempts=1, delay_seconds=0, error="boom")  # due now
    claimed = await queue.claim_next()
    assert claimed is not None and claimed["attempts"] == 1


async def test_mark_failed_stores_attempts(setup):
    store, queue = setup
    jid = (await queue.enqueue("/tmp/a.txt")).job_id
    await queue.claim_next()
    await queue.mark_failed(jid, "permanent", attempts=2)
    failed = await queue.list_jobs(status="failed")
    assert failed[0]["attempts"] == 2 and failed[0]["error"] == "permanent"


async def test_requeue_resets_failed_job(setup):
    store, queue = setup
    jid = (await queue.enqueue("/tmp/a.txt")).job_id
    await queue.claim_next()
    await queue.mark_failed(jid, "boom", attempts=3)
    assert await queue.requeue(jid) is True
    q = await queue.list_jobs(status="queued")
    assert len(q) == 1 and q[0]["attempts"] == 0
    assert await queue.requeue(jid) is False           # already queued, not failed
    assert await queue.requeue("ingestion_job:nope") is False
    assert await queue.requeue("not-an-id") is False   # malformed -> False, no raise
