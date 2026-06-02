"""Background worker: claim ingestion jobs and process them sequentially.

The worker owns no persistence logic — it asks the JobQueue for the next job,
reuses the synchronous ingest_source for the actual work, and records the outcome.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from ragcore.errors import classify_error, is_transient
from ragcore.ingest import ingest_source
from ragcore.jobs import JobQueue
from ragcore.store import Store


async def run_worker(config, *, once: bool = False, poll_interval: float = 2.0,
                     queue=None, store=None) -> None:
    queue = queue or JobQueue(config.surreal)
    store = store or Store(config.surreal)
    while True:
        job = await queue.claim_next()
        if job is None:
            if once:
                return
            await asyncio.sleep(poll_interval)
            continue
        try:
            result = await ingest_source(
                job["origin"], store, config, chunk_size=config.chunking.chunk_size)
            await queue.mark_done(job["id"], result.source_id)
            logger.info(f"job {job['id']} done -> {result.source_id} (created={result.created})")
        except Exception as e:  # noqa: BLE001 — one bad job must not stop the loop
            _, message = classify_error(e)
            attempt = job["attempts"] + 1
            if is_transient(e) and attempt < config.worker.max_attempts:
                delay = config.worker.retry_base_seconds * (2 ** (attempt - 1))
                await queue.mark_retry(job["id"], attempt, delay, message)
                logger.warning(
                    f"job {job['id']} attempt {attempt} failed (transient): {message}; "
                    f"retrying in {delay}s")
            else:
                await queue.mark_failed(job["id"], message, attempt)
                logger.warning(f"job {job['id']} failed after {attempt} attempt(s): {message}")
