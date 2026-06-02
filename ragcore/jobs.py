"""SurrealDB-backed ingestion job queue (own minimal queue; no surreal-commands).

Mirrors ragcore.store.Store: built from a SurrealConfig, connection-per-operation,
errors wrapped in StoreError.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError


@dataclass
class EnqueueResult:
    job_id: Optional[str]
    status: str  # "queued" | "exists" | "already_ingested"


class JobQueue:
    def __init__(self, config: SurrealConfig):
        self.config = config

    def _connect(self) -> AsyncSurreal:
        return AsyncSurreal(self.config.url)

    async def _signin(self, db) -> None:
        await db.signin({"username": self.config.user, "password": self.config.password})
        await db.use(self.config.namespace, self.config.database)

    @staticmethod
    def _rows(result: Any) -> list:
        return result if isinstance(result, list) else (result or [])

    async def enqueue(self, origin: str) -> EnqueueResult:
        # Single-user tool: the dedup SELECTs below carry a benign TOCTOU window vs a
        # concurrent enqueue of the same origin. Worst case is a redundant job that the
        # worker resolves to created=False (no duplicate data). Not guarded, by design.
        db = self._connect()
        try:
            await self._signin(db)
            existing = self._rows(await db.query(
                "SELECT id FROM source WHERE origin = $o LIMIT 1;", {"o": origin}))
            if existing:
                return EnqueueResult(job_id=None, status="already_ingested")
            pending = self._rows(await db.query(
                "SELECT id FROM ingestion_job WHERE origin = $o "
                "AND status IN ['queued', 'running'] LIMIT 1;", {"o": origin}))
            if pending:
                return EnqueueResult(job_id=str(pending[0]["id"]), status="exists")
            created = self._rows(await db.query(
                "CREATE ingestion_job SET origin = $o, status = 'queued';", {"o": origin}))
            return EnqueueResult(job_id=str(created[0]["id"]), status="queued")
        except Exception as e:
            raise StoreError(f"enqueue failed: {e}") from e
        finally:
            await db.close()

    async def claim_next(self) -> Optional[dict]:
        """Atomically move the oldest queued job to 'running' and return it.

        Two-step (SELECT oldest id, then conditional UPDATE with a status guard) so we
        do not rely on ORDER BY/LIMIT support inside UPDATE. A worker that loses the race
        sees the guard fail one of two ways: an empty result, or a SurrealDB transaction
        conflict — both mean "someone else claimed it", so we return None either way.
        """
        db = self._connect()
        try:
            await self._signin(db)
            cands = self._rows(await db.query(
                "SELECT id, created FROM ingestion_job WHERE status = 'queued' "
                "ORDER BY created ASC LIMIT 1;"))
            if not cands:
                return None
            rid = cands[0]["id"]
            try:
                claimed = self._rows(await db.query(
                    "UPDATE $rid SET status = 'running', updated = time::now() "
                    "WHERE status = 'queued' RETURN AFTER;", {"rid": rid}))
            except Exception as e:
                msg = str(e).lower()
                if "conflict" in msg or "transaction" in msg:
                    return None  # lost the race to a concurrent worker
                raise
            if not claimed:
                return None
            job = claimed[0]
            return {"id": str(job["id"]), "origin": job["origin"]}
        except Exception as e:
            raise StoreError(f"claim_next failed: {e}") from e
        finally:
            await db.close()

    async def mark_done(self, job_id: str, source_id: str) -> None:
        await self._set(job_id, {"status": "done", "source_id": source_id})

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self._set(job_id, {"status": "failed", "error": error})

    async def _set(self, job_id: str, fields: dict) -> None:
        db = self._connect()
        try:
            await self._signin(db)
            assignments = ", ".join(f"{k} = ${k}" for k in fields)
            await db.query(
                f"UPDATE $rid SET {assignments}, updated = time::now();",
                {"rid": RecordID.parse(job_id), **fields})
        except Exception as e:
            raise StoreError(f"job update failed: {e}") from e
        finally:
            await db.close()

    async def list_jobs(self, status: Optional[str] = None) -> list[dict]:
        db = self._connect()
        try:
            await self._signin(db)
            if status:
                rows = await db.query(
                    "SELECT id, origin, status, source_id, error, created "
                    "FROM ingestion_job WHERE status = $s ORDER BY created DESC, id DESC;",
                    {"s": status})
            else:
                rows = await db.query(
                    "SELECT id, origin, status, source_id, error, created "
                    "FROM ingestion_job ORDER BY created DESC, id DESC;")
            out = []
            for r in self._rows(rows):
                out.append({
                    "id": str(r["id"]),
                    "origin": r.get("origin"),
                    "status": r.get("status"),
                    "source_id": str(r["source_id"]) if r.get("source_id") else None,
                    "error": r.get("error"),
                    "created": str(r.get("created")),
                })
            return out
        except Exception as e:
            raise StoreError(f"list_jobs failed: {e}") from e
        finally:
            await db.close()
