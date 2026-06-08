from __future__ import annotations

from typing import Any, Optional

from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import RagcoreError, StoreError

_PTR = "deployment:current"


class DeploymentStore:
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

    async def _row(self, db) -> Optional[dict]:
        rows = self._rows(await db.query(f"SELECT * FROM {_PTR};"))
        return rows[0] if rows else None

    async def current(self) -> Optional[str]:
        db = self._connect()
        try:
            await self._signin(db)
            row = await self._row(db)
            return row.get("run_id") if row else None
        except Exception as e:
            raise StoreError(f"current failed: {e}") from e
        finally:
            await db.close()

    async def promote(self, run_id: str, require_gate: bool = False) -> None:
        db = self._connect()
        try:
            await self._signin(db)
            if require_gate:
                rr = self._rows(await db.query(
                    "SELECT gate_passed FROM $rid;", {"rid": RecordID.parse(run_id)}))
                if not rr or rr[0].get("gate_passed") is not True:
                    raise RagcoreError(f"run {run_id} did not pass its gate; refusing promote")
            row = await self._row(db)
            history = (row.get("history") or []) if row else []
            if row and row.get("run_id"):
                history = history + [row["run_id"]]
            # Use CREATE then UPDATE pattern (existence-check + upsert via DELETE+CREATE)
            # to handle the singleton record reliably on SurrealDB 3.1.2.
            existing = self._rows(await db.query(f"SELECT id FROM {_PTR};"))
            if existing:
                await db.query(
                    f"UPDATE {_PTR} SET run_id=$rid, history=$h;",
                    {"rid": run_id, "h": history})
            else:
                await db.query(
                    f"CREATE {_PTR} SET run_id=$rid, history=$h;",
                    {"rid": run_id, "h": history})
        except RagcoreError:
            raise
        except Exception as e:
            raise StoreError(f"promote failed: {e}") from e
        finally:
            await db.close()

    async def rollback(self) -> Optional[str]:
        db = self._connect()
        try:
            await self._signin(db)
            row = await self._row(db)
            if not row or not row.get("history"):
                return None
            history = list(row["history"])
            prev = history.pop()
            existing = self._rows(await db.query(f"SELECT id FROM {_PTR};"))
            if existing:
                await db.query(
                    f"UPDATE {_PTR} SET run_id=$rid, history=$h;",
                    {"rid": prev, "h": history})
            else:
                await db.query(
                    f"CREATE {_PTR} SET run_id=$rid, history=$h;",
                    {"rid": prev, "h": history})
            return prev
        except Exception as e:
            raise StoreError(f"rollback failed: {e}") from e
        finally:
            await db.close()
