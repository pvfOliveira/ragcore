from __future__ import annotations

import json
from typing import Any, Optional

from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError


class RunRegistry:
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

    async def record(self, *, metrics: dict, config_snapshot: dict,
                     dataset: str | None = None, tag: str | None = None,
                     gate_passed: bool | None = None) -> str:
        db = self._connect()
        try:
            await self._signin(db)
            created = self._rows(await db.query(
                "CREATE eval_run SET tag=$tag, dataset=$ds, config_snapshot=$cfg, "
                "metrics=$m, gate_passed=$gp;",
                {"tag": tag, "ds": dataset, "cfg": json.dumps(config_snapshot),
                 "m": json.dumps(metrics), "gp": gate_passed}))
            return str(created[0]["id"])
        except Exception as e:
            raise StoreError(f"record run failed: {e}") from e
        finally:
            await db.close()

    def _parse(self, r: dict) -> dict:
        return {"id": str(r["id"]), "tag": r.get("tag"), "dataset": r.get("dataset"),
                "metrics": json.loads(r["metrics"]) if r.get("metrics") else {},
                "config_snapshot": json.loads(r["config_snapshot"]) if r.get("config_snapshot") else {},
                "gate_passed": r.get("gate_passed"), "created": str(r.get("created"))}

    async def get(self, run_id: str) -> Optional[dict]:
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query("SELECT * FROM $rid;",
                                             {"rid": RecordID.parse(run_id)}))
            return self._parse(rows[0]) if rows else None
        except Exception as e:
            raise StoreError(f"get run failed: {e}") from e
        finally:
            await db.close()

    async def list_runs(self) -> list[dict]:
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "SELECT * FROM eval_run ORDER BY created DESC, id DESC;"))
            return [self._parse(r) for r in rows]
        except Exception as e:
            raise StoreError(f"list runs failed: {e}") from e
        finally:
            await db.close()

    async def resolve(self, ref: str) -> Optional[dict]:
        """A run id (record link) -> that run; otherwise treat ref as a tag -> latest."""
        if ":" in ref:
            got = await self.get(ref)
            if got:
                return got
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "SELECT * FROM eval_run WHERE tag = $t ORDER BY created DESC, id DESC LIMIT 1;",
                {"t": ref}))
            return self._parse(rows[0]) if rows else None
        except Exception as e:
            raise StoreError(f"resolve ref failed: {e}") from e
        finally:
            await db.close()
