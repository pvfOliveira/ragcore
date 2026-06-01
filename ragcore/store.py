"""SurrealDB repository: connection per operation + schema init + search.

Connection-per-operation mirrors open-notebook (adequate for a single-user CLI;
a pool is the documented seam if throughput ever matters).

Targets the surrealdb 2.0 Python SDK against a SurrealDB 3.x server:
  - ``AsyncSurreal(url)`` is a factory returning a connection object.
  - ``conn.query(...)`` returns the raw result directly (e.g. a list of row
    dicts for SELECT / CREATE / ``RETURN fn::...``), not a 1.x status envelope.
  - record-link fields typed ``record<source>`` will not coerce from a bare
    string, so a ``RecordID`` must be passed for the link.
  - ``id`` / ``source`` come back as ``RecordID`` objects; we stringify them.
"""
from __future__ import annotations

from importlib import resources
from typing import Any

from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError


class Store:
    def __init__(self, config: SurrealConfig):
        self.config = config

    def _connect(self):
        return AsyncSurreal(self.config.url)

    async def _signin(self, db) -> None:
        await db.signin({"username": self.config.user, "password": self.config.password})
        await db.use(self.config.namespace, self.config.database)

    async def init_schema(self) -> None:
        schema = resources.files("ragcore.db").joinpath("schema.surql").read_text()
        db = self._connect()
        try:
            await self._signin(db)
            await db.query(schema)
        except Exception as e:
            raise StoreError(f"Schema init failed: {e}") from e
        finally:
            await db.close()

    async def create_source(self, title: str, full_text: str, origin: str) -> str:
        db = self._connect()
        try:
            await self._signin(db)
            rows = await db.query(
                "CREATE source SET title=$title, full_text=$ft, origin=$origin;",
                {"title": title, "ft": full_text, "origin": origin},
            )
            record = rows[0] if isinstance(rows, list) else rows
            return str(record["id"])
        except Exception as e:
            raise StoreError(f"create_source failed: {e}") from e
        finally:
            await db.close()

    async def add_embeddings(self, source_id: str, chunks: list[dict[str, Any]]) -> None:
        # `source` is typed record<source>; a bare string will not coerce, so
        # parse the "source:xxxx" string into a RecordID for the link.
        src = RecordID.parse(source_id)
        db = self._connect()
        try:
            await self._signin(db)
            for ch in chunks:
                await db.query(
                    "CREATE source_embedding SET source=$src, `order`=$order, "
                    "content=$content, embedding=$embedding;",
                    {"src": src, "order": ch["order"],
                     "content": ch["content"], "embedding": ch["embedding"]},
                )
        except Exception as e:
            raise StoreError(f"add_embeddings failed: {e}") from e
        finally:
            await db.close()

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict[str, Any]]:
        return await self._call_fn("fn::vector_search($q, $k)", {"q": query, "k": k})

    async def text_search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        return await self._call_fn("fn::text_search($q, $k)", {"q": query, "k": k})

    async def _call_fn(self, call: str, vars: dict[str, Any]) -> list[dict[str, Any]]:
        db = self._connect()
        try:
            await self._signin(db)
            rows = await db.query(f"RETURN {call};", vars)
            # `RETURN fn::...()` yields the function's array directly in SDK 2.0.
            result = rows if isinstance(rows, list) else (rows or [])
            return [{**r, "id": str(r["id"]), "source": str(r["source"])} for r in result]
        except Exception as e:
            raise StoreError(f"search failed: {e}") from e
        finally:
            await db.close()
