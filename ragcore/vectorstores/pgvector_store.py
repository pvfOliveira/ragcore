"""pgvector-backed VectorStore over a local Homebrew Postgres (no Docker).

Uses psycopg3 async + the pgvector type adapter. The vector index is HNSW with
``vector_cosine_ops``; ``vector_search`` returns cosine *similarity*
(``1 - (embedding <=> q)``) so higher = better, matching the protocol.

HONEST HOLDOUT NOTE — pgvectorscale: StreamingDiskANN ships as a *separate*
extension (pgvectorscale), not brewed, requiring a Rust/pgrx source build and
Linux-first support. The claim here is **pgvector** (owner decision
2026-06-11); pgvectorscale stays an open gap.
"""
from __future__ import annotations

import re
import uuid

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class PgVectorStore:
    """VectorStore over Postgres + pgvector (HNSW, cosine)."""

    def __init__(self, dsn: str, collection: str = "ragcore") -> None:
        if not _IDENT.fullmatch(collection):
            raise ValueError(
                f"invalid collection name {collection!r} (lowercase identifier required)"
            )
        self._dsn = dsn
        self._table = collection
        self._conn = None
        self._dim = None

    async def _get_conn(self):
        if self._conn is None:
            import psycopg  # lazy — optional extra
            from pgvector.psycopg import register_vector_async

            self._conn = await psycopg.AsyncConnection.connect(
                self._dsn, autocommit=True
            )
            await self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector_async(self._conn)
        return self._conn

    async def _ensure_table(self, dim: int) -> None:
        if self._dim is not None:
            return
        conn = await self._get_conn()
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} ("
            "id uuid PRIMARY KEY, chunk_id text UNIQUE, source_id text, "
            f"text text, embedding vector({dim}))"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw "
            f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
        )
        self._dim = dim

    async def add_embeddings(self, source_id: str, chunks: list[dict]) -> None:
        if not chunks:
            return
        await self._ensure_table(len(chunks[0]["embedding"]))
        conn = await self._get_conn()
        for c in chunks:
            await conn.execute(
                f"INSERT INTO {self._table} (id, chunk_id, source_id, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "text = EXCLUDED.text, embedding = EXCLUDED.embedding, "
                "source_id = EXCLUDED.source_id",
                (
                    str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
                    c["id"],
                    source_id,
                    c["text"],
                    self._vec(c["embedding"]),
                ),
            )

    async def vector_search(self, query: list[float], k: int = 10) -> list[dict]:
        await self._ensure_table(len(query))
        conn = await self._get_conn()
        qv = self._vec(query)
        cur = await conn.execute(
            f"SELECT chunk_id, text, 1 - (embedding <=> %s) AS score "
            f"FROM {self._table} ORDER BY embedding <=> %s LIMIT %s",
            (qv, qv, k),
        )
        rows = await cur.fetchall()
        return [{"id": r[0], "text": r[1], "score": float(r[2])} for r in rows]

    @staticmethod
    def _vec(values: list[float]):
        from pgvector import Vector  # lazy — optional extra

        return Vector(values)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
