"""SurrealDB-backed chat sessions + messages (mirrors Store/JobQueue)."""
from __future__ import annotations

from typing import Any, Optional

from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError


class SessionStore:
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

    async def create_session(self, title: Optional[str] = None) -> str:
        db = self._connect()
        try:
            await self._signin(db)
            created = self._rows(await db.query(
                "CREATE chat_session SET title = $title;", {"title": title}))
            return str(created[0]["id"])
        except Exception as e:
            raise StoreError(f"create_session failed: {e}") from e
        finally:
            await db.close()

    async def add_message(self, session_id: str, role: str, content: str,
                          citations: Optional[list[str]] = None) -> None:
        db = self._connect()
        try:
            await self._signin(db)
            sid = RecordID.parse(session_id)
            # seq = current message count. Assumes a single writer per session (the
            # interactive REPL); concurrent writers could collide on seq (accepted).
            existing = self._rows(await db.query(
                "SELECT id FROM chat_message WHERE session = $sid;", {"sid": sid}))
            seq = len(existing)
            await db.query(
                "CREATE chat_message SET session = $sid, seq = $seq, role = $role, "
                "content = $content, citations = $citations;",
                {"sid": sid, "seq": seq, "role": role, "content": content,
                 "citations": citations})
        except Exception as e:
            raise StoreError(f"add_message failed: {e}") from e
        finally:
            await db.close()

    async def get_history(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        db = self._connect()
        try:
            await self._signin(db)
            rows = self._rows(await db.query(
                "SELECT seq, role, content, citations, created FROM chat_message "
                "WHERE session = $sid ORDER BY seq ASC;",
                {"sid": RecordID.parse(session_id)}))
            history = [{"role": r.get("role"), "content": r.get("content"),
                        "citations": r.get("citations"), "created": str(r.get("created"))}
                       for r in rows]
            if limit is not None:
                history = history[-limit:] if limit > 0 else []
            return history
        except Exception as e:
            raise StoreError(f"get_history failed: {e}") from e
        finally:
            await db.close()

    async def list_sessions(self) -> list[dict]:
        db = self._connect()
        try:
            await self._signin(db)
            sessions = self._rows(await db.query(
                "SELECT id, title, created FROM chat_session ORDER BY created DESC, id DESC;"))
            counts = self._rows(await db.query(
                "SELECT session, count() AS n FROM chat_message GROUP BY session;"))
            count_map = {str(c["session"]): c["n"] for c in counts}
            return [{"id": str(s["id"]), "title": s.get("title"),
                     "created": str(s.get("created")),
                     "messages": count_map.get(str(s["id"]), 0)} for s in sessions]
        except Exception as e:
            raise StoreError(f"list_sessions failed: {e}") from e
        finally:
            await db.close()

    async def rename_session(self, session_id: str, title: str) -> bool:
        try:
            rid = RecordID.parse(session_id)
        except Exception:
            return False
        db = self._connect()
        try:
            await self._signin(db)
            existing = self._rows(await db.query(
                "SELECT id FROM chat_session WHERE id = $rid;", {"rid": rid}))
            if not existing:
                return False
            await db.query(
                "UPDATE $rid SET title = $title, updated = time::now();",
                {"rid": rid, "title": title})
            return True
        except Exception as e:
            raise StoreError(f"rename_session failed: {e}") from e
        finally:
            await db.close()

    async def delete_session(self, session_id: str) -> bool:
        try:
            rid = RecordID.parse(session_id)
        except Exception:
            return False
        db = self._connect()
        try:
            await self._signin(db)
            deleted = self._rows(await db.query(
                "DELETE $rid RETURN BEFORE;", {"rid": rid}))
            return bool(deleted)
        except Exception as e:
            raise StoreError(f"delete_session failed: {e}") from e
        finally:
            await db.close()
