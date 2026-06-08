"""Graph-RAG: LLM triple extraction + SurrealDB entity/relation graph.

Opt-in via config.graph.enabled (default False).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Coroutine

from ai_prompter import Prompter
from surrealdb import AsyncSurreal, RecordID

from ragcore.config import SurrealConfig
from ragcore.errors import StoreError

logger = logging.getLogger(__name__)

_PROMPT_DIR = str(Path(__file__).parent / "prompts")
_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


# ---------------------------------------------------------------------------
# Extraction (parsing, no DB)
# ---------------------------------------------------------------------------

async def extract_triples(
    chunk: str,
    chat_fn: Callable[[str], Coroutine[Any, Any, str]],
) -> list[dict]:
    """Render the extraction prompt, call chat_fn, parse the result defensively.

    Returns a list of dicts with keys subject/predicate/object.
    Never raises — extraction is best-effort.
    """
    try:
        prompt = _render("graph_extract", {"chunk": chunk})
        text = await chat_fn(prompt)
        # Strip markdown code fences if present
        m = _CODE_FENCE.match(text.strip())
        if m:
            text = m.group(1)
        data = json.loads(text)
        triples = data.get("triples", []) if isinstance(data, dict) else []
        if not isinstance(triples, list):
            return []
        # Filter to well-formed triples only
        result = []
        for t in triples:
            if (
                isinstance(t, dict)
                and isinstance(t.get("subject"), str)
                and isinstance(t.get("predicate"), str)
                and isinstance(t.get("object"), str)
            ):
                result.append({"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]})
        return result
    except Exception as e:
        logger.debug("graph extraction parse failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Live chat_fn factory (wraps ask.py helpers)
# ---------------------------------------------------------------------------

def _chat_fn(config):
    """Return an async fn(prompt) -> str using the configured local model."""
    from ragcore.ask import _build_chat, _clean

    async def fn(prompt: str) -> str:
        # no content= : graph extraction always routes via the role default (local model)
        msg = await _build_chat(config).ainvoke(prompt)
        return _clean(msg.content)

    return fn


# ---------------------------------------------------------------------------
# GraphStore: SurrealDB entity + relates graph
# ---------------------------------------------------------------------------

class GraphStore:
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

    async def _upsert_entity(self, db, name: str, source_id: str) -> RecordID:
        """Return the RecordID of the entity with this norm, creating it if needed.

        Appends source_id to an existing entity's sources list.
        """
        norm = name.strip().lower()
        rows = self._rows(await db.query(
            "SELECT id FROM entity WHERE norm = $norm LIMIT 1;",
            {"norm": norm},
        ))
        if rows:
            rid = rows[0]["id"]
            # Append source_id to the sources array (read-modify-write; single-user tool)
            existing = self._rows(await db.query(
                "SELECT sources FROM $rid;", {"rid": rid}))
            sources = (existing[0].get("sources") or []) if existing else []
            if source_id not in sources:
                sources = sources + [source_id]
                await db.query(
                    "UPDATE $rid SET sources = $sources;",
                    {"rid": rid, "sources": sources},
                )
            return rid
        created = self._rows(await db.query(
            "CREATE entity SET name = $name, norm = $norm, sources = [$src];",
            {"name": name.strip(), "norm": norm, "src": source_id},
        ))
        return created[0]["id"]

    async def upsert_triples(self, source_id: str, triples: list[dict]) -> None:
        """Upsert entity nodes + RELATE edges for a list of triples.

        Raises StoreError on DB failure (so callers can log it), but individual
        bad triples are silently skipped.
        """
        db = self._connect()
        try:
            await self._signin(db)
            for triple in triples:
                try:
                    subj_name = triple.get("subject", "")
                    obj_name = triple.get("object", "")
                    predicate = triple.get("predicate", "")
                    if not (subj_name and obj_name and predicate):
                        continue
                    subj_id = await self._upsert_entity(db, subj_name, source_id)
                    obj_id = await self._upsert_entity(db, obj_name, source_id)
                    existing = self._rows(await db.query(
                        "SELECT id, sources FROM relates WHERE `in` = $i AND out = $o AND predicate = $p LIMIT 1;",
                        {"i": subj_id, "o": obj_id, "p": predicate},
                    ))
                    if existing:
                        edge = existing[0]
                        sources = edge.get("sources") or []
                        if source_id not in sources:
                            sources = sources + [source_id]
                            await db.query(
                                "UPDATE $eid SET sources = $s;",
                                {"eid": edge["id"], "s": sources},
                            )
                    else:
                        await db.query(
                            "RELATE $i->relates->$o SET predicate = $p, sources = $s;",
                            {"i": subj_id, "o": obj_id, "p": predicate, "s": [source_id]},
                        )
                except Exception:
                    continue  # skip individual bad triples
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"upsert_triples failed: {e}") from e
        finally:
            await db.close()

    async def neighbors(self, name: str, hops: int = 1) -> list[dict]:
        """Return entities reachable from the entity with this name via ->relates->entity.

        hops=1 returns direct neighbors. Used in Task 9 retrieval and tests.
        """
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
        norm = name.strip().lower()
        db = self._connect()
        try:
            await self._signin(db)
            # Find the source entity by norm
            rows = self._rows(await db.query(
                "SELECT id FROM entity WHERE norm = $norm LIMIT 1;",
                {"norm": norm},
            ))
            if not rows:
                return []
            src_id = rows[0]["id"]
            # Build traversal path: ->relates->entity (repeated hops times)
            traversal = "->relates->entity" * hops
            result = self._rows(await db.query(
                f"SELECT {traversal}.* AS neighbors FROM $src;",
                {"src": src_id},
            ))
            if not result:
                return []
            neighbors = result[0].get("neighbors") or []
            if not isinstance(neighbors, list):
                neighbors = [neighbors]
            out = []
            for n in neighbors:
                if isinstance(n, dict) and n.get("name"):
                    out.append({"name": n["name"], "norm": n.get("norm", ""), "id": str(n.get("id", ""))})
            return out
        except Exception as e:
            raise StoreError(f"neighbors failed: {e}") from e
        finally:
            await db.close()

    async def edge_count(self, subj_name: str, obj_name: str) -> int:
        """Return the number of relates edges between two entities (by name).

        Used in tests to assert deduplication.
        """
        subj_norm = subj_name.strip().lower()
        obj_norm = obj_name.strip().lower()
        db = self._connect()
        try:
            await self._signin(db)
            subj_rows = self._rows(await db.query(
                "SELECT id FROM entity WHERE norm = $norm LIMIT 1;",
                {"norm": subj_norm},
            ))
            obj_rows = self._rows(await db.query(
                "SELECT id FROM entity WHERE norm = $norm LIMIT 1;",
                {"norm": obj_norm},
            ))
            if not subj_rows or not obj_rows:
                return 0
            subj_id = subj_rows[0]["id"]
            obj_id = obj_rows[0]["id"]
            edges = self._rows(await db.query(
                "SELECT id FROM relates WHERE `in` = $i AND out = $o;",
                {"i": subj_id, "o": obj_id},
            ))
            return len(edges)
        except Exception as e:
            raise StoreError(f"edge_count failed: {e}") from e
        finally:
            await db.close()
