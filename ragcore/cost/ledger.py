"""SQLite usage ledger for per-request token accounting (opt-in cost layer)."""
from __future__ import annotations

import sqlite3


class CostLedger:
    """Persistent usage ledger backed by stdlib SQLite."""

    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS usage("
            "role TEXT, provider TEXT, model TEXT, "
            "prompt_tokens INT, completion_tokens INT, cached INT)"
        )
        self._conn.commit()

    def record(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
    ) -> None:
        self._conn.execute(
            "INSERT INTO usage(role, provider, model, prompt_tokens, completion_tokens, cached) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (role, provider, model, prompt_tokens, completion_tokens, int(cached)),
        )
        self._conn.commit()

    def aggregate(self) -> dict:
        cursor = self._conn.execute(
            "SELECT COUNT(*), SUM(prompt_tokens), SUM(completion_tokens), "
            "SUM(cached), SUM(1 - cached) FROM usage"
        )
        row = cursor.fetchone()
        total_requests = row[0] or 0
        total_prompt = row[1] or 0
        total_completion = row[2] or 0
        hits = row[3] or 0
        misses = row[4] or 0
        hit_rate = hits / total_requests if total_requests > 0 else 0.0

        by_model_cursor = self._conn.execute(
            "SELECT provider, model, role, "
            "SUM(prompt_tokens), SUM(completion_tokens), COUNT(*) "
            "FROM usage GROUP BY provider, model, role"
        )
        by_model = [
            {
                "provider": r[0],
                "model": r[1],
                "role": r[2],
                "prompt_tokens": r[3] or 0,
                "completion_tokens": r[4] or 0,
                "requests": r[5],
            }
            for r in by_model_cursor
        ]

        return {
            "totals": {
                "requests": total_requests,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
            },
            "cache": {
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
            },
            "by_model": by_model,
        }


def over_budget(prompt_tokens: int, budget_tokens: int) -> bool:
    """Return True if prompt_tokens exceeds budget_tokens (budget_tokens=0 means no budget)."""
    return budget_tokens > 0 and prompt_tokens > budget_tokens
