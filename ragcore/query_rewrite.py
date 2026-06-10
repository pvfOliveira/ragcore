"""Query rewriting: expand one query into search variants (opt-in, default-OFF).

Strategies: multi_query (paraphrases), hyde (hypothetical answer doc),
decompose (sub-questions). chat_fn(prompt:str) -> str is any async LLM callable.
"""
from __future__ import annotations

import re
from pathlib import Path

from ai_prompter import Prompter

_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


def _parse_lines(text: str) -> list[str]:
    """Parse an LLM list (numbered or bulleted) into clean lines."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", raw).strip()
        if line:
            out.append(line)
    return out


async def rewrite_query(query: str, config, chat_fn) -> list[str]:
    cfg = config.query_rewrite
    if not cfg.enabled:
        return [query]
    if cfg.strategy == "multi_query":
        text = await chat_fn(_render("query_multi", {"query": query, "n": cfg.n}))
        return _parse_lines(text)[: cfg.n] or [query]
    raise ValueError(f"Unknown query_rewrite strategy {cfg.strategy!r}")
