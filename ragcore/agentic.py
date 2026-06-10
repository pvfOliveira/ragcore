"""Corrective / self-RAG: grade retrieved docs, re-retrieve on weak context,
grade the final answer for groundedness. Opt-in alternate ask path (default-OFF)."""
from __future__ import annotations

from pathlib import Path

from ai_prompter import Prompter

_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


def _is_yes(text: str) -> bool:
    return text.strip().lower().startswith("y")


async def grade_documents(question: str, chunks: list[dict], chat_fn) -> list[dict]:
    """Return only chunks the grader marks relevant to *question*."""
    kept = []
    for c in chunks:
        verdict = await chat_fn(_render("agentic_grade_docs",
                                        {"question": question, "content": c.get("content", "")}))
        if _is_yes(verdict):
            kept.append(c)
    return kept
