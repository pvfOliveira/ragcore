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


async def grade_answer(question: str, answer: str, chunks: list[dict], chat_fn) -> bool:
    context = "\n".join(c.get("content", "") for c in chunks)
    verdict = await chat_fn(_render("agentic_grade_answer",
                                    {"question": question, "answer": answer, "context": context}))
    return _is_yes(verdict)


async def run_agentic(question: str, config, chat_fn, search_fn,
                      answer_fn=None) -> dict:
    """Drive the corrective loop. search_fn(term)->chunks; chat_fn(prompt)->str;
    answer_fn(question, chunks)->str (defaults to a simple LLM synthesis)."""
    from ragcore.query_rewrite import _parse_lines

    async def _default_answer(q, chunks):
        ctx = "\n".join(c.get("content", "") for c in chunks)
        return await chat_fn(f"Answer the question using only this context.\n\nContext:\n{ctx}\n\nQuestion: {q}")

    answer_fn = answer_fn or _default_answer
    term = question
    chunks = await search_fn(term)
    relevant = await grade_documents(question, chunks, chat_fn)
    iterations = 0
    while len(relevant) < config.min_relevant and iterations < config.max_iterations:
        iterations += 1
        rewrite = await chat_fn(
            f"Generate one alternative search query for: {question}\nReturn just the query.")
        term = (_parse_lines(rewrite) or [question])[0]
        more = await search_fn(term)
        chunks = chunks + more
        relevant = await grade_documents(question, chunks, chat_fn)
    use = relevant or chunks
    answer = await answer_fn(question, use)
    grounded = await grade_answer(question, answer, use, chat_fn)
    return {"answer": answer, "citations": sorted({c.get("source", c["id"]) for c in use}),
            "grounded": grounded, "iterations": iterations}
