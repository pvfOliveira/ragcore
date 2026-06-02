"""Multi-turn chat over the corpus: history-aware RAG (plain async, no LangGraph)."""
from __future__ import annotations

from pathlib import Path

from ai_prompter import Prompter

from ragcore.ask import _build_chat, _clean  # reuse model provisioning + thinking-strip
from ragcore.retrieve import hybrid_search

_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


async def chat_turn(session_store, store, config, session_id, message, embedder_fn) -> dict:
    history = await session_store.get_history(session_id, limit=config.chat.history_window)

    if history:
        chat = _build_chat(config, content=message)
        rmsg = await chat.ainvoke(_render("chat_reformulate", {"history": history, "message": message}))
        query = _clean(rmsg.content) or message
    else:
        query = message

    chunks = await hybrid_search(store, embedder_fn, query, k=10)
    chat = _build_chat(config, content=message)
    amsg = await chat.ainvoke(_render("chat_answer",
                                      {"history": history, "chunks": chunks, "message": message}))
    answer = _clean(amsg.content)
    citations = sorted({c["source"] for c in chunks})

    await session_store.add_message(session_id, "user", message)
    await session_store.add_message(session_id, "assistant", answer, citations)
    return {"answer": answer, "citations": citations}
