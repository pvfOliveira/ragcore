"""Multi-turn chat over the corpus: history-aware RAG (plain async, no LangGraph)."""
from __future__ import annotations

from pathlib import Path

from ai_prompter import Prompter

from ragcore.ask import _build_chat, _clean  # reuse model provisioning + thinking-strip
from ragcore.retrieve import hybrid_search, vector_store_for

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

    vector_store = vector_store_for(config)
    extra = {"vector_store": vector_store} if vector_store is not None else {}
    chunks = await hybrid_search(store, embedder_fn, query, k=10, config=config, **extra)
    if config.compression.enabled:
        from ragcore.compress import compress_context
        chunks, _comp_stats = compress_context(chunks, query, config)
    chat = _build_chat(config, content=message)
    amsg = await chat.ainvoke(_render("chat_answer",
                                      {"history": history, "chunks": chunks, "message": message}))
    answer = _clean(amsg.content)
    citations = sorted({c["source"] for c in chunks})

    # Awaited (not create_task): the CLI runs asyncio.run per turn — a detached
    # task would be cancelled at loop teardown. Sampled turns pay the judge
    # latency; that's the documented trade-off of online scoring.
    if getattr(config, "online_eval", None) is not None and config.online_eval.enabled:
        from ragcore.eval import online
        await online.maybe_score_turn(message, answer, chunks, config)

    await session_store.add_message(session_id, "user", message)
    await session_store.add_message(session_id, "assistant", answer, citations)
    return {"answer": answer, "citations": citations}
