"""Async LangGraph 'ask' workflow: strategy -> parallel hybrid retrieve -> synthesize.

Mirrors open-notebook's ask graph shape, but retrieval is hybrid (vector + text)
instead of vector-only.
"""
from __future__ import annotations

import json
import operator
import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

from ai_prompter import Prompter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ragcore.cache import SemanticCache
from ragcore.providers import build_chat_model
from ragcore.rerank import rerank
from ragcore.retrieve import hybrid_search, vector_store_for
from ragcore.routing import select_model

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


def _clean(text: str) -> str:
    text = _THINK.sub("", text)
    # Strip a dangling, unclosed <think> ... (truncated reasoning) to end-of-string.
    idx = text.find("<think>")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _build_chat(config, content: str = "", force_cloud: bool = False):
    provider, model = select_model(config, "chat", content=content, force_cloud=force_cloud)
    return build_chat_model(provider, model)


class AskState(TypedDict, total=False):
    question: str
    searches: list[str]
    answers: Annotated[list, operator.add]
    answer: str
    citations: list[str]
    _store: Any
    _config: Any
    _embedder_fn: Any
    _force_cloud: bool


async def _strategy(state: AskState) -> dict:
    prompt = _render("ask_strategy", {"question": state["question"], "max_searches": 5})
    chat = _build_chat(state["_config"], content=state["question"], force_cloud=state.get("_force_cloud", False))
    msg = await chat.ainvoke(prompt)
    try:
        parsed = json.loads(_clean(msg.content))
        searches = parsed.get("searches", []) if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, ValueError, TypeError):
        searches = []
    return {"searches": searches or [state["question"]]}


def _fan_out(state: AskState):
    return [Send("retrieve_answer", {**state, "_term": term}) for term in state["searches"]]


async def _retrieve_answer(state: dict) -> dict:
    term = state["_term"]
    cfg = state.get("_config")
    vector_store = vector_store_for(cfg)
    extra = {"vector_store": vector_store} if vector_store is not None else {}
    chunks = await hybrid_search(state["_store"], state["_embedder_fn"], term, k=10, **extra)
    if cfg is not None and cfg.rerank.enabled:
        chunks = rerank(term, chunks, top_k=cfg.rerank.top_k, model=cfg.rerank.model)
    prompt = _render("ask_answer", {"term": term, "chunks": chunks})
    chat = _build_chat(state["_config"], force_cloud=state.get("_force_cloud", False))
    msg = await chat.ainvoke(prompt)
    cites = [c["source"] for c in chunks]
    return {"answers": [{"answer": _clean(msg.content), "citations": cites}]}


async def _synthesize(state: AskState) -> dict:
    partials = [a["answer"] for a in state["answers"]]
    citations = sorted({c for a in state["answers"] for c in a["citations"]})
    prompt = _render("ask_final", {"question": state["question"], "answers": partials})
    chat = _build_chat(state["_config"], force_cloud=state.get("_force_cloud", False))
    msg = await chat.ainvoke(prompt)
    return {"answer": _clean(msg.content), "citations": citations}


def _build_graph():
    g = StateGraph(AskState)
    g.add_node("strategy", _strategy)
    g.add_node("retrieve_answer", _retrieve_answer)
    g.add_node("synthesize", _synthesize)
    g.add_edge(START, "strategy")
    g.add_conditional_edges("strategy", _fan_out, ["retrieve_answer"])
    g.add_edge("retrieve_answer", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_CACHE_PATH = str(Path(__file__).parent.parent / "data" / "semantic_cache.db")
_cache: SemanticCache | None = None


def _get_cache(config) -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(path=_CACHE_PATH, threshold=config.cache.threshold)
    return _cache


async def answer_question(question: str, store, config, embedder_fn, force_cloud: bool = False) -> dict:
    if config is not None and getattr(config, "cache", None) is not None and config.cache.enabled:
        from ragcore.embedding import generate_embedding
        q_emb = await generate_embedding(question, config)
        cache = _get_cache(config)
        hit = cache.get(q_emb)
        if hit is not None:
            return {"answer": hit["answer"], "citations": hit["sources"]}

        graph = _build_graph()
        result = await graph.ainvoke({
            "question": question, "answers": [],
            "_store": store, "_config": config, "_embedder_fn": embedder_fn,
            "_force_cloud": force_cloud,
        })
        answer = result["answer"]
        citations = result.get("citations", [])
        cache.put(question, q_emb, answer=answer, sources=citations)
        return {"answer": answer, "citations": citations}

    graph = _build_graph()
    result = await graph.ainvoke({
        "question": question, "answers": [],
        "_store": store, "_config": config, "_embedder_fn": embedder_fn,
        "_force_cloud": force_cloud,
    })
    return {"answer": result["answer"], "citations": result.get("citations", [])}
