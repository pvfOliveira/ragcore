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
from pydantic import BaseModel

from ragcore.cache import SemanticCache
from ragcore.observability.otel import set_gen_ai_attributes
from ragcore.observability.spans import traced_span
from ragcore.observability.tracing import get_tracer
from ragcore.providers import build_chat_model
from ragcore.rerank import rerank
from ragcore.retrieve import hybrid_search, vector_store_for
from ragcore.routing import select_model

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


def _render_inline(template_text: str, data: dict) -> str:
    """Render a raw template string (compiled prompts aren't files in prompt_dir)."""
    from jinja2 import Template

    return Template(template_text).render(**data)


def _render_strategy(config, data: dict) -> str:
    dspy_cfg = getattr(config, "dspy", None)
    if dspy_cfg is not None and dspy_cfg.enabled:
        from ragcore.dspy_optimizer import load_compiled_strategy

        compiled = load_compiled_strategy(dspy_cfg.compiled_path)
        if compiled is not None:
            return _render_inline(compiled, data)
    return _render("ask_strategy", data)


def _clean(text: str) -> str:
    text = _THINK.sub("", text)
    # Strip a dangling, unclosed <think> ... (truncated reasoning) to end-of-string.
    idx = text.find("<think>")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _select_and_build(config, content: str = "", force_cloud: bool = False):
    provider, model = select_model(config, "chat", content=content, force_cloud=force_cloud)
    from ragcore.gateway import get_gateway
    gateway = get_gateway(config)
    if gateway is not None:
        return gateway.chat_for(provider, model), provider, model
    return build_chat_model(provider, model), provider, model


# Kept for test-monkeypatching compatibility; production callers use _select_and_build.
def _build_chat(config, content: str = "", force_cloud: bool = False):
    chat, _provider, _model = _select_and_build(config, content, force_cloud)
    return chat


async def _traced_invoke(config, chat, provider, model, prompt: str, stage: str):
    """Invoke the chat model inside a gen_ai span (no-op when tracing disabled).

    The span name carries the pipeline stage for readability, but the OTel
    GenAI ``gen_ai.operation.name`` stays the conformant value ``"chat"``; the
    stage goes in a separate ``ragcore.pipeline.stage`` attribute.
    """
    tracer = get_tracer(config)
    with traced_span(tracer, f"{stage} {model}") as span:
        msg = await chat.ainvoke(prompt)
        if span is not None:
            from ragcore.chunking import token_count
            set_gen_ai_attributes(
                span, system=provider, operation="chat", model=model,
                input_tokens=token_count(prompt),
                output_tokens=token_count(getattr(msg, "content", "") or ""),
            )
            span.set_attribute("ragcore.pipeline.stage", stage)
        return msg


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
    cfg = state["_config"]
    prompt = _render_strategy(cfg, {"question": state["question"], "max_searches": 5})
    chat, provider, model = _select_and_build(cfg, content=state["question"],
                                              force_cloud=state.get("_force_cloud", False))
    msg = await _traced_invoke(cfg, chat, provider, model, prompt, "strategy")
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
    async def _search(term: str):
        return await hybrid_search(state["_store"], state["_embedder_fn"], term, k=10, config=cfg, **extra)

    if cfg is not None and getattr(cfg, "query_rewrite", None) is not None and cfg.query_rewrite.enabled:
        from ragcore.query_rewrite import expand_and_fuse
        chat_qr, p_qr, m_qr = _select_and_build(cfg, force_cloud=state.get("_force_cloud", False))
        async def _chat_fn(prompt: str):
            msg = await _traced_invoke(cfg, chat_qr, p_qr, m_qr, prompt, "query_rewrite")
            return _clean(msg.content)
        chunks = await expand_and_fuse(term, cfg, _chat_fn, _search, k=10)
    else:
        chunks = await _search(term)

    if cfg is not None and cfg.rerank.enabled:
        chunks = rerank(term, chunks, top_k=cfg.rerank.top_k, model=cfg.rerank.model)
    prompt = _render("ask_answer", {"term": term, "chunks": chunks})
    chat, provider, model = _select_and_build(cfg, force_cloud=state.get("_force_cloud", False))
    msg = await _traced_invoke(cfg, chat, provider, model, prompt, "retrieve_answer")
    cites = [c["source"] for c in chunks]
    return {"answers": [{"answer": _clean(msg.content), "citations": cites}]}


async def _synthesize(state: AskState) -> dict:
    cfg = state["_config"]
    partials = [a["answer"] for a in state["answers"]]
    citations = sorted({c for a in state["answers"] for c in a["citations"]})
    prompt = _render("ask_final", {"question": state["question"], "answers": partials})
    chat, provider, model = _select_and_build(cfg, force_cloud=state.get("_force_cloud", False))
    msg = await _traced_invoke(cfg, chat, provider, model, prompt, "synthesize")
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

_ledger: dict[str, Any] = {}  # keyed by ledger_path


def _get_cache(config) -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(path=_CACHE_PATH, threshold=config.cache.threshold)
    return _cache


def _get_ledger(config):
    from ragcore.cost.ledger import CostLedger

    path = config.cost.ledger_path
    if path not in _ledger:
        _ledger[path] = CostLedger(path=path)
    return _ledger[path]


def _record_usage(config, question: str, answer: str, *, cached: bool,
                  force_cloud: bool = False) -> None:
    """Record one usage row when cost tracking is enabled; no-op otherwise."""
    if config is None or getattr(config, "cost", None) is None or not config.cost.enabled:
        return
    from ragcore.chunking import token_count
    provider, model = select_model(config, "chat", content=question, force_cloud=force_cloud)
    _get_ledger(config).record(
        role="chat", provider=provider, model=model,
        prompt_tokens=0 if cached else token_count(question),
        completion_tokens=0 if cached else token_count(answer),
        cached=cached,
    )


async def _answer_question_inner(question: str, store, config, embedder_fn, force_cloud: bool = False) -> dict:
    if config is not None and getattr(config, "agentic", None) is not None and config.agentic.enabled:
        import ragcore.agentic as _ag
        cfg = config

        async def _search(term: str):
            vs = vector_store_for(cfg)
            extra = {"vector_store": vs} if vs is not None else {}
            return await hybrid_search(store, embedder_fn, term, k=10, config=cfg, **extra)

        async def _chat_fn(prompt: str):
            chat, provider, model = _select_and_build(cfg, content=prompt, force_cloud=force_cloud)
            msg = await _traced_invoke(cfg, chat, provider, model, prompt, "agentic")
            return _clean(msg.content)

        result = await _ag.run_agentic(question, cfg.agentic, _chat_fn, _search)
        _record_usage(cfg, question, result["answer"], cached=False, force_cloud=force_cloud)
        return {"answer": result["answer"], "citations": result["citations"]}

    # Budget enforcement: check BEFORE any LLM call.
    if config is not None and getattr(config, "cost", None) is not None and config.cost.enabled:
        if config.cost.enforce:
            from ragcore.chunking import token_count
            from ragcore.cost.ledger import over_budget
            from ragcore.errors import RagcoreError

            if over_budget(token_count(question), config.cost.budget_tokens):
                raise RagcoreError(
                    f"request of {token_count(question)} tokens exceeds budget ({config.cost.budget_tokens})"
                )

    if config is not None and getattr(config, "cache", None) is not None and config.cache.enabled:
        from ragcore.embedding import generate_embedding
        q_emb = await generate_embedding(question, config)
        cache = _get_cache(config)
        hit = cache.get(q_emb)
        if hit is not None:
            _record_usage(config, question, "", cached=True, force_cloud=force_cloud)
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
        _record_usage(config, question, answer, cached=False, force_cloud=force_cloud)
        return {"answer": answer, "citations": citations}

    graph = _build_graph()
    result = await graph.ainvoke({
        "question": question, "answers": [],
        "_store": store, "_config": config, "_embedder_fn": embedder_fn,
        "_force_cloud": force_cloud,
    })
    answer = result["answer"]
    citations = result.get("citations", [])
    _record_usage(config, question, answer, cached=False, force_cloud=force_cloud)
    return {"answer": answer, "citations": citations}


async def answer_question(question: str, store, config, embedder_fn, force_cloud: bool = False) -> dict:
    tracer = get_tracer(config)
    with traced_span(tracer, "ask") as root:
        if root is not None:
            root.set_attribute("ragcore.question_chars", len(question))
        return await _answer_question_inner(question, store, config, embedder_fn, force_cloud)


class StructuredAnswer(BaseModel):
    answer: str
    citations: list[str] = []
    confidence: float = 0.0


async def answer_structured(question: str, store, config, embedder_fn):
    """Retrieve as usual, then synthesise a schema-validated answer object.

    Structured backends (instructor/outlines) run against the local Ollama model
    in ``config.structured.model``; there is no cloud-escalation path here, so this
    intentionally takes no ``force_cloud`` (unlike ``answer_question``)."""
    from ragcore.structured import generate_structured, _render as _srender
    vector_store = vector_store_for(config)
    extra = {"vector_store": vector_store} if vector_store is not None else {}
    chunks = await hybrid_search(store, embedder_fn, question, k=10, config=config, **extra)
    prompt = _srender("structured_answer", {"question": question, "chunks": chunks})
    return await generate_structured(prompt, StructuredAnswer, config)
