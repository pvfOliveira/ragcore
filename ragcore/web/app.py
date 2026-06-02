"""Thin FastAPI web UI over the ragcore library (chat-centric, local-only)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ragcore.chat import chat_turn
from ragcore.config import Config
from ragcore.embedding import generate_embedding
from ragcore.errors import ConfigurationError, classify_error
from ragcore.ingest import ingest_source
from ragcore.sessions import SessionStore
from ragcore.store import Store

_STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class AddSourceRequest(BaseModel):
    origin: str


def get_config() -> Config:  # overridden in create_app
    raise RuntimeError("config dependency not configured")


def get_store(cfg: Config = Depends(get_config)) -> Store:
    return Store(cfg.surreal)


def get_session_store(cfg: Config = Depends(get_config)) -> SessionStore:
    return SessionStore(cfg.surreal)


def get_embedder(cfg: Config = Depends(get_config)):
    async def fn(query: str):
        return await generate_embedding(query, cfg, chunk_size=cfg.chunking.chunk_size)
    return fn


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="ragcore")
    app.dependency_overrides[get_config] = lambda: config

    page = (_STATIC / "index.html").read_text()

    # Catch-all so the two most common runtime failures (provider down during chat,
    # bad/empty source during ingest) return a friendly {detail} instead of a raw 500.
    # classify_error handles RagcoreError and raw provider/value errors alike.
    @app.exception_handler(Exception)
    async def _on_error(request, exc: Exception):
        cls, message = classify_error(exc)
        is_config = isinstance(exc, ConfigurationError) or issubclass(cls, ConfigurationError)
        return JSONResponse(status_code=400 if is_config else 502, content={"detail": message})

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(page)

    @app.get("/api/sessions")
    async def list_sessions(ss: SessionStore = Depends(get_session_store)):
        return await ss.list_sessions()

    @app.post("/api/sessions")
    async def create_session(req: CreateSessionRequest,
                             ss: SessionStore = Depends(get_session_store)):
        return {"id": await ss.create_session(title=req.title)}

    @app.get("/api/sessions/{sid}/messages")
    async def get_messages(sid: str, ss: SessionStore = Depends(get_session_store)):
        return await ss.get_history(sid)

    @app.post("/api/sessions/{sid}/chat")
    async def chat(sid: str, req: ChatRequest,
                   ss: SessionStore = Depends(get_session_store),
                   store: Store = Depends(get_store),
                   cfg: Config = Depends(get_config),
                   embed=Depends(get_embedder)):
        return await chat_turn(ss, store, cfg, sid, req.message, embed)

    @app.get("/api/sources")
    async def list_sources(store: Store = Depends(get_store)):
        return await store.list_sources()

    @app.post("/api/sources")
    async def add_source(req: AddSourceRequest, store: Store = Depends(get_store),
                         cfg: Config = Depends(get_config)):
        result = await ingest_source(req.origin, store, cfg, chunk_size=cfg.chunking.chunk_size)
        return {"source_id": result.source_id, "created": result.created}

    @app.delete("/api/sources/{source_id}")
    async def remove_source(source_id: str, store: Store = Depends(get_store)):
        return {"deleted": await store.delete_source(source_id)}

    return app
