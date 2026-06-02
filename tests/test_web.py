import pytest
from fastapi.testclient import TestClient

import ragcore.web.app as webapp
from ragcore.config import ChatConfig, ChunkingConfig, Config, ModelRole, SurrealConfig
from ragcore.errors import StoreError
from ragcore.ingest import IngestResult
from ragcore.web.app import create_app, get_embedder, get_session_store, get_store


def _config():
    return Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="m"),
            "embedding": ModelRole(local_provider="ollama", local_model="e"),
        },
        chunking=ChunkingConfig(),
        surreal=SurrealConfig(),
        chat=ChatConfig(),
    )


class FakeSessionStore:
    async def list_sessions(self):
        return [{"id": "chat_session:1", "title": "A", "messages": 2, "created": "t"}]

    async def create_session(self, title=None):
        return "chat_session:new"

    async def get_history(self, sid, limit=None):
        return [{"role": "user", "content": "hi", "citations": None, "created": "t"}]


class FakeStore:
    async def list_sources(self):
        return [{"id": "source:1", "title": "Doc", "origin": "/a", "chunks": 3, "created": "t"}]

    async def delete_source(self, source_id):
        return True


@pytest.fixture()
def client(monkeypatch):
    app = create_app(_config())
    app.dependency_overrides[get_session_store] = lambda: FakeSessionStore()
    app.dependency_overrides[get_store] = lambda: FakeStore()
    app.dependency_overrides[get_embedder] = lambda: (lambda q: None)

    async def fake_chat_turn(ss, store, cfg, sid, message, embed):
        return {"answer": f"echo:{message}", "citations": ["source:1"]}

    async def fake_ingest(origin, store, cfg, chunk_size=400):
        return IngestResult(source_id="source:new", created=True)

    monkeypatch.setattr(webapp, "chat_turn", fake_chat_turn)
    monkeypatch.setattr(webapp, "ingest_source", fake_ingest)
    # raise_server_exceptions=False mirrors real uvicorn: the Exception handler's
    # response is returned (not re-raised), so error mapping is testable.
    return TestClient(app, raise_server_exceptions=False)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ragcore" in r.text.lower()


def test_create_and_list_sessions(client):
    r = client.post("/api/sessions", json={"title": "X"})
    assert r.status_code == 200 and r.json()["id"] == "chat_session:new"
    r = client.get("/api/sessions")
    assert r.status_code == 200 and r.json()[0]["id"] == "chat_session:1"


def test_get_messages(client):
    r = client.get("/api/sessions/chat_session:1/messages")
    assert r.status_code == 200 and r.json()[0]["role"] == "user"


def test_chat_endpoint(client):
    r = client.post("/api/sessions/chat_session:1/chat", json={"message": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "echo:hello" and body["citations"] == ["source:1"]


def test_sources_crud(client):
    assert client.get("/api/sources").json()[0]["id"] == "source:1"
    r = client.post("/api/sources", json={"origin": "/tmp/x.txt"})
    assert r.status_code == 200 and r.json() == {"source_id": "source:new", "created": True}
    r = client.delete("/api/sources/source:1")
    assert r.status_code == 200 and r.json() == {"deleted": True}


def test_store_error_maps_to_502(client):
    class Boom(FakeStore):
        async def list_sources(self):
            raise StoreError("db down")
    client.app.dependency_overrides[get_store] = lambda: Boom()
    r = client.get("/api/sources")
    assert r.status_code == 502


def test_create_session_without_title(client):
    # The frontend always POSTs {} (no title) — exercise that path.
    r = client.post("/api/sessions", json={})
    assert r.status_code == 200 and r.json()["id"] == "chat_session:new"


def test_chat_raw_provider_error_maps_to_502(client, monkeypatch):
    async def boom(ss, store, cfg, sid, message, embed):
        raise ConnectionError("connection refused")
    monkeypatch.setattr(webapp, "chat_turn", boom)
    r = client.post("/api/sessions/chat_session:1/chat", json={"message": "hi"})
    assert r.status_code == 502 and "detail" in r.json()


def test_add_source_value_error_maps_to_502(client, monkeypatch):
    async def boom(origin, store, cfg, chunk_size=400):
        raise ValueError("No content extracted")
    monkeypatch.setattr(webapp, "ingest_source", boom)
    r = client.post("/api/sources", json={"origin": "/bad"})
    assert r.status_code == 502 and "detail" in r.json()
    assert "detail" in r.json()
