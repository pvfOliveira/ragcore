"""Tests for ingest._extract URL-vs-path routing and the empty-chunk guard."""
import pytest

from ragcore import ingest as ingest_mod
from ragcore.ingest import _extract, ingest_source


class _FakeExtractState:
    """Mimics content-core's ProcessSourceOutput (.content, .title)."""

    def __init__(self, content="extracted body", title="A Title"):
        self.content = content
        self.title = title


async def test_extract_routes_http_to_url_field(monkeypatch):
    captured = {}

    async def fake_extract_content(request):
        captured.update(request)
        return _FakeExtractState()

    monkeypatch.setattr(ingest_mod, "extract_content", fake_extract_content)

    result = await _extract("https://example.com/article")

    assert captured.get("url") == "https://example.com/article"
    assert "file_path" not in captured
    assert result == {"title": "A Title", "content": "extracted body",
                      "origin": "https://example.com/article"}


async def test_extract_routes_path_to_file_path_field(monkeypatch):
    captured = {}

    async def fake_extract_content(request):
        captured.update(request)
        return _FakeExtractState(title=None)  # no title -> fall back to the path

    monkeypatch.setattr(ingest_mod, "extract_content", fake_extract_content)

    result = await _extract("/tmp/doc.txt")

    assert captured.get("file_path") == "/tmp/doc.txt"
    assert "url" not in captured
    assert result["title"] == "/tmp/doc.txt"


class _FakeStore:
    async def find_source_id_by_origin(self, origin):
        return None

    async def create_source(self, title, full_text, origin):
        return "source:x"

    async def add_embeddings(self, source_id, chunks):
        raise AssertionError("add_embeddings must not run when there are no chunks")


async def test_ingest_raises_when_chunking_yields_nothing(monkeypatch):
    async def fake_extract(path_or_url):
        return {"title": "T", "content": "non-empty content", "origin": path_or_url}

    monkeypatch.setattr(ingest_mod, "_extract", fake_extract)
    monkeypatch.setattr(ingest_mod, "chunk_text", lambda *a, **k: [])

    with pytest.raises(ValueError, match="no chunks"):
        await ingest_source("/tmp/x.txt", store=_FakeStore(), config=None)
