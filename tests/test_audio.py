"""Deterministic tests for the audio ingest pipeline (mocked transcriber).

The transcript flows through the EXISTING ingest_source path via its new
content= override — first-party pipeline (module + config + CLI routing),
not a one-off script (the brief's landmine)."""
import pytest

import ragcore.audio as audio_mod
from ragcore.audio import is_audio_path, transcribe_audio
from ragcore.config import AudioConfig, ChunkingConfig
from ragcore.errors import RagcoreError


class Cfg:
    def __init__(self, **kw):
        self.audio = AudioConfig(**kw)
        self.chunking = ChunkingConfig()


def test_is_audio_path():
    assert is_audio_path("talk.mp3") and is_audio_path("a/b/REC.WAV")
    assert is_audio_path("x.m4a") and is_audio_path("x.flac")
    assert not is_audio_path("doc.pdf") and not is_audio_path("img.png")


def test_transcribe_disabled_raises_actionable_error():
    with pytest.raises(RagcoreError, match=r"\[audio\]"):
        transcribe_audio("talk.mp3", Cfg(enabled=False))


def test_format_transcript_plain_and_timestamped():
    segments = [{"start": 0.0, "end": 2.5, "text": " Hello there."},
                {"start": 2.5, "end": 5.0, "text": " General Kenobi."}]
    plain = audio_mod.format_transcript(segments, timestamps=False)
    assert plain == "Hello there. General Kenobi."
    stamped = audio_mod.format_transcript(segments, timestamps=True)
    assert stamped.startswith("[00:00] Hello there.")
    assert "[00:02] General Kenobi." in stamped


async def test_ingest_audio_routes_transcript_through_ingest_source(monkeypatch):
    calls = {}

    def fake_transcribe(path, config):
        calls["transcribed"] = path
        return {"text": "spoken words here", "segments": [
            {"start": 0.0, "end": 1.0, "text": "spoken words here"}], "language": "en"}

    async def fake_ingest_source(path, store, config, chunk_size=400,
                                 content=None, title=None):
        calls["ingest"] = {"path": path, "content": content, "title": title}
        from ragcore.ingest import IngestResult
        return IngestResult(source_id="src9", created=True)

    monkeypatch.setattr(audio_mod, "transcribe_audio", fake_transcribe)
    # ingest_audio resolves ingest_source through the module attribute at call time, so patching the module-level name works
    monkeypatch.setattr("ragcore.ingest.ingest_source", fake_ingest_source)

    result = await audio_mod.ingest_audio("talk.mp3", store=object(),
                                          config=Cfg(enabled=True))
    assert result.source_id == "src9"
    assert calls["transcribed"] == "talk.mp3"
    assert calls["ingest"]["content"] == "spoken words here"
    assert calls["ingest"]["title"] == "talk.mp3"


async def test_ingest_source_content_override_skips_extraction(monkeypatch):
    """The new content= param must bypass content_core extraction entirely
    and otherwise behave identically (chunk → embed → store)."""
    from ragcore.ingest import ingest_source

    class FakeStore:
        def __init__(self):
            self.created = None
            self.rows = None

        async def find_source_id_by_origin(self, origin):
            return None

        async def create_source(self, title, full_text, origin):
            self.created = {"title": title, "full_text": full_text, "origin": origin}
            return "src1"

        async def add_embeddings(self, source_id, rows):
            self.rows = rows

    async def fake_embeddings(chunks, config):
        return [[0.0] * 3 for _ in chunks]

    monkeypatch.setattr("ragcore.ingest.generate_embeddings", fake_embeddings)

    def boom(*a, **k):
        raise AssertionError("content_core extraction must not run")
    monkeypatch.setattr("ragcore.ingest._extract", boom)

    store = FakeStore()
    res = await ingest_source("talk.mp3", store, config=None,
                              content="the transcript text", title="talk.mp3")
    assert res.created and store.created["full_text"] == "the transcript text"
    assert store.created["title"] == "talk.mp3"
    assert store.rows                                   # chunked + embedded


async def test_ingest_audio_empty_transcription_raises(monkeypatch):
    monkeypatch.setattr(audio_mod, "transcribe_audio",
                        lambda path, config: {"text": "", "segments": [], "language": ""})
    with pytest.raises(RagcoreError, match="empty output"):
        await audio_mod.ingest_audio("silent.mp3", store=object(),
                                     config=Cfg(enabled=True))
