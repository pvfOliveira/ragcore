"""LIVE: a real WAV becomes searchable — mlx-whisper transcribes the committed
fixture on Metal, the transcript is ingested through the standard pipeline
(chunk -> real Ollama embeddings -> SurrealDB), and hybrid search finds the
spoken content. First run downloads the whisper model (network).
Records RunRegistry tag audio-v4."""
from __future__ import annotations

import asyncio
import copy
import socket
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

pytest.importorskip("mlx_whisper")

FIXTURE = Path(__file__).parent / "fixtures" / "spoken.wav"


def _network_up(host="huggingface.co", port=443) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_up(), reason="network needed for whisper model download")
def test_spoken_fixture_becomes_searchable(surreal_url):
    from ragcore.audio import ingest_audio, transcribe_audio
    from ragcore.config import SurrealConfig, load_config
    from ragcore.embedding import generate_embedding
    from ragcore.llmops.registry import RunRegistry
    from ragcore.retrieve import hybrid_search
    from ragcore.store import Store

    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_au", database="ragcore_au")
    cfg.store.vector_backend = "surreal"
    cfg.audio.enabled = True

    tr = transcribe_audio(str(FIXTURE), cfg)
    text = tr["text"].lower()
    # whisper-tiny may fumble "zephyr"; the rare-ish content words are the check.
    # (It normalizes spoken "seventeen" to the digits "17" — same content.)
    assert "capacitor" in text, f"transcript missed 'capacitor': {tr['text']!r}"
    assert "seventeen" in text or "17" in text, f"transcript missed the degree value: {tr['text']!r}"
    assert tr["segments"], "expected segment timing from mlx-whisper"

    # transcript flows through the standard pipeline and is retrievable
    store = Store(cfg.surreal)
    asyncio.run(store.init_schema())

    async def _run():
        result = await ingest_audio(str(FIXTURE), store, cfg)
        assert result.source_id

        async def embedder(q):
            return await generate_embedding(q, cfg, chunk_size=cfg.chunking.chunk_size)

        try:
            hits = await hybrid_search(store, embedder,
                                       "what overheats at seventeen degrees?", k=5)
            joined = " ".join(h["content"].lower() for h in hits)
            assert "capacitor" in joined, "spoken content not retrieved"

            run_id = await RunRegistry(cfg.surreal).record(
                metrics={"audio": {"transcribed": 1.0, "retrieved": 1.0,
                                   "segments": float(len(tr["segments"]))}},
                config_snapshot={"audio_model": cfg.audio.model, "backend": "mlx-whisper"},
                dataset=None, tag="audio-v4", gate_passed=True)
            assert run_id and ":" in run_id
        finally:
            await store.delete_source(result.source_id)  # leave the corpus clean

        return run_id

    run_id = asyncio.run(_run())
    print(f"\n[audio] transcript={tr['text']!r} segments={len(tr['segments'])} run_id={run_id}")
