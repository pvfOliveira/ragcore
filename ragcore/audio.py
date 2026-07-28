"""Audio ingest: mlx-whisper transcription -> the existing text pipeline.

First-party pipeline (module + config + CLI routing + tests): the transcript
becomes a normal text source via ``ingest_source(content=...)`` — chunked,
embedded, hybrid-searchable, citable (the VLM-caption precedent). mlx-whisper
runs Whisper natively on Apple-Silicon Metal via MLX.

Timestamps: segment starts can be inlined as ``[mm:ss]`` markers (config
``timestamps=true``) so they survive as searchable text. (The originally
planned "timestamps in chunk metadata" is realized as inline markers — the
chunk store carries no metadata column; noted deviation, same honest capability.)
"""
from __future__ import annotations

from pathlib import Path

from ragcore.errors import RagcoreError

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}  # deliberate minimal set; mlx-whisper/ffmpeg also accept .ogg/.aac/.opus — extend when needed


def is_audio_path(source: str) -> bool:
    return Path(source).suffix.lower() in _AUDIO_EXTENSIONS


def transcribe_audio(path: str, config) -> dict:
    """Transcribe *path* with mlx-whisper.

    Returns ``{"text": str, "segments": [{"start","end","text"}], "language": str}``.
    """
    if not config.audio.enabled:
        raise RagcoreError(
            "Audio ingest is disabled. Set [audio] enabled = true in config.toml "
            "and install the extra: uv pip install -e '.[audio]'"
        )
    import mlx_whisper  # lazy — optional extra

    result = mlx_whisper.transcribe(path, path_or_hf_repo=config.audio.model)
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
        for s in result.get("segments", [])
    ]
    return {"text": result["text"].strip(), "segments": segments,
            "language": result.get("language", "")}


def format_transcript(segments: list[dict], timestamps: bool = False) -> str:
    """Join segments into transcript text, optionally with [mm:ss] markers.

    Minutes are unbounded (no hour field), e.g. ``[75:30]`` for 1h15m30s.
    """
    if not timestamps:
        return " ".join(s["text"].strip() for s in segments).strip()
    parts = []
    for s in segments:
        m, sec = divmod(int(s["start"]), 60)
        parts.append(f"[{m:02d}:{sec:02d}] {s['text'].strip()}")
    return "\n".join(parts)


async def ingest_audio(path: str, store, config):
    """Transcribe and ingest an audio file as a searchable text source."""
    import ragcore.ingest as _ingest

    tr = transcribe_audio(path, config)
    if not tr["text"] and not tr["segments"]:
        raise RagcoreError(f"Transcription returned empty output for {path}")
    content = format_transcript(tr["segments"], timestamps=config.audio.timestamps) \
        if tr["segments"] else tr["text"]
    return await _ingest.ingest_source(
        path, store, config,
        chunk_size=config.chunking.chunk_size,
        content=content, title=Path(path).name,
    )
