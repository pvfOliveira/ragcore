"""Token-based, content-type-aware chunking (reimplemented from open-notebook lessons)."""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

import tiktoken

_ENCODER = tiktoken.get_encoding("o200k_base")


def token_count(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODER.encode(text))


class ContentType(Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    PLAIN = "plain"


_EXT = {
    ".md": ContentType.MARKDOWN, ".markdown": ContentType.MARKDOWN,
    ".html": ContentType.HTML, ".htm": ContentType.HTML,
}


def detect_content_type(text: str, file_path: Optional[str] = None) -> ContentType:
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in _EXT:
            return _EXT[ext]
    if re.search(r"<html|<body|<div", text, re.IGNORECASE):
        return ContentType.HTML
    if len(re.findall(r"^#{1,6}\s+\S", text, re.MULTILINE)) >= 1:
        return ContentType.MARKDOWN
    return ContentType.PLAIN


def _split_recursive(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy paragraph/sentence/word splitter measured in tokens."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(s: str, seps: list[str]) -> list[str]:
        if token_count(s) <= chunk_size or not seps:
            return [s]
        sep, rest = seps[0], seps[1:]
        parts = s.split(sep)
        out, buf = [], ""
        for p in parts:
            candidate = (buf + sep + p) if buf else p
            if token_count(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    out.append(buf)
                buf = p if token_count(p) <= chunk_size else ""
                if not buf:
                    out.extend(_split(p, rest))
        if buf:
            out.append(buf)
        return out

    chunks = _split(text, separators)
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    # Prepend tail of previous chunk (token-bounded) for context continuity.
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tokens = _ENCODER.encode(chunks[i - 1])[-overlap:]
        tail = _ENCODER.decode(prev_tokens)
        overlapped.append(tail + " " + chunks[i])
    return overlapped


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_size: int = 5,
    file_path: Optional[str] = None,
) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    if token_count(text) <= chunk_size:
        return [text]
    chunks = [c.strip() for c in _split_recursive(text, chunk_size, chunk_overlap) if c.strip()]
    if min_chunk_size > 0 and len(chunks) > 1:
        kept = [c for c in chunks if token_count(c) >= min_chunk_size]
        if kept:
            chunks = kept
    return chunks
