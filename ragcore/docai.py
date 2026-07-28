"""Document-AI: parse PDFs / scans into markdown for the ingest path, and a modest
VLM image-caption assist. All opt-in; heavy libs lazy-imported."""
from __future__ import annotations


def _parse_docling(path: str) -> str:
    from docling.document_converter import DocumentConverter  # lazy — optional extra
    conv = DocumentConverter()
    return conv.convert(path).document.export_to_markdown()


def _parse_pymupdf(path: str) -> str:
    import pymupdf4llm  # lazy — fallback
    return pymupdf4llm.to_markdown(path)


def parse_document(path: str, config) -> str:
    """Return markdown for *path* using config.docai.parser (docling | pymupdf)."""
    cfg = config.docai
    if cfg.parser == "docling":
        return _parse_docling(path)
    if cfg.parser == "pymupdf":
        return _parse_pymupdf(path)
    raise ValueError(f"Unknown docai parser {cfg.parser!r}")


_DOC_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


def extract_if_document(origin: str, config) -> str | None:
    """If docai is enabled and *origin* is a parseable document, return its markdown;
    otherwise None so the caller uses its normal extraction path."""
    from pathlib import Path
    cfg = getattr(config, "docai", None)
    if cfg is None or not cfg.enabled:
        return None
    if Path(origin).suffix.lower() not in _DOC_SUFFIXES:
        return None
    return parse_document(origin, config)


def _ollama_caption(model: str, path: str) -> str:
    import ollama  # lazy — uses local Ollama
    resp = ollama.chat(model=model, messages=[{
        "role": "user",
        "content": "Describe this image in one concise sentence for search indexing.",
        "images": [path],
    }])
    return resp["message"]["content"].strip()


def caption_image(path: str, config) -> str | None:
    """Return a VLM caption for *path* when multimodal.vlm_enabled, else None.

    Local vision is weak — this is an image→caption→text-index assist, NOT doc-VQA."""
    mm = getattr(config, "multimodal", None)
    if mm is None or not getattr(mm, "vlm_enabled", False):
        return None
    return _ollama_caption(mm.vlm_model, path)
