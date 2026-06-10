"""Document-AI: parse PDFs / scans into markdown for the ingest path, and a modest
VLM image-caption assist. All opt-in; heavy libs lazy-imported."""
from __future__ import annotations


def _parse_docling(path: str) -> str:
    from docling.document_converter import DocumentConverter  # lazy — optional extra
    conv = DocumentConverter()
    return conv.convert(path).document.export_to_markdown()


def _parse_pymupdf(path: str) -> str:
    import pymupdf4llm                                       # lazy — fallback
    return pymupdf4llm.to_markdown(path)


def parse_document(path: str, config) -> str:
    """Return markdown for *path* using config.docai.parser (docling | pymupdf)."""
    cfg = config.docai
    if cfg.parser == "docling":
        return _parse_docling(path)
    if cfg.parser == "pymupdf":
        return _parse_pymupdf(path)
    raise ValueError(f"Unknown docai parser {cfg.parser!r}")
