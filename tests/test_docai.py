from ragcore.config import DocaiConfig
from ragcore.docai import parse_document


class _Cfg:
    def __init__(self, **kw): self.docai = DocaiConfig(**kw)
    multimodal = None


def test_parse_document_dispatches_to_parser(monkeypatch, tmp_path):
    p = tmp_path / "doc.pdf"; p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr("ragcore.docai._parse_docling", lambda path: "# Title\n\nbody text")
    cfg = _Cfg(enabled=True, parser="docling")
    out = parse_document(str(p), cfg)
    assert "body text" in out


def test_ingest_uses_docai_for_pdf(monkeypatch, tmp_path):
    import ragcore.docai as docai
    monkeypatch.setattr(docai, "_parse_docling", lambda path: "extracted markdown body")
    cfg = _Cfg(enabled=True, parser="docling")
    p = tmp_path / "f.pdf"; p.write_bytes(b"%PDF-1.4")
    text = docai.extract_if_document(str(p), cfg)
    assert text == "extracted markdown body"
    # non-document path returns None (caller falls back to existing extraction)
    assert docai.extract_if_document(str(tmp_path / "f.txt"), cfg) is None
