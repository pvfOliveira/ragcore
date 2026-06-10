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
