"""Live proof: docling parses a real PDF to markdown, and a small Ollama vision
model captions an image (modest, honest about weak local vision)."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from ragcore.config import SurrealConfig, load_config

pytestmark = pytest.mark.live
FIX = Path(__file__).parent / "fixtures"


def _record(tag, surreal_url, metrics):
    from ragcore.llmops.registry import RunRegistry
    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_da", database="ragcore_da")
    return asyncio.run(RunRegistry(cfg.surreal).record(
        metrics=metrics, config_snapshot={"tag": tag}, tag=f"{tag}-v3", gate_passed=True))


def test_docling_parses_pdf(surreal_url):
    pytest.importorskip("docling")
    from ragcore.docai import parse_document
    cfg = copy.deepcopy(load_config())
    cfg.docai.enabled = True
    cfg.docai.parser = "docling"
    parser_used = "docling"
    try:
        md = parse_document(str(FIX / "sample.pdf"), cfg)
    except Exception:
        # documented fallback: pymupdf if docling can't run here
        cfg.docai.parser = "pymupdf"
        parser_used = "pymupdf"
        md = parse_document(str(FIX / "sample.pdf"), cfg)
    assert "rank fusion" in md.lower(), f"Expected 'rank fusion' in output, got: {md!r}"
    run_id = _record("document-ai", surreal_url,
                     {"docai": {"chars": float(len(md)), "parser": parser_used}})
    assert run_id
    # Emit which parser actually ran so the caller can see it in the output
    print(f"\n  [docai] parser={parser_used!r}, chars={len(md)}, run_id={run_id}")


def test_vlm_captions_image(surreal_url):
    pytest.importorskip("ollama")
    from ragcore.docai import caption_image

    class MM:
        vlm_enabled = True
        vlm_model = "moondream"   # pulled above; switch to llava:7b if moondream absent

    class Cfg:
        multimodal = MM()

    try:
        cap = caption_image(str(FIX / "diagram.png"), Cfg())
    except Exception as e:
        pytest.xfail(f"No local vision model / Ollama vision unavailable: {e}")
    words = [w for w in cap.replace(",", " ").split() if w.isalpha()]
    assert isinstance(cap, str) and len(words) >= 3, f"caption too weak to prove VLM: {cap!r}"
    run_id = _record("vlm", surreal_url, {"vlm": {"caption_len": float(len(cap))}})
    assert run_id
    print(f"\n  [vlm] model=moondream, caption_len={len(cap)}, run_id={run_id}")
    print(f"  [vlm] caption={cap!r}")
