"""The golden dataset is a versioned, provenance-noted eval artifact:
schema-validated, citing real corpus sources — not generate-and-forget."""
import json
from pathlib import Path

import pytest

from ragcore.eval.golden import GOLDEN_DIR, load_golden

REQUIRED = {"id", "question", "ground_truth", "source_ids", "rationale",
            "author", "date"}


def test_v1_exists_loads_and_validates():
    items = load_golden("v1")
    assert len(items) >= 10
    for it in items:
        assert REQUIRED <= set(it), f"missing fields in {it.get('id')}"
        assert it["question"].strip() and it["ground_truth"].strip()
        assert it["source_ids"], f"{it['id']} cites no sources"
        assert it["rationale"].strip(), f"{it['id']} has no rationale"


def test_ids_are_unique():
    items = load_golden("v1")
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))


def test_unknown_version_raises():
    with pytest.raises(FileNotFoundError):
        load_golden("v999")


def test_malformed_item_rejected(tmp_path, monkeypatch):
    bad = tmp_path / "vbad.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q?"}) + "\n")
    monkeypatch.setattr("ragcore.eval.golden.GOLDEN_DIR", tmp_path)
    with pytest.raises(ValueError, match="missing fields"):
        load_golden("vbad")


def test_harness_default_dataset_is_golden_v1():
    from ragcore.eval.harness import _DEFAULT_DATASET
    assert _DEFAULT_DATASET == GOLDEN_DIR / "v1.jsonl"
    assert Path(_DEFAULT_DATASET).exists()


def test_manifest_documents_provenance():
    manifest = (GOLDEN_DIR / "MANIFEST.md").read_text()
    assert "v1" in manifest and "provenance" in manifest.lower()


def test_golden_sources_exist():
    """Every cited corpus document must exist in the repo (provenance is real)."""
    repo = Path(__file__).parent.parent
    for it in load_golden("v1"):
        for sid in it["source_ids"]:
            doc = sid.split("#")[0]
            assert (repo / doc).exists(), f"{it['id']} cites missing {doc}"
