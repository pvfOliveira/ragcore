"""Versioned golden eval datasets with provenance.

A golden item is ``{id, question, ground_truth, source_ids, rationale,
author, date}`` — the extra fields document WHY each question exists and
WHAT corpus content answers it. ``ragcore eval``/``gate`` consume golden v1
by default (``--dataset`` still overrides)."""
from __future__ import annotations

import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden"

REQUIRED_FIELDS = {"id", "question", "ground_truth", "source_ids",
                   "rationale", "author", "date"}


def load_golden(version: str = "v1") -> list[dict]:
    """Load and validate a golden dataset version. Raises on malformed items."""
    path = GOLDEN_DIR / f"{version}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no golden dataset {version!r} at {path}")
    items = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(f"{path.name}:{n} missing fields: {sorted(missing)}")
        items.append(item)
    return items
