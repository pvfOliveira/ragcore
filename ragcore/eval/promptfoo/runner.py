"""Invoke the Promptfoo Node CLI and parse its JSON results into the registry's
metric shape. Promptfoo is a Node tool; this is a thin, skip-if-absent wrapper —
CI never depends on Node, the live tier requires the CLI present."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_CONFIG = Path(__file__).parent / "promptfooconfig.yaml"


def promptfoo_available() -> bool:
    return shutil.which("promptfoo") is not None


def run_promptfoo(out_path: str | Path = "data/eval/promptfoo.json") -> dict[str, dict[str, float]]:
    """Run `promptfoo eval` and return the parsed metric dict. Raises if the CLI
    is absent (callers in the live tier guard with ``promptfoo_available``)."""
    if not promptfoo_available():
        raise RuntimeError("promptfoo CLI not found on PATH (npm i -g promptfoo)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["promptfoo", "eval", "-c", str(_CONFIG), "-o", str(out)],
        check=True, capture_output=True, text=True,
    )
    return parse_results(out)


def parse_results(out_path: str | Path) -> dict[str, dict[str, float]]:
    data = json.loads(Path(out_path).read_text())
    stats = data.get("results", {}).get("stats", {})
    successes = float(stats.get("successes", 0))
    failures = float(stats.get("failures", 0))
    total = successes + failures
    return {"promptfoo": {
        "pass_rate": (successes / total) if total else 0.0,
        "successes": successes, "failures": failures,
    }}
