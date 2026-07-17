"""Prompt templates are pinned supply-chain artifacts (OWASP ASI04):
any change must be deliberate -- edit, review, regenerate the lock."""
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "prompts.lock"
PROMPT_GLOBS = ["ragcore/prompts/*.jinja"]


def _entries() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest
    return out


def test_prompt_templates_match_lock():
    entries = _entries()
    assert entries, "prompts.lock has no entries"
    for rel, digest in entries.items():
        p = REPO / rel
        assert p.exists(), f"locked prompt missing from tree: {rel}"
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == digest, (
            f"prompt template drifted: {rel} -- if deliberate, regenerate: "
            f"cd {REPO.name} && shasum -a 256 ragcore/prompts/*.jinja > prompts.lock"
        )


def test_no_unpinned_prompts():
    entries = _entries()
    for glob in PROMPT_GLOBS:
        for p in sorted(REPO.glob(glob)):
            rel = p.relative_to(REPO).as_posix()
            assert rel in entries, f"unpinned prompt template: {rel}"
