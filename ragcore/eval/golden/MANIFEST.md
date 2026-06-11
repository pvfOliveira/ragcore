# Golden eval datasets

| version | items | corpus | date | author |
|---|---|---|---|---|
| v1 | 10 | tests/live/fixtures/grid_storage.md | 2026-06-11 | pvf |

## Provenance (v1)

Hand-curated against `tests/live/fixtures/grid_storage.md` (the committed
grid-scale energy storage reference; the live eval tier ingests this exact
document). Each item cites its source section(s) in `source_ids`
(`<repo-relative path>#<section-anchor>`) and states in `rationale` which
failure mode it probes. Ground truths are written from the document text,
never from world knowledge.

Items g01–g03 carry over the pre-v4 `ragcore/eval/dataset.jsonl` questions
verbatim (kept for metric continuity); g04–g10 extend coverage:

- g04 — context precision (enumerating list vs adjacent per-technology prose)
- g05 — rare-term recall ("vanadium redox"/"VRFB" appears in one section only)
- g06 — paraphrase robustness (question wording shares almost no tokens with
  the answering section)
- g07 — multi-hop (capacity share and efficiency comparison span two sections)
- g08 — negation/contrast (the corpus states what storage is NOT)
- g09 — numeric precision (competing efficiency range one section away)
- g10 — groundedness trap (plausible-but-absent explanation; only the
  documented figures are in the corpus)

Versioning: golden sets are append-only — corrections create v2, never edit
v1 in place (run history must stay comparable).
