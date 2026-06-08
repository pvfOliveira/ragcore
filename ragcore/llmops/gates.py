from __future__ import annotations

import math


def _flatten(metrics: dict) -> dict[str, float]:
    """Flatten {family: {name: float}} metric reports to {'family.name': float}.
    Only nested numeric leaves are compared; top-level scalars are intentionally
    ignored (run_eval emits nested families)."""
    out = {}
    for family, scores in metrics.items():
        if isinstance(scores, dict):
            for name, val in scores.items():
                if isinstance(val, (int, float)):
                    out[f"{family}.{name}"] = float(val)
    return out


def check_gate(current: dict, baseline: dict, tolerance: float):
    """Higher-is-better metrics. Regression = current < baseline - tolerance.
    Returns (ok: bool, regressions: list[dict])."""
    cur, base = _flatten(current), _flatten(baseline)
    regressions = []
    for metric, base_val in base.items():
        if metric in cur:
            delta = cur[metric] - base_val
            if delta < -tolerance:
                regressions.append({"metric": metric, "delta": delta,
                                    "baseline": base_val, "current": cur[metric]})
    return (len(regressions) == 0), regressions


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 1.0   # different embedding dimensionality => maximal drift
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def check_drift(baseline_centroid, current_centroid, threshold: float):
    """Returns (drifted, distance); a dimensionality mismatch counts as maximal drift (distance 1.0)."""
    distance = _cosine_distance(baseline_centroid, current_centroid)
    return (distance > threshold), distance
