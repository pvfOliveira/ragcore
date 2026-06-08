"""Build a cost/usage report from a CostLedger aggregate."""
from __future__ import annotations


def build_report(agg: dict, rates: dict, bench) -> dict:
    """Return a report dict with spend_usd, cache summary, and right-sizing hints.

    Args:
        agg: Output of ``CostLedger.aggregate()``.
        rates: Mapping of ``"provider:model"`` -> USD per 1 000 tokens.
               Keys absent from this dict are treated as zero-cost (local models).
        bench: Optional bench report dict (from Task 7). When provided, a compact
               ``bench_summary`` is included. ``None`` is safe.
    """
    totals = agg["totals"]
    cache_raw = agg["cache"]
    by_model = agg["by_model"]

    # --- spend per model key ---
    spend_usd: dict[str, float] = {}
    for row in by_model:
        key = f"{row['provider']}:{row['model']}"
        tokens = row["prompt_tokens"] + row["completion_tokens"]
        spend = tokens / 1000.0 * rates.get(key, 0.0)
        spend_usd[key] = spend_usd.get(key, 0.0) + spend

    total_spend_usd = sum(spend_usd.values())
    spend_usd = {k: v for k, v in spend_usd.items() if v > 0.0}

    # --- cache summary ---
    hits = cache_raw["hits"]
    misses = cache_raw["misses"]
    hit_rate = cache_raw["hit_rate"]
    total_tokens = totals["prompt_tokens"] + totals["completion_tokens"]
    mean_tokens_per_non_cached = total_tokens / max(1, misses)
    tokens_avoided_estimate = hits * mean_tokens_per_non_cached

    cache_summary = {
        "hit_rate": hit_rate,
        "tokens_avoided_estimate": tokens_avoided_estimate,
    }

    # --- right-sizing hints ---
    hints: list[str] = []

    # Cloud spend hint
    cloud_keys = [k for k, v in spend_usd.items() if v > 0.0]
    if cloud_keys:
        models_str = ", ".join(cloud_keys)
        hints.append(
            f"cloud spend ${total_spend_usd:.4f} across {models_str}"
            " — consider routing more requests to local models"
        )

    # Low cache hit-rate hint
    total_requests = totals["requests"]
    if total_requests > 0 and hit_rate < 0.1:
        hints.append(
            f"low cache hit-rate ({hit_rate * 100:.1f}%)"
            " — consider raising the cache window or lowering the threshold"
        )

    # Roles that never escalated to cloud
    cloud_providers: set[str] = set()
    for k in cloud_keys:
        provider = k.split(":")[0]
        cloud_providers.add(provider)

    role_providers: dict[str, set[str]] = {}
    for row in by_model:
        role = row["role"]
        role_providers.setdefault(role, set()).add(row["provider"])

    for role, providers in role_providers.items():
        if not providers & cloud_providers:
            hints.append(
                f"role '{role}' never escalated to cloud"
                " — its token budget can shrink"
            )

    # --- bench summary (optional) ---
    bench_summary = None
    if bench is not None:
        try:
            bt = bench.get("best_throughput") or {}
            bl = bench.get("best_latency") or {}
            bench_summary = {
                "best_tok_per_s": bt.get("tok_per_s"),
                "best_ttft_s": bl.get("ttft_s"),
                "best_total_latency_s": bl.get("total_latency_s"),
            }
        except (TypeError, AttributeError, ValueError, KeyError):
            bench_summary = None

    report: dict = {
        "spend_usd": spend_usd,
        "total_spend_usd": total_spend_usd,
        "cache": cache_summary,
        "rightsizing": hints,
        "bench_summary": bench_summary,
    }
    return report
