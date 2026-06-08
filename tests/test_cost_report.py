import pytest

from ragcore.cost.report import build_report


def _agg():
    return {
        "totals": {"requests": 4, "prompt_tokens": 1000, "completion_tokens": 400},
        "cache": {"hits": 1, "misses": 3, "hit_rate": 0.25},
        "by_model": [
            {"provider": "ollama", "model": "qwen2.5:7b-instruct", "role": "chat",
             "prompt_tokens": 800, "completion_tokens": 300, "requests": 3},
            {"provider": "anthropic", "model": "claude", "role": "chat",
             "prompt_tokens": 200, "completion_tokens": 100, "requests": 1},
        ],
    }


def test_report_costs_and_rightsizing():
    rates = {"anthropic:claude": 3.0}     # USD / 1k tokens; ollama omitted -> 0
    rep = build_report(_agg(), rates, bench=None)
    assert rep["spend_usd"]["anthropic:claude"] == pytest.approx((200 + 100) / 1000 * 3.0)
    assert rep["spend_usd"].get("ollama:qwen2.5:7b-instruct", 0.0) == 0.0
    assert rep["cache"]["tokens_avoided_estimate"] >= 0
    assert isinstance(rep["rightsizing"], list)
    assert any("cloud" in h.lower() or "budget" in h.lower() or "cache" in h.lower()
               for h in rep["rightsizing"])


def test_report_handles_empty_ledger():
    empty = {"totals": {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0},
             "cache": {"hits": 0, "misses": 0, "hit_rate": 0.0}, "by_model": []}
    rep = build_report(empty, {}, bench=None)
    assert rep["spend_usd"] == {}
    assert rep["total_spend_usd"] == 0.0


def test_report_includes_bench_summary():
    agg = {"totals": {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5},
           "cache": {"hits": 0, "misses": 1, "hit_rate": 0.0}, "by_model": []}
    bench = {
        "runs": [{"num_ctx": 2048, "num_batch": 128, "concurrency": 1,
                  "ttft_s": 0.2, "total_latency_s": 1.5, "tok_per_s": 35.7}],
        "best_throughput": {"num_ctx": 2048, "num_batch": 128, "concurrency": 1,
                            "ttft_s": 0.2, "total_latency_s": 1.5, "tok_per_s": 35.7},
        "best_latency": {"num_ctx": 2048, "num_batch": 128, "concurrency": 1,
                         "ttft_s": 0.2, "total_latency_s": 1.5, "tok_per_s": 35.7},
    }
    rep = build_report(agg, {}, bench=bench)
    assert rep["bench_summary"] is not None
    assert rep["bench_summary"]["best_tok_per_s"] == pytest.approx(35.7)   # NOT 0.0
    assert rep["bench_summary"]["best_ttft_s"] == pytest.approx(0.2)
    assert rep["bench_summary"]["best_total_latency_s"] == pytest.approx(1.5)


def test_spend_accumulates_across_roles_for_same_model():
    agg = {
        "totals": {"requests": 2, "prompt_tokens": 400, "completion_tokens": 200},
        "cache": {"hits": 0, "misses": 2, "hit_rate": 0.0},
        "by_model": [
            {"provider": "anthropic", "model": "claude", "role": "chat",
             "prompt_tokens": 100, "completion_tokens": 50, "requests": 1},
            {"provider": "anthropic", "model": "claude", "role": "strategy",
             "prompt_tokens": 300, "completion_tokens": 150, "requests": 1},
        ],
    }
    rep = build_report(agg, {"anthropic:claude": 3.0}, bench=None)
    # both rows must contribute: total tokens = (100+50) + (300+150) = 600
    # spend = 600/1000 * 3.0 = 1.8
    assert rep["spend_usd"]["anthropic:claude"] == pytest.approx(1.8)
