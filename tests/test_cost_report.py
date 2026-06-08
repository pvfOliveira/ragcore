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
