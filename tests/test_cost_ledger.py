import pytest

from ragcore.cost.ledger import CostLedger, over_budget


def test_record_and_aggregate(tmp_path):
    led = CostLedger(path=str(tmp_path / "cost.db"))
    led.record(role="chat", provider="ollama", model="qwen2.5:7b-instruct",
               prompt_tokens=100, completion_tokens=50, cached=False)
    led.record(role="chat", provider="anthropic", model="claude",
               prompt_tokens=200, completion_tokens=80, cached=False)
    led.record(role="chat", provider="ollama", model="qwen2.5:7b-instruct",
               prompt_tokens=0, completion_tokens=0, cached=True)
    agg = led.aggregate()
    assert agg["totals"]["requests"] == 3
    assert agg["totals"]["prompt_tokens"] == 300
    assert agg["totals"]["completion_tokens"] == 130
    assert agg["cache"]["hits"] == 1
    assert agg["cache"]["misses"] == 2
    assert agg["cache"]["hit_rate"] == pytest.approx(1/3)
    by = {(r["provider"], r["model"]): r for r in agg["by_model"]}
    assert by[("anthropic", "claude")]["completion_tokens"] == 80
    assert by[("anthropic", "claude")]["requests"] == 1

def test_over_budget():
    assert over_budget(prompt_tokens=3000, budget_tokens=2000) is True
    assert over_budget(prompt_tokens=1000, budget_tokens=2000) is False
    assert over_budget(prompt_tokens=9999, budget_tokens=0) is False   # 0 = no budget
