"""Test that answer_question records a ledger row when cost.enabled=True."""
import pytest

from ragcore import ask as ask_mod
from ragcore.ask import answer_question
from ragcore.chunking import token_count
from ragcore.config import Config, CostConfig, ModelRole
from ragcore.cost.ledger import CostLedger

FIXED_ANSWER = "This is the synthesized answer."


@pytest.fixture()
def cost_config(tmp_path):
    ledger_path = str(tmp_path / "cost.db")
    return Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="qwen2.5:7b-instruct"),
            "embedding": ModelRole(local_provider="ollama", local_model="nomic-embed-text"),
        },
        cost=CostConfig(enabled=True, ledger_path=ledger_path),
    )


@pytest.fixture(autouse=True)
def patch_graph(monkeypatch):
    class FakeGraph:
        async def ainvoke(self, state):
            return {"answer": FIXED_ANSWER, "citations": ["doc:0"]}

    monkeypatch.setattr(ask_mod, "_build_graph", lambda: FakeGraph())


async def test_cost_wiring_records_ledger_row(cost_config):
    # Reset the module-level ledger cache so our tmp path is picked up fresh.
    ask_mod._ledger.clear()

    result = await answer_question(
        "What is a vector database?",
        store=object(),
        config=cost_config,
        embedder_fn=None,
    )
    assert result["answer"] == FIXED_ANSWER

    ledger = CostLedger(path=cost_config.cost.ledger_path)
    agg = ledger.aggregate()

    assert agg["totals"]["requests"] == 1
    assert agg["totals"]["completion_tokens"] == token_count(FIXED_ANSWER)
    assert agg["totals"]["prompt_tokens"] == token_count("What is a vector database?")
    assert agg["cache"]["misses"] == 1
    assert agg["cache"]["hits"] == 0

    row = agg["by_model"][0]
    assert row["provider"] == "ollama"
    assert row["model"] == "qwen2.5:7b-instruct"
    assert row["role"] == "chat"


async def test_cost_disabled_no_ledger_write(tmp_path, monkeypatch):
    """When cost.enabled=False, no ledger is created or written."""
    ledger_path = str(tmp_path / "cost_disabled.db")
    config = Config(
        models={
            "chat": ModelRole(local_provider="ollama", local_model="qwen2.5:7b-instruct"),
            "embedding": ModelRole(local_provider="ollama", local_model="nomic-embed-text"),
        },
        cost=CostConfig(enabled=False, ledger_path=ledger_path),
    )
    ask_mod._ledger.clear()

    result = await answer_question(
        "Hello?", store=object(), config=config, embedder_fn=None
    )
    assert result["answer"] == FIXED_ANSWER

    import os
    assert not os.path.exists(ledger_path), "Ledger file must NOT be created when cost disabled"


async def test_budget_enforcement_raises(cost_config):
    """When enforce=True and question exceeds budget, RagcoreError is raised."""
    from ragcore.config import CostConfig
    from ragcore.errors import RagcoreError

    cfg = Config(
        models=cost_config.models,
        cost=CostConfig(
            enabled=True,
            enforce=True,
            budget_tokens=1,
            ledger_path=cost_config.cost.ledger_path,
        ),
    )
    ask_mod._ledger.clear()

    with pytest.raises(RagcoreError, match="token budget"):
        await answer_question(
            "This question has more than one token.",
            store=object(),
            config=cfg,
            embedder_fn=None,
        )
