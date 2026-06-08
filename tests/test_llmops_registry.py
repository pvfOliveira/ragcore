import pytest

from ragcore.config import SurrealConfig
from ragcore.llmops.registry import RunRegistry
from ragcore.store import Store


@pytest.fixture()
async def surreal_config(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()      # creates eval_run (+ all) tables
    yield cfg


async def test_record_and_get(surreal_config):
    reg = RunRegistry(surreal_config)
    run_id = await reg.record(
        metrics={"ragas": {"faithfulness": 0.8}},
        config_snapshot={"models": {"chat": "qwen2.5:7b-instruct"}},
        dataset="eval/dataset.jsonl", tag="baseline",
    )
    got = await reg.get(run_id)
    assert got["metrics"]["ragas"]["faithfulness"] == 0.8
    assert got["config_snapshot"]["models"]["chat"] == "qwen2.5:7b-instruct"
    assert got["tag"] == "baseline"
    listed = await reg.list_runs()
    assert any(r["id"] == run_id for r in listed)


async def test_get_by_tag_returns_latest(surreal_config):
    reg = RunRegistry(surreal_config)
    await reg.record(metrics={"m": {"x": 0.1}}, config_snapshot={}, tag="prod")
    second = await reg.record(metrics={"m": {"x": 0.2}}, config_snapshot={}, tag="prod")
    resolved = await reg.resolve("prod")
    assert resolved["id"] == second
