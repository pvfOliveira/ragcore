import pytest

from ragcore.config import SurrealConfig
from ragcore.errors import RagcoreError
from ragcore.llmops.deploy import DeploymentStore
from ragcore.llmops.registry import RunRegistry
from ragcore.store import Store


@pytest.fixture()
async def surreal_config(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()
    yield cfg


async def test_promote_then_rollback(surreal_config):
    reg = RunRegistry(surreal_config)
    r1 = await reg.record(metrics={"m": {"x": 0.7}}, config_snapshot={}, gate_passed=True)
    r2 = await reg.record(metrics={"m": {"x": 0.9}}, config_snapshot={}, gate_passed=True)
    dep = DeploymentStore(surreal_config)
    await dep.promote(r1)
    assert (await dep.current()) == r1
    await dep.promote(r2)
    assert (await dep.current()) == r2
    assert (await dep.rollback()) == r1
    assert (await dep.current()) == r1


async def test_promote_refuses_failing_run(surreal_config):
    reg = RunRegistry(surreal_config)
    bad = await reg.record(metrics={"m": {"x": 0.1}}, config_snapshot={}, gate_passed=False)
    dep = DeploymentStore(surreal_config)
    with pytest.raises(RagcoreError):
        await dep.promote(bad, require_gate=True)


async def test_rollback_with_no_history_returns_none(surreal_config):
    dep = DeploymentStore(surreal_config)
    assert (await dep.rollback()) is None


async def test_rollback_two_steps(surreal_config):
    reg = RunRegistry(surreal_config)
    r1 = await reg.record(metrics={"m": {"x": 0.5}}, config_snapshot={}, gate_passed=True)
    r2 = await reg.record(metrics={"m": {"x": 0.6}}, config_snapshot={}, gate_passed=True)
    r3 = await reg.record(metrics={"m": {"x": 0.7}}, config_snapshot={}, gate_passed=True)
    dep = DeploymentStore(surreal_config)
    await dep.promote(r1)
    await dep.promote(r2)
    await dep.promote(r3)
    assert (await dep.current()) == r3
    assert (await dep.rollback()) == r2
    assert (await dep.rollback()) == r1
    assert (await dep.current()) == r1
