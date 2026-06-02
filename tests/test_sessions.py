import pytest

from ragcore.config import SurrealConfig
from ragcore.sessions import SessionStore
from ragcore.store import Store


@pytest.fixture()
async def sessions(surreal_url):
    cfg = SurrealConfig(url=surreal_url, namespace="ragcore", database="ragcore",
                        user="root", password="root")
    await Store(cfg).init_schema()
    return SessionStore(cfg)


async def test_create_add_and_get_history(sessions):
    sid = await sessions.create_session(title="Chat 1")
    assert sid.startswith("chat_session:")
    await sessions.add_message(sid, "user", "hello")
    await sessions.add_message(sid, "assistant", "hi there", citations=["source:1"])
    hist = await sessions.get_history(sid)
    assert [m["role"] for m in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "hello"
    assert hist[1]["citations"] == ["source:1"]


async def test_get_history_limit_keeps_most_recent(sessions):
    sid = await sessions.create_session()
    for i in range(5):
        await sessions.add_message(sid, "user", f"m{i}")
    hist = await sessions.get_history(sid, limit=2)
    assert [m["content"] for m in hist] == ["m3", "m4"]


async def test_list_sessions_counts_messages(sessions):
    s1 = await sessions.create_session(title="A")
    await sessions.add_message(s1, "user", "x")
    s2 = await sessions.create_session(title="B")
    by_id = {s["id"]: s for s in await sessions.list_sessions()}
    assert by_id[s1]["messages"] == 1 and by_id[s1]["title"] == "A"
    assert by_id[s2]["messages"] == 0


async def test_rename_session(sessions):
    sid = await sessions.create_session(title="old")
    assert await sessions.rename_session(sid, "new") is True
    titles = {s["id"]: s["title"] for s in await sessions.list_sessions()}
    assert titles[sid] == "new"
    assert await sessions.rename_session("chat_session:nope", "x") is False
    assert await sessions.rename_session("not-an-id", "x") is False


async def test_delete_session_cascades(sessions):
    sid = await sessions.create_session(title="d")
    await sessions.add_message(sid, "user", "hi")
    assert await sessions.delete_session(sid) is True
    assert sid not in {s["id"] for s in await sessions.list_sessions()}
    assert await sessions.get_history(sid) == []          # messages cascaded away
    assert await sessions.delete_session(sid) is False     # already gone
    assert await sessions.delete_session("not-an-id") is False
