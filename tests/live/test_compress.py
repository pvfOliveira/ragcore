"""LIVE: real LLMLingua-2 compression on this host — measured ratio + the
compressed context still answers correctly over Ollama. First run downloads
the LLMLingua-2 encoder (network). Compression is query-agnostic (LLMLingua-2
API has no question param). Records RunRegistry tag compress-v4."""
import asyncio
import copy
import socket

import pytest

pytestmark = pytest.mark.live

pytest.importorskip("llmlingua")


def _squash(s: str) -> str:
    # LLMLingua-2's BERT detokenizer renders "11.6" as "11. 6" (space after
    # the wordpiece split); compare with whitespace removed.
    return s.replace(" ", "")


def _network_up(host="huggingface.co", port=443) -> bool:
    try:
        socket.create_connection((host, port), timeout=2).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _network_up(), reason="network needed for LLMLingua-2 model download")
def test_real_compression_ratio_and_answer_quality(surreal_url):
    from ragcore.compress import compress_context
    from ragcore.config import SurrealConfig, load_config

    cfg = copy.deepcopy(load_config())
    cfg.surreal = SurrealConfig(url=surreal_url, namespace="ragcore_cmp", database="ragcore_cmp")
    cfg.compression.enabled = True
    cfg.compression.rate = 0.5

    filler = ("It is worth noting, generally speaking and in most practical "
              "circumstances that have been observed over the years, that ")
    fact = "The Mont Blanc tunnel connects France to Italy and is 11.6 km long."
    chunks = [
        {"id": "c1", "content": filler * 6 + fact + " " + filler * 6,
         "source": "s1", "score": 0.9},
        {"id": "c2", "content": filler * 12, "source": "s2", "score": 0.5},
    ]

    out, stats = compress_context(chunks, "How long is the Mont Blanc tunnel?", cfg)
    assert stats["enabled"] is True
    assert stats["ratio"] < 0.8, f"no real compression: ratio={stats['ratio']}"
    assert "11.6" in _squash(out[0]["content"]), "compression dropped the key fact"

    # the compressed context still answers correctly over Ollama
    from ragcore.ask import _build_chat, _clean

    async def _answer():
        chat = _build_chat(cfg, content="q")
        ctx = "\n".join(c["content"] for c in out)
        msg = await chat.ainvoke(
            f"Answer from the context only.\nContext:\n{ctx}\n\n"
            "Question: How long is the Mont Blanc tunnel?")
        return _clean(msg.content)

    answer = asyncio.run(_answer())
    assert "11.6" in _squash(answer)

    async def _record():
        from ragcore.llmops.registry import RunRegistry
        return await RunRegistry(cfg.surreal).record(
            metrics={"compression": {"ratio": stats["ratio"],
                                     "orig_chars": float(stats["orig_chars"]),
                                     "compressed_chars": float(stats["compressed_chars"]),
                                     "answer_correct": 1.0}},
            config_snapshot={"compression_model": cfg.compression.model,
                             "rate": cfg.compression.rate},
            dataset=None, tag="compress-v4", gate_passed=True)

    run_id = asyncio.run(_record())
    assert run_id and ":" in run_id
