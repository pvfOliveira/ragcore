"""Tests for embedding retry behavior and mean-pooling numerics."""
import numpy as np
import pytest

from ragcore import embedding as emb_mod
from ragcore.embedding import _mean_pool, generate_embeddings


class _FlakyEmbedder:
    """Raises on the first `fail_times` calls, then returns embeddings."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    async def aembed(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient provider error")
        return [[1.0, 2.0] for _ in texts]


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    # Keep retries instant so the tests don't wait the real 2s backoff.
    monkeypatch.setattr(emb_mod, "_RETRY_DELAY", 0)


async def test_retry_succeeds_after_transient_failures(monkeypatch):
    # Fails twice, succeeds on the 3rd (== _MAX_RETRIES) attempt.
    embedder = _FlakyEmbedder(fail_times=2)
    monkeypatch.setattr(emb_mod, "_get_embedder", lambda cfg: embedder)

    vecs = await generate_embeddings(["a", "b"], config=None)

    assert vecs == [[1.0, 2.0], [1.0, 2.0]]
    assert embedder.calls == 3


async def test_retry_exhausted_reraises(monkeypatch):
    embedder = _FlakyEmbedder(fail_times=99)
    monkeypatch.setattr(emb_mod, "_get_embedder", lambda cfg: embedder)

    with pytest.raises(RuntimeError, match="transient"):
        await generate_embeddings(["a"], config=None)

    assert embedder.calls == emb_mod._MAX_RETRIES


def test_mean_pool_orthogonal_unit_vectors_are_equally_weighted():
    pooled = _mean_pool([[1.0, 0.0], [0.0, 1.0]])

    # Mean of two orthogonal unit vectors has equal components and unit norm.
    assert pooled[0] == pytest.approx(pooled[1])
    assert float(np.linalg.norm(pooled)) == pytest.approx(1.0)


def test_mean_pool_normalizes_before_averaging():
    # Both inputs point the same direction; magnitude must not skew the result.
    pooled = _mean_pool([[3.0, 4.0], [3.0, 4.0]])  # each normalizes to [0.6, 0.8]

    assert pooled[0] == pytest.approx(0.6)
    assert pooled[1] == pytest.approx(0.8)
