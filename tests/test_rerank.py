"""Tests for the FlashRank reranking stage."""
from ragcore.rerank import rerank


def test_rerank_reorders_by_scorer():
    results = [{"id": "a", "content": "off topic"}, {"id": "b", "content": "the answer is here"}]

    def scorer(query, passages):  # passages: list[str]; returns list[float] aligned
        return [1.0 if "answer" in p else 0.0 for p in passages]

    out = rerank("what is the answer", results, top_k=2, _scorer=scorer)
    assert out[0]["id"] == "b"
    assert out[1]["id"] == "a"


def test_rerank_truncates_to_top_k():
    results = [{"id": str(i), "content": f"c{i}"} for i in range(5)]

    def scorer(q, passages):
        return [float(i) for i in range(len(passages))]  # ascending

    out = rerank("q", results, top_k=2, _scorer=scorer)
    assert len(out) == 2 and out[0]["id"] == "4"  # highest score first


def test_rerank_adds_rerank_score():
    results = [{"id": "x", "content": "hello", "score": 0.9}]

    def scorer(q, passages):
        return [0.75]

    out = rerank("hi", results, top_k=1, _scorer=scorer)
    assert out[0]["rerank_score"] == 0.75
    assert out[0]["score"] == 0.9  # original field preserved


def test_rerank_empty_results():
    out = rerank("q", [], top_k=5, _scorer=lambda q, p: [])
    assert out == []
