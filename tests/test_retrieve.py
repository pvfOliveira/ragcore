from ragcore.retrieve import reciprocal_rank_fusion


def test_rrf_merges_and_ranks():
    vector = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    text = [{"id": "b"}, {"id": "d"}, {"id": "a"}]
    fused = reciprocal_rank_fusion([vector, text], key="id", k=60)
    ids = [r["id"] for r in fused]
    # 'a' (ranks 1 and 3) and 'b' (ranks 2 and 1) appear in both -> top two.
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids) == {"a", "b", "c", "d"}
    assert "score" in fused[0]


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []], key="id") == []
