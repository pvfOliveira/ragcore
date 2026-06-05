from ragcore.cache import SemanticCache


def test_cache_hit_on_paraphrase(tmp_path):
    c = SemanticCache(path=str(tmp_path / "cache.db"), threshold=0.95)
    assert c.get([1.0, 0.0]) is None                       # empty → miss
    c.put("orig query", [1.0, 0.0], answer="A1", sources=["s1"])
    hit = c.get([0.99, 0.01])                               # near-identical → hit
    assert hit is not None and hit["answer"] == "A1" and hit["sources"] == ["s1"]
    assert c.get([0.0, 1.0]) is None                        # orthogonal → miss


def test_cache_persists(tmp_path):
    p = str(tmp_path / "cache.db")
    SemanticCache(p, 0.95).put("q", [1.0, 0.0], answer="A", sources=[])
    assert SemanticCache(p, 0.95).get([1.0, 0.0])["answer"] == "A"   # reopened db
