import pytest
from ragcore.errors import RagcoreError


def test_weaviate_is_documented_holdout():
    from ragcore.vectorstores.weaviate_store import WeaviateStore
    with pytest.raises(RagcoreError) as e:
        WeaviateStore(path="x", collection="t")
    assert "Docker" in str(e.value) and "holdout" in str(e.value).lower()
