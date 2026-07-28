from ragcore.chunking import ContentType, chunk_text, detect_content_type, token_count


def test_token_count_nonzero():
    assert token_count("hello world this is a test") > 0


def test_short_text_single_chunk():
    assert chunk_text("a short sentence.", chunk_size=400) == ["a short sentence."]


def test_long_text_splits():
    text = ". ".join([f"sentence number {i}" for i in range(2000)])
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert token_count(c) <= 130  # chunk_size + overlap headroom


def test_detects_markdown():
    md = "# Title\n\nSome text\n\n## Section\n\n- item one\n- item two\n"
    assert detect_content_type(md, file_path="notes.md") == ContentType.MARKDOWN


def test_drops_tiny_fragments():
    # min_chunk_size drops sub-threshold fragments so local embedders don't get
    # null-vector-producing inputs (lesson from open-notebook chunking.py).
    text = "real meaningful content here. " * 50 + "\n\n.\n\n"
    chunks = chunk_text(text, chunk_size=50, min_chunk_size=5)
    assert all(token_count(c) >= 5 for c in chunks)
