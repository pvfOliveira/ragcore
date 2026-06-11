import pytest

from ragcore.eval.harness import compute_metrics


class _StubJudge:
    """Returns fixed per-metric scores so the harness aggregation runs offline."""

    def score(self, metric, record):  # adapt signature to your harness design
        return 0.8


def test_compute_metrics_shape():
    records = [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "a"}]
    report = compute_metrics(records, judge=_StubJudge())
    assert "ragas" in report and "trulens" in report
    assert "faithfulness" in report["ragas"]
    assert "groundedness" in report["trulens"]
    # scores are floats
    assert isinstance(report["ragas"]["faithfulness"], float)
    assert isinstance(report["trulens"]["groundedness"], float)


def test_compute_metrics_aggregates_stub_scores():
    records = [
        {"question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "g1"},
        {"question": "q2", "answer": "a2", "contexts": ["c2"], "ground_truth": "g2"},
    ]
    report = compute_metrics(records, judge=_StubJudge())
    # every metric in both families is the mean of the stub's fixed score (0.8)
    for family in ("ragas", "trulens"):
        for value in report[family].values():
            assert value == 0.8
    assert set(report["ragas"]) == {"faithfulness", "answer_relevancy", "context_precision"}
    assert set(report["trulens"]) == {"groundedness", "context_relevance", "answer_relevance"}


class _SometimesUnscorableJudge:
    """Returns None for records it cannot score (mimics an unparseable trulens
    judge response); the harness must mean over the scored records only."""

    def __init__(self, scores):
        self._scores = scores  # per-record score or None, cycled by record index

    def score(self, metric, record):
        return self._scores[int(record["question"][1:]) - 1]


def test_compute_metrics_skips_unscorable_records():
    records = [
        {"question": f"q{i}", "answer": "a", "contexts": ["c"], "ground_truth": "g"}
        for i in (1, 2, 3)
    ]
    report = compute_metrics(records, judge=_SometimesUnscorableJudge([0.4, None, 0.8]))
    # mean over the two scored records only, not over 3 with a fabricated 0.0
    for family in ("ragas", "trulens"):
        for value in report[family].values():
            assert value == pytest.approx(0.6)


def test_compute_metrics_raises_when_no_record_scorable():
    records = [
        {"question": f"q{i}", "answer": "a", "contexts": ["c"], "ground_truth": "g"}
        for i in (1, 2)
    ]
    with pytest.raises(RuntimeError, match="faithfulness"):
        compute_metrics(records, judge=_SometimesUnscorableJudge([None, None]))
