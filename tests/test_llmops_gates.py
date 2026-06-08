import pytest

from ragcore.llmops.gates import check_drift, check_gate


def test_gate_passes_within_tolerance():
    baseline = {"ragas": {"faithfulness": 0.80}, "trulens": {"groundedness": 0.70}}
    current  = {"ragas": {"faithfulness": 0.78}, "trulens": {"groundedness": 0.71}}
    ok, regressions = check_gate(current, baseline, tolerance=0.05)
    assert ok is True
    assert regressions == []

def test_gate_fails_on_regression():
    baseline = {"ragas": {"faithfulness": 0.80}}
    current  = {"ragas": {"faithfulness": 0.60}}
    ok, regressions = check_gate(current, baseline, tolerance=0.05)
    assert ok is False
    assert regressions[0]["metric"] == "ragas.faithfulness"
    assert regressions[0]["delta"] == pytest.approx(-0.20)

def test_drift_flags_far_centroid():
    drifted, distance = check_drift([1.0, 0.0], [0.0, 1.0], threshold=0.15)
    assert drifted is True
    assert distance == pytest.approx(1.0)

def test_drift_ok_near_centroid():
    drifted, distance = check_drift([1.0, 0.0], [0.99, 0.01], threshold=0.15)
    assert drifted is False
