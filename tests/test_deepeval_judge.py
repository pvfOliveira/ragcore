def test_deepeval_judge_honors_score_protocol():
    """DeepEvalJudge.score(metric, record) returns a float for each ragas metric
    name, delegating to injected metric objects (no live LLM)."""
    from ragcore.eval.deepeval_judge import DeepEvalJudge

    class _FakeMetric:
        def __init__(self, value): self._value = value
        def measure(self, test_case): self.score = self._value

    judge = DeepEvalJudge.__new__(DeepEvalJudge)   # bypass heavy __init__
    judge._metrics = {
        "faithfulness": _FakeMetric(0.9),
        "answer_relevancy": _FakeMetric(0.8),
        "context_precision": _FakeMetric(0.7),
    }
    rec = {"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}
    assert judge.score("faithfulness", rec) == 0.9
    assert judge.score("answer_relevancy", rec) == 0.8


def test_deepeval_judge_returns_zero_for_unknown_metric():
    """DeepEval populates only the ragas family; unknown (trulens) metrics → 0.0."""
    from ragcore.eval.deepeval_judge import DeepEvalJudge
    judge = DeepEvalJudge.__new__(DeepEvalJudge)
    judge._metrics = {}
    assert judge.score("groundedness", {"question": "q", "answer": "a", "contexts": ["c"]}) == 0.0


def test_build_judge_for_framework_selects_deepeval(monkeypatch):
    """_build_judge_for_framework dispatches on config.eval.framework: 'deepeval'
    builds DeepEvalJudge, anything else builds the Ollama Ragas/TruLens judge."""
    import ragcore.eval.harness as harness
    import ragcore.eval.deepeval_judge as de

    class _Cfg:
        class eval:  # noqa: A003 - mirrors EvalConfig attr access
            framework = "deepeval"
            judge_model = "ollama:qwen2.5:7b-instruct"

    sentinel = object()
    monkeypatch.setattr(de, "DeepEvalJudge", lambda config: sentinel)
    assert harness._build_judge_for_framework(_Cfg()) is sentinel

    _Cfg.eval.framework = "ragas"
    monkeypatch.setattr(harness, "_OllamaJudge", lambda config: "ollama-judge")
    assert harness._build_judge_for_framework(_Cfg()) == "ollama-judge"
