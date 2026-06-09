"""DSPy prompt optimization for ONE small bounded task: the ask `strategy` step
(question -> list of search terms). `compile_strategy` compiles a DSPy program
with BootstrapFewShot over Ollama and saves the resulting prompt text to a JSON
artifact; ragcore's ask graph loads it when [dspy] enabled. Heavy dspy imports
stay inside compile_strategy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def load_compiled_strategy(path: str) -> Optional[str]:
    """Return the compiled strategy prompt text, or None if no artifact."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("strategy_prompt")


def save_compiled_strategy(path: str, prompt: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"strategy_prompt": prompt}, indent=2))


def compile_strategy(config: Any, dataset_path: str | None = None) -> str:
    """Compile the strategy prompt with DSPy BootstrapFewShot over Ollama and
    persist it to config.dspy.compiled_path. Returns the compiled prompt text.
    Validated live in Task 12."""
    import dspy

    from ragcore.eval.harness import _DEFAULT_DATASET, _litellm_model, _load_dataset

    model_id = _litellm_model(config.eval.judge_model)  # "ollama/<model>"
    lm = dspy.LM(
        f"ollama_chat/{model_id.split('/', 1)[1]}",
        api_base="http://localhost:11434",
    )  # verify form live (Task 12)
    dspy.configure(lm=lm)

    class StrategySig(dspy.Signature):
        """Decompose a question into up to N search terms (JSON list)."""

        question: str = dspy.InputField()
        searches: str = dspy.OutputField(desc='JSON like {"searches": [...]}')

    program = dspy.ChainOfThought(StrategySig)
    dataset = _load_dataset(Path(dataset_path) if dataset_path else _DEFAULT_DATASET)
    trainset = [
        dspy.Example(
            question=r["question"],
            searches=json.dumps({"searches": [r["question"]]}),
        ).with_inputs("question")
        for r in dataset[: max(1, config.dspy.max_demos)]
    ]

    def _metric(example, pred, trace=None):
        try:
            json.loads(pred.searches)
            return 1.0
        except Exception:
            return 0.0

    optimizer = dspy.BootstrapFewShot(
        metric=_metric, max_bootstrapped_demos=config.dspy.max_demos
    )
    compiled = optimizer.compile(program, trainset=trainset)

    # Extract the bootstrapped few-shot demos from the (single) optimized
    # predictor. dspy 3.2.x stores them as ``dspy.Example`` objects on
    # ``predictor.demos`` (each exposes ``.get``); ``predictors()`` is the
    # canonical accessor and avoids depending on the program's attribute name.
    predictors = compiled.predictors()
    predictor = predictors[0] if predictors else compiled
    demos = list(getattr(predictor, "demos", []) or [])
    demo_text = "\n".join(
        f"Q: {d.get('question', '')}\nA: {d.get('searches', '')}"
        for d in demos
        if d.get("question")
    )
    # The optimizer may also tune the signature instruction; prefer it.
    sig = getattr(predictor, "signature", None)
    instruction = (
        getattr(sig, "instructions", None)
        or "Decompose a question into up to N search terms (JSON list)."
    )
    prompt = (
        f"{instruction}\n"
        "Return up to {{max_searches}} search terms as JSON "
        '{"searches": [...]}.\n'
        + (f"Examples:\n{demo_text}\n" if demo_text else "")
        + "Question: {{question}}"
    )
    save_compiled_strategy(config.dspy.compiled_path, prompt)
    return prompt
