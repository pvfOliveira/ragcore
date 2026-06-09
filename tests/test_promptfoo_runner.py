import json


def test_parse_promptfoo_results_to_metrics(tmp_path):
    """The runner maps a promptfoo JSON result file into a pass-rate metric dict."""
    from ragcore.eval.promptfoo.runner import parse_results

    out = tmp_path / "pf.json"
    out.write_text(json.dumps({
        "results": {
            "stats": {"successes": 3, "failures": 1},
            "results": [{"success": True}, {"success": True}, {"success": True}, {"success": False}],
        }
    }))
    metrics = parse_results(out)
    assert metrics["promptfoo"]["pass_rate"] == 0.75
    assert metrics["promptfoo"]["successes"] == 3.0
    assert metrics["promptfoo"]["failures"] == 1.0


def test_promptfooconfig_yaml_is_valid():
    import importlib.resources as r
    import yaml
    text = (r.files("ragcore.eval.promptfoo") / "promptfooconfig.yaml").read_text()
    doc = yaml.safe_load(text)
    assert "providers" in doc and "prompts" in doc and "tests" in doc
