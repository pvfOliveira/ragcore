def test_set_gen_ai_attributes_sets_stable_keys():
    from ragcore.observability.otel import set_gen_ai_attributes

    class FakeSpan:
        def __init__(self):
            self.attrs = {}
        def set_attribute(self, key, value):
            self.attrs[key] = value

    span = FakeSpan()
    set_gen_ai_attributes(
        span, system="ollama", operation="chat",
        model="qwen2.5:7b-instruct", input_tokens=12, output_tokens=34,
    )
    assert span.attrs["gen_ai.system"] == "ollama"
    assert span.attrs["gen_ai.operation.name"] == "chat"
    assert span.attrs["gen_ai.request.model"] == "qwen2.5:7b-instruct"
    assert span.attrs["gen_ai.usage.input_tokens"] == 12
    assert span.attrs["gen_ai.usage.output_tokens"] == 34


def test_set_gen_ai_attributes_omits_none_token_counts():
    from ragcore.observability.otel import set_gen_ai_attributes

    class FakeSpan:
        def __init__(self):
            self.attrs = {}
        def set_attribute(self, key, value):
            self.attrs[key] = value

    span = FakeSpan()
    set_gen_ai_attributes(span, system="ollama", operation="chat", model="m")
    assert "gen_ai.usage.input_tokens" not in span.attrs
    assert "gen_ai.usage.output_tokens" not in span.attrs
