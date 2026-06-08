import pytest

from ragcore.bench.harness import run_bench


class _FakeClient:
    def generate(self, *, num_ctx, num_batch, keep_alive, prompt):
        # deterministic fake timings; throughput scales with num_batch
        return {"total_latency_s": 1.0, "ttft_s": 0.2,
                "eval_count": num_batch, "eval_duration_s": 1.0}

def test_run_bench_sweeps_params_and_builds_report():
    report = run_bench(_FakeClient(),
                       num_ctx=[2048], num_batch=[128, 256], concurrency=[1],
                       keep_alive="5m", prompt="hi")
    assert len(report["runs"]) == 2                       # one per (ctx, batch, concurrency)
    assert report["best_throughput"]["num_batch"] == 256  # higher batch -> higher tok/s
    assert "tok_per_s" in report["runs"][0]

def test_run_bench_picks_best_latency():
    report = run_bench(_FakeClient(),
                       num_ctx=[2048, 4096], num_batch=[128], concurrency=[1],
                       keep_alive="5m", prompt="hi")
    assert len(report["runs"]) == 2
    assert "best_latency" in report

def test_serving_stub_requires_cuda():
    from ragcore.bench.serving import vllm_serve
    with pytest.raises(RuntimeError, match="CUDA"):
        vllm_serve(model="x")
