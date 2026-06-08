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
    class _VaryingClient:
        def generate(self, *, num_ctx, num_batch, keep_alive, prompt):
            return {"total_latency_s": num_ctx * 0.001, "ttft_s": 0.1,
                    "eval_count": 100, "eval_duration_s": 1.0}
    report = run_bench(_VaryingClient(),
                       num_ctx=[4096, 2048], num_batch=[128], concurrency=[1],
                       keep_alive="5m", prompt="hi")
    assert len(report["runs"]) == 2
    assert report["best_latency"]["num_ctx"] == 2048   # lower ctx -> lower latency

def test_run_bench_concurrency_aggregates_across_workers():
    calls = []
    class _CountingClient:
        def generate(self, *, num_ctx, num_batch, keep_alive, prompt):
            calls.append((num_ctx, num_batch))
            return {"total_latency_s": 2.0, "ttft_s": 0.3,
                    "eval_count": 100, "eval_duration_s": 1.0}   # 100 tok/s each
    report = run_bench(_CountingClient(),
                       num_ctx=[2048], num_batch=[128], concurrency=[3],
                       keep_alive="5m", prompt="hi")
    assert len(calls) == 3                              # 3 concurrent generate calls fired
    run = report["runs"][0]
    assert run["concurrency"] == 3
    assert run["tok_per_s"] == pytest.approx(300.0)     # sum of 3 workers' 100 tok/s
    assert run["total_latency_s"] == pytest.approx(2.0) # max worker latency
    assert run["ttft_s"] == pytest.approx(0.3)          # min worker ttft

def test_serving_stub_requires_cuda():
    from ragcore.bench.serving import vllm_serve
    with pytest.raises(RuntimeError, match="CUDA"):
        vllm_serve(model="x")
