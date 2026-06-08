"""MPS Ollama serving benchmark harness.

Sweeps the cartesian product of num_ctx × num_batch × concurrency, calls
the provided client for each combination, and returns a report with per-run
timing/throughput and the best-throughput / best-latency highlights.

Concurrency handling: for concurrency > 1 this module fires N generate calls
concurrently via a ThreadPoolExecutor and records:
  - tok_per_s = sum of individual tok_per_s across the N workers (aggregate
    throughput from the server's perspective)
  - total_latency_s = max across workers (wall-clock latency of the slowest
    concurrent call)
  - ttft_s = min across workers (fastest first-token of any concurrent call)

The fake _FakeClient in tests is synchronous, so threading is safe there too.
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Any


def _tok_per_s(eval_count: int, eval_duration_s: float) -> float:
    return eval_count / eval_duration_s if eval_duration_s > 0 else 0.0


def run_bench(
    client: Any,
    *,
    num_ctx: list[int],
    num_batch: list[int],
    concurrency: list[int],
    keep_alive: str,
    prompt: str,
) -> dict:
    """Sweep num_ctx × num_batch × concurrency, return a structured report.

    Each entry in ``report["runs"]`` contains:
      num_ctx, num_batch, concurrency, ttft_s, total_latency_s, tok_per_s

    ``report["best_throughput"]`` is the run with the highest tok_per_s.
    ``report["best_latency"]`` is the run with the lowest total_latency_s.
    """
    runs: list[dict] = []

    for ctx, batch, conc in product(num_ctx, num_batch, concurrency):
        if conc <= 1:
            result = client.generate(
                num_ctx=ctx,
                num_batch=batch,
                keep_alive=keep_alive,
                prompt=prompt,
            )
            eval_count = result["eval_count"]
            eval_duration_s = result["eval_duration_s"]
            tok_per_s = _tok_per_s(eval_count, eval_duration_s)
            run = {
                "num_ctx": ctx,
                "num_batch": batch,
                "concurrency": conc,
                "ttft_s": result["ttft_s"],
                "total_latency_s": result["total_latency_s"],
                "tok_per_s": tok_per_s,
            }
        else:
            # Fire conc concurrent generate calls; aggregate throughput + max latency
            def _call(_ctx=ctx, _batch=batch):
                return client.generate(
                    num_ctx=_ctx,
                    num_batch=_batch,
                    keep_alive=keep_alive,
                    prompt=prompt,
                )

            with ThreadPoolExecutor(max_workers=conc) as pool:
                futures = [pool.submit(_call) for _ in range(conc)]
                results = [f.result() for f in as_completed(futures)]

            agg_tok_per_s = sum(
                _tok_per_s(r["eval_count"], r["eval_duration_s"]) for r in results
            )
            max_latency = max(r["total_latency_s"] for r in results)
            min_ttft = min(r["ttft_s"] for r in results)
            run = {
                "num_ctx": ctx,
                "num_batch": batch,
                "concurrency": conc,
                "ttft_s": min_ttft,
                "total_latency_s": max_latency,
                "tok_per_s": agg_tok_per_s,
            }

        runs.append(run)

    if not runs:
        raise ValueError(
            "run_bench: no parameter combinations to run "
            "(num_ctx, num_batch, and concurrency must be non-empty)"
        )

    best_throughput = max(runs, key=lambda r: r["tok_per_s"])
    best_latency = min(runs, key=lambda r: r["total_latency_s"])

    return {
        "runs": runs,
        "best_throughput": best_throughput,
        "best_latency": best_latency,
    }


class OllamaBenchClient:
    """Real Ollama client used by ``ragcore bench``.

    Calls POST /api/generate with stream=false and converts Ollama's
    nanosecond timing fields to seconds.

    Ollama /api/generate response fields (verified against docs 2026-06):
      total_duration      (int, ns) — total wall-clock time for the request
      prompt_eval_duration (int, ns) — time spent evaluating the prompt (≈ TTFT)
      eval_count          (int)     — number of output tokens generated
      eval_duration       (int, ns) — time spent generating output tokens
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        num_ctx: int,
        num_batch: int,
        keep_alive: str,
        prompt: str,
    ) -> dict:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "num_ctx": num_ctx,
                "num_batch": num_batch,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        total_duration_ns: int = data.get("total_duration", 0)
        prompt_eval_duration_ns: int = data.get("prompt_eval_duration", 0)
        eval_count: int = data.get("eval_count", 0)
        eval_duration_ns: int = data.get("eval_duration", 0)

        return {
            "total_latency_s": total_duration_ns / 1e9,
            "ttft_s": prompt_eval_duration_ns / 1e9,
            "eval_count": eval_count,
            "eval_duration_s": eval_duration_ns / 1e9,
        }
