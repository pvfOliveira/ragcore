"""DESIGN-ONLY: vLLM/TGI quantized serving layer (CUDA-gated holdout).

Production inference-optimization requires hardware and dependencies that are
not present on this Apple-Silicon (MPS) host:

  Continuous batching + paged KV-cache (vLLM PagedAttention):
    vLLM schedules requests against a fixed pool of KV-cache pages, enabling
    many concurrent sequences without padding waste. Throughput can be 10-24×
    higher than naive request-at-a-time serving.

  Quantized weights (AWQ / GPTQ):
    4-bit or 8-bit weight quantization (AutoAWQ, auto-gptq) shrinks the GPU
    memory footprint 2-4× so larger models fit within a single GPU and
    generation memory-bandwidth becomes the bottleneck rather than capacity.

  TGI (Text Generation Inference by HuggingFace):
    Alternative GPU serving runtime with Flash-Attention 2, continuous
    batching, and a Prometheus metrics endpoint. Shares the same CUDA
    requirement.

Why this cannot run on this host:
  * vLLM and TGI require CUDA (NVIDIA GPU) at install and runtime.
  * Apple MPS (Metal Performance Shaders) is not a supported backend for
    either framework as of mid-2026.
  * Installing the CUDA builds of vLLM/TGI on arm64/macOS fails at the
    wheel level; they are not listed as optional deps in this project.

Honest artifact: `ragcore bench` (ragcore/bench/harness.py) provides a REAL,
runnable MPS benchmark that sweeps Ollama serving params (num_ctx, num_batch,
concurrency, keep_alive) and measures latency and throughput against a local
Ollama server. Use that on this host.

The skill node for production GPU serving is not claimed on this host
and is NOT claimed as proven/experienced on this machine.
"""
from __future__ import annotations


def vllm_serve(model: str, **kwargs):
    """vLLM/TGI quantized serving entrypoint — DESIGN ONLY on this host.

    Production inference-optimization (continuous batching + paged KV-cache +
    quantized weights) requires CUDA; this MacBook (MPS) cannot run it. This
    function is intentionally gated so the claim is never silently faked. Use
    `ragcore bench` for local Ollama serving optimization on MPS instead.
    """
    import shutil

    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "vLLM/TGI serving requires CUDA (no nvidia-smi found). On this MPS "
            "host use `ragcore bench` for local Ollama serving optimization; the "
            "GPU serving layer is a documented design-only holdout."
        )
    raise NotImplementedError("GPU serving path is design-only in this repo.")
