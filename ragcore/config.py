"""Load typed configuration from a TOML file."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ragcore.errors import ConfigurationError


class ModelRole(BaseModel):
    local_provider: str
    local_model: str
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None


class RoutingConfig(BaseModel):
    escalate_over_tokens: int = 100_000


class ChunkingConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 60
    min_chunk_size: int = 5


class SurrealConfig(BaseModel):
    url: str = "ws://localhost:8000/rpc"
    namespace: str = "ragcore"
    database: str = "ragcore"
    user: str = "root"
    password: str = "root"


class WorkerConfig(BaseModel):
    max_attempts: int = 3
    retry_base_seconds: float = 2


class ChatConfig(BaseModel):
    history_window: int = 10


class StoreConfig(BaseModel):
    vector_backend: str = "surreal"  # surreal | chroma | faiss | milvus | qdrant | pgvector
    # Persistence locations for the pluggable backends. Defaults live under the
    # repo `data/` dir; only the selected backend's location is used.
    chroma_path: str = "data/chroma"
    faiss_path: str = "data/faiss"
    milvus_uri: str = "data/milvus.db"
    qdrant_path: str = "data/qdrant"
    pgvector_dsn: str = "postgresql://localhost/ragcore"
    # pgvector table name reuses `collection` below.
    collection: str = "ragcore"


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "ms-marco-MiniLM-L-12-v2"
    top_k: int = 5


class CacheConfig(BaseModel):
    enabled: bool = False
    threshold: float = 0.95


class EvalConfig(BaseModel):
    judge_model: str = "ollama:qwen3:8b"
    framework: str = "ragas"          # ragas | deepeval


class LlmopsConfig(BaseModel):
    tolerance: float = 0.05          # max allowed metric regression vs baseline
    drift_threshold: float = 0.15    # cosine-distance drift that fails `drift`


class CostConfig(BaseModel):
    enabled: bool = False            # record usage into the ledger
    enforce: bool = False            # block requests over budget (else warn only)
    budget_tokens: int = 0           # 0 = no budget
    ledger_path: str = "data/cost.db"
    rates: dict[str, float] = {}     # "provider:model" -> USD per 1k tokens (local omitted = 0)


class BenchConfig(BaseModel):
    num_ctx: list[int] = [2048, 4096]
    num_batch: list[int] = [128, 256]
    concurrency: list[int] = [1, 2, 4]
    keep_alive: str = "5m"
    prompt: str = "Summarize the theory of relativity in three sentences."


class GraphConfig(BaseModel):
    enabled: bool = False
    hops: int = 1                    # traversal depth at query time


class MultimodalConfig(BaseModel):
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "mps"
    vlm_enabled: bool = False
    vlm_model: str = "moondream"   # small Ollama vision model


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    backend: str = "phoenix"          # phoenix (in-process) — live-proven backend
    otlp_endpoint: str = "http://localhost:6006/v1/traces"  # Phoenix default collector
    # Langfuse is a wired OTLP code path, NOT required to run for any proof.
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


class GatewayConfig(BaseModel):
    enabled: bool = False
    # litellm model ids, tried in order; first is primary. e.g. "ollama/qwen2.5:7b-instruct"
    fallback_chain: list[str] = []


class DspyConfig(BaseModel):
    enabled: bool = False
    compiled_path: str = "data/dspy_compiled.json"
    max_demos: int = 4                # keep the optimized task small


class QueryRewriteConfig(BaseModel):
    enabled: bool = False
    strategy: str = "multi_query"   # multi_query | hyde | decompose
    n: int = 3


class AgenticConfig(BaseModel):
    enabled: bool = False
    max_iterations: int = 2
    min_relevant: int = 2


class StructuredConfig(BaseModel):
    enabled: bool = False
    backend: str = "instructor"     # instructor | outlines
    ollama_base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b-instruct"
    outlines_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny local model on MPS


class DocaiConfig(BaseModel):
    enabled: bool = False
    parser: str = "docling"        # docling | pymupdf (fallback)


class Config(BaseModel):
    models: dict[str, ModelRole]
    routing: RoutingConfig = RoutingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    surreal: SurrealConfig = SurrealConfig()
    worker: WorkerConfig = WorkerConfig()
    chat: ChatConfig = ChatConfig()
    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()
    agentic: AgenticConfig = AgenticConfig()
    structured: "StructuredConfig" = StructuredConfig()
    docai: "DocaiConfig" = DocaiConfig()
    store: StoreConfig = StoreConfig()
    rerank: RerankConfig = RerankConfig()
    cache: CacheConfig = CacheConfig()
    eval: EvalConfig = EvalConfig()
    llmops: LlmopsConfig = LlmopsConfig()
    cost: CostConfig = CostConfig()
    bench: BenchConfig = BenchConfig()
    graph: GraphConfig = GraphConfig()
    multimodal: MultimodalConfig = MultimodalConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    gateway: GatewayConfig = GatewayConfig()
    dspy: DspyConfig = DspyConfig()


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(
            f"Config not found at {path}. Copy config.example.toml to config.toml."
        )
    data = tomllib.loads(path.read_text())
    models = {name: ModelRole(**role) for name, role in data.get("models", {}).items()}
    if "chat" not in models or "embedding" not in models:
        raise ConfigurationError("config must define [models.chat] and [models.embedding].")
    return Config(
        models=models,
        routing=RoutingConfig(**data.get("routing", {})),
        chunking=ChunkingConfig(**data.get("chunking", {})),
        surreal=SurrealConfig(**data.get("surreal", {})),
        worker=WorkerConfig(**data.get("worker", {})),
        chat=ChatConfig(**data.get("chat", {})),
        query_rewrite=QueryRewriteConfig(**data.get("query_rewrite", {})),
        agentic=AgenticConfig(**data.get("agentic", {})),
        structured=StructuredConfig(**data.get("structured", {})),
        docai=DocaiConfig(**data.get("docai", {})),
        store=StoreConfig(**data.get("store", {})),
        rerank=RerankConfig(**data.get("rerank", {})),
        cache=CacheConfig(**data.get("cache", {})),
        eval=EvalConfig(**data.get("eval", {})),
        llmops=LlmopsConfig(**data.get("llmops", {})),
        cost=CostConfig(**data.get("cost", {})),
        bench=BenchConfig(**data.get("bench", {})),
        graph=GraphConfig(**data.get("graph", {})),
        multimodal=MultimodalConfig(**data.get("multimodal", {})),
        observability=ObservabilityConfig(**data.get("observability", {})),
        gateway=GatewayConfig(**data.get("gateway", {})),
        dspy=DspyConfig(**data.get("dspy", {})),
    )
