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


class Config(BaseModel):
    models: dict[str, ModelRole]
    routing: RoutingConfig = RoutingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    surreal: SurrealConfig = SurrealConfig()
    worker: WorkerConfig = WorkerConfig()


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
    )
