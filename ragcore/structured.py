"""Structured (schema-typed) generation over local Ollama.

Backends: instructor (Ollama OpenAI-compatible endpoint, JSON mode) and outlines
(constrained decoding over a tiny local transformers model on MPS). Opt-in."""
from __future__ import annotations

from pathlib import Path
from typing import Type

from ai_prompter import Prompter
from pydantic import BaseModel

_PROMPT_DIR = str(Path(__file__).parent / "prompts")


def _render(template: str, data: dict) -> str:
    return Prompter(prompt_template=template, prompt_dir=_PROMPT_DIR).render(data=data)


def _instructor_client(config):
    import instructor                       # lazy — optional extra
    from openai import OpenAI

    cfg = config.structured
    base = OpenAI(base_url=cfg.ollama_base_url, api_key="ollama")
    return instructor.from_openai(base, mode=instructor.Mode.JSON), cfg.model


def _instructor_create(client, model: str, prompt: str, schema: Type[BaseModel]):
    return client.chat.completions.create(
        model=model, response_model=schema,
        messages=[{"role": "user", "content": prompt}],
        max_retries=2,
    )


async def generate_structured(prompt: str, schema: Type[BaseModel], config) -> BaseModel:
    cfg = config.structured
    if not cfg.enabled:
        raise RuntimeError("structured generation is disabled ([structured].enabled=false)")
    if cfg.backend == "instructor":
        client, model = _instructor_client(config)
        return _instructor_create(client, model, prompt, schema)
    raise ValueError(f"Unknown structured backend {cfg.backend!r}")
