"""Centralised configuration for the CodeDebug OpenEnv project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvConfig:
    """Environment-level settings."""
    task_dir: str = "tasks"
    max_steps: int = 10
    executor_timeout: int = 10


@dataclass(frozen=True)
class ServerConfig:
    """HTTP server settings."""
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", "7860"))
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    log_level: str = os.environ.get("LOG_LEVEL", "info")


@dataclass(frozen=True)
class InferenceConfig:
    """LLM inference settings."""
    api_base_url: str = os.environ.get(
        "API_BASE_URL", "https://api-inference.huggingface.co/v1"
    )
    model_name: str = os.environ.get(
        "MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    hf_token: str = os.environ.get("HF_TOKEN", "")
    env_url: str = os.environ.get("ENV_URL", "http://localhost:7860")
    temperature: float = 0.2
    max_tokens: int = 2048
    max_retries: int = 3


# Singletons
ENV_CONFIG = EnvConfig()
SERVER_CONFIG = ServerConfig()
INFERENCE_CONFIG = InferenceConfig()
