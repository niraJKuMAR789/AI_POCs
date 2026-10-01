from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUILL_", env_file=".env", extra="ignore")

    provider: Literal["nvidia", "openai_compatible", "offline"] = "nvidia"
    nvidia_api_key: SecretStr | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    base_url: str = "https://integrate.api.nvidia.com/v1"
    # Used for provider=openai_compatible, e.g. a vLLM server hosting the fine-tuned LoRA model.
    api_key: SecretStr | None = None

    sql_model: str = "nvidia/nemotron-3-super-120b-a12b"
    answer_model: str = "nvidia/nemotron-3-nano-30b-a3b"
    temperature: float = 0.0
    max_tokens: int = 1024

    db_path: Path = Path("data/claims.db")
    semantic_layer: Path = Path("config/semantic_layer.yaml")
    policies: Path = Path("config/policies.yaml")
    examples_path: Path = Path("data/splits/train.jsonl")

    default_role: str = "analyst"
    # API key -> principal mapping, e.g. '{"key-abc": {"role": "provider_portal", "context": {"provider_id": 7}}}'.
    # When empty (local dev), callers may pick a role with the X-Quill-Role header.
    api_principals: dict[str, dict] = {}
    max_repairs: int = 2
    few_shot_k: int = 3
    query_timeout_s: float = 5.0

    @property
    def offline(self) -> bool:
        return self.provider == "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
