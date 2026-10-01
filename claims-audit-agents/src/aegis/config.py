from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    provider: Literal["nvidia", "openai_compatible", "offline"] = "nvidia"
    nvidia_api_key: SecretStr | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    api_key: SecretStr | None = None
    base_url: str = "https://integrate.api.nvidia.com/v1"

    clinical_model: str = "nvidia/nemotron-3-super-120b-a12b"
    coding_model: str = "nvidia/nemotron-3-super-120b-a12b"
    vision_model: str = "nvidia/nemotron-nano-12b-v2-vl"

    reference: Path = Path("config/reference.yaml")
    policy_manual: Path = Path("config/policy_manual.md")
    registry: Path = Path("data/registry.json")
    state_dir: Path = Path(".aegis")

    override_threshold: float = 0.80
    uphold_threshold: float = 0.70

    @property
    def offline(self) -> bool:
        return self.provider == "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
