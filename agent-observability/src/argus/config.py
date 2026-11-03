from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARGUS_", env_file=".env", extra="ignore")

    db_path: Path = Path(".argus/argus.db")
    policies: Path = Path("config/policies.yaml")
    pricing: Path = Path("config/pricing.yaml")
    admin_token: SecretStr | None = None  # required for /admin endpoints

    provider: Literal["nvidia", "openai_compatible", "offline"] = "offline"
    nvidia_api_key: SecretStr | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    base_url: str = "https://integrate.api.nvidia.com/v1"
    investigator_model: str = "nvidia/nemotron-3-super-120b-a12b"
    detect_interval_s: float = 0  # >0 runs detection in the background on this interval

    @property
    def offline(self) -> bool:
        return self.provider == "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()
