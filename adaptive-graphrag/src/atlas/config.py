"""Runtime configuration, sourced from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    provider: Literal["nvidia", "offline"] = "nvidia"
    nvidia_api_key: SecretStr | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    nvidia_base_url: str | None = None

    generator_model: str = "nvidia/nemotron-3-super-120b-a12b"
    fast_model: str = "nvidia/nemotron-3-nano-30b-a3b"
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    temperature: float = 0.1
    max_tokens: int = 2048

    data_dir: Path = Path(".atlas")
    collection: str = "atlas_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None

    graph_backend: Literal["networkx", "neo4j"] = "networkx"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("atlas-password")

    # Optional shared secret for the HTTP API (sent as `X-API-Key`). Unset = open (local dev).
    api_key: SecretStr | None = None

    tavily_api_key: SecretStr | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    # Ingestion
    chunk_size: int = 1200
    chunk_overlap: int = 150
    extract_graph: bool = True

    # Retrieval / agent behaviour
    dense_k: int = 20
    sparse_k: int = 20
    rerank_top_n: int = 6
    graph_hops: int = 2
    graph_max_facts: int = 25
    community_top_k: int = 4
    min_relevant_docs: int = 2
    max_rewrites: int = 2
    max_generation_retries: int = 1

    @property
    def offline(self) -> bool:
        return self.provider == "offline"

    @property
    def web_search_enabled(self) -> bool:
        return self.tavily_api_key is not None and bool(self.tavily_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
