from __future__ import annotations

from dataclasses import dataclass

from atlas.config import Settings
from atlas.providers.base import LLM, Embedder, Reranker


@dataclass(frozen=True)
class Providers:
    generator: LLM  # high-quality model: answers, community reports, graph extraction
    fast: LLM  # low-latency model: routing, grading, rewriting, reflection
    embedder: Embedder
    reranker: Reranker


def build_providers(settings: Settings) -> Providers:
    if settings.offline:
        from atlas.providers.offline import HashEmbedder, OfflineLLM, OverlapReranker

        llm = OfflineLLM()
        return Providers(generator=llm, fast=llm, embedder=HashEmbedder(), reranker=OverlapReranker())

    from atlas.providers.nvidia import NvidiaEmbedder, NvidiaLLM, NvidiaReranker

    key = settings.nvidia_api_key.get_secret_value() if settings.nvidia_api_key else None
    if not key and not settings.nvidia_base_url:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Get one at https://build.nvidia.com, set ATLAS_NVIDIA_BASE_URL for a "
            "self-hosted NIM, or run with ATLAS_PROVIDER=offline."
        )
    common = {"api_key": key, "base_url": settings.nvidia_base_url}
    return Providers(
        generator=NvidiaLLM(
            settings.generator_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            **common,
        ),
        fast=NvidiaLLM(settings.fast_model, temperature=0.0, max_tokens=1024, **common),
        embedder=NvidiaEmbedder(settings.embedding_model, **common),
        reranker=NvidiaReranker(settings.rerank_model, **common),
    )
