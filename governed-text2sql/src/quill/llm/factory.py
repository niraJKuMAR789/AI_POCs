from __future__ import annotations

from dataclasses import dataclass

from quill.config import Settings
from quill.llm.base import LLM


@dataclass(frozen=True)
class Models:
    sql: LLM
    answer: LLM


def build_models(settings: Settings) -> Models:
    if settings.offline:
        from quill.llm.offline import OfflineLLM

        return Models(sql=OfflineLLM(), answer=OfflineLLM())
    from quill.llm.openai_compat import OpenAICompatibleLLM

    secret = settings.nvidia_api_key if settings.provider == "nvidia" else settings.api_key
    key = secret.get_secret_value() if secret else None
    if settings.provider == "nvidia" and not key:
        raise RuntimeError("NVIDIA_API_KEY is not set (get one at build.nvidia.com) or use QUILL_PROVIDER=offline.")
    return Models(
        sql=OpenAICompatibleLLM(
            settings.sql_model,
            base_url=settings.base_url,
            api_key=key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        ),
        answer=OpenAICompatibleLLM(
            settings.answer_model, base_url=settings.base_url, api_key=key, temperature=0.2, max_tokens=512
        ),
    )
