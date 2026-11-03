"""Instrument a small tool-using agent on NVIDIA NIM and send its traces to Argus.

argus create-key support-bot          # prints an API key
argus serve &                         # collector on :8003
ARGUS_API_KEY=... NVIDIA_API_KEY=... python examples/nim_agent.py "How do I reset my password?"
"""

from __future__ import annotations

import asyncio
import os
import sys

from openai import AsyncOpenAI

from argus.sdk import get_tracer, init, trace_agent, trace_tool
from argus.sdk.instrument.openai import instrument_openai

init(
    endpoint=os.getenv("ARGUS_ENDPOINT", "http://localhost:8003"),
    api_key=os.environ["ARGUS_API_KEY"],
    service="support-bot",
)
client = instrument_openai(
    AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
)
MODEL = "nvidia/nemotron-3-nano-30b-a3b"


@trace_tool("search_kb")
def search_kb(query: str) -> str:
    return "To reset your password, open Settings > Security > Reset password and follow the email link."


@trace_agent("support-bot", version="v1.0")
async def answer(question: str) -> str:
    context = search_kb(question)
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"Answer using this context:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(asyncio.run(answer(" ".join(sys.argv[1:]) or "How do I reset my password?")))
    get_tracer().exporter.flush()  # type: ignore[union-attr]
