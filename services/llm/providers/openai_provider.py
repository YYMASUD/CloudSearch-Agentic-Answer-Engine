"""
OpenAI LLM provider — GPT-4o / GPT-4o-mini via OpenAI API.
Uses the streaming chat completions endpoint.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

from .base_provider import LLMConfig, LLMMessage, LLMProvider, ProviderName

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT streaming provider."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._default_model = default_model
        self._base_url = base_url

    @property
    def name(self) -> ProviderName:
        return ProviderName.OPENAI

    @property
    def default_model(self) -> str:
        return self._default_model

    async def health_check(self) -> bool:
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        if not self._api_key:
            logger.error("OPENAI_API_KEY not set — skipping OpenAI provider.")
            return

        cfg = config or LLMConfig()
        payload = {
            "model": cfg.model or self._default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                            delta = data["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as exc:
            logger.exception("OpenAIProvider stream_complete failed: %s", exc)
            return
