"""
Ollama LLM provider — local inference via Ollama REST API.

Ollama must be running locally (or at OLLAMA_BASE_URL).
Default model: llama3.2:3b (configurable via OLLAMA_DEFAULT_MODEL).

Streams token-by-token using Ollama's /api/chat endpoint.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import httpx

from .base_provider import LLMConfig, LLMMessage, LLMProvider, ProviderName

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Local Ollama inference provider."""

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._default_model = default_model or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
        self._timeout = timeout_s

    @property
    def name(self) -> ProviderName:
        return ProviderName.OLLAMA

    @property
    def default_model(self) -> str:
        return self._default_model

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or LLMConfig()
        model = cfg.model or self._default_model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as exc:
            logger.exception("OllamaProvider stream_complete failed: %s", exc)
            return
