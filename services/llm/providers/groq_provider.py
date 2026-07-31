"""
Groq LLM provider — ultra-fast inference via Groq Cloud API.
Groq uses the same OpenAI-compatible streaming format.
Default model: llama-3.1-8b-instant (fastest Groq model).
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from .base_provider import LLMConfig, LLMMessage, LLMProvider, ProviderName
from .openai_provider import OpenAIProvider  # Reuse OpenAI-compatible implementation

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq Cloud inference — OpenAI-compatible streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "llama-3.1-8b-instant",
    ) -> None:
        api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._impl = OpenAIProvider(
            api_key=api_key,
            default_model=default_model,
            base_url="https://api.groq.com/openai/v1",
        )

    @property
    def name(self) -> ProviderName:
        return ProviderName.GROQ

    @property
    def default_model(self) -> str:
        return self._impl.default_model

    async def health_check(self) -> bool:
        return await self._impl.health_check()

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._impl.stream_complete(messages, config):
            yield token
