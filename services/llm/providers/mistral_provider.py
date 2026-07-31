"""
Mistral LLM provider — inference via Mistral AI API.
Uses OpenAI-compatible streaming endpoint.
Default model: mistral-small-latest.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator

from .base_provider import LLMConfig, LLMMessage, LLMProvider, ProviderName
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    """Mistral AI provider — OpenAI-compatible streaming."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "mistral-small-latest",
    ) -> None:
        api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        self._impl = OpenAIProvider(
            api_key=api_key,
            default_model=default_model,
            base_url="https://api.mistral.ai/v1",
        )

    @property
    def name(self) -> ProviderName:
        return ProviderName.MISTRAL

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
