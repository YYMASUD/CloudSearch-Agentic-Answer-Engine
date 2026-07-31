"""
Model Router — selects the best available LLM provider based on policy.

Routing policy (LLM_ROUTING_POLICY env var):
    "cost"    — prefer cheapest: Ollama → Groq → Mistral → OpenAI
    "latency" — prefer fastest:  Groq → Ollama → Mistral → OpenAI
    "privacy" — prefer local:    Ollama → (cloud blocked)
    "quality" — prefer best:     OpenAI → Mistral → Groq → Ollama

The router probes each provider's health_check() in priority order and
returns the first healthy one. Results are cached for 60s to avoid
re-probing on every request.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator

from .providers.base_provider import LLMConfig, LLMMessage, LLMProvider, ProviderName
from .providers.ollama_provider import OllamaProvider
from .providers.openai_provider import OpenAIProvider
from .providers.groq_provider import GroqProvider
from .providers.mistral_provider import MistralProvider

logger = logging.getLogger(__name__)

# Provider priority chains by policy
_POLICY_CHAINS: dict[str, list[ProviderName]] = {
    "cost":    [ProviderName.OLLAMA, ProviderName.GROQ, ProviderName.MISTRAL, ProviderName.OPENAI],
    "latency": [ProviderName.GROQ, ProviderName.OLLAMA, ProviderName.MISTRAL, ProviderName.OPENAI],
    "privacy": [ProviderName.OLLAMA],  # Cloud providers blocked in privacy mode
    "quality": [ProviderName.OPENAI, ProviderName.MISTRAL, ProviderName.GROQ, ProviderName.OLLAMA],
}

_HEALTH_CACHE_TTL = 60.0  # seconds


class StubProvider(LLMProvider):
    """No-op stub for testing and graceful total failure."""

    @property
    def name(self) -> ProviderName:
        return ProviderName.STUB

    @property
    def default_model(self) -> str:
        return "stub"

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        yield "⚠️ No LLM providers available. Please configure Ollama or an API key."


class ModelRouter:
    """
    Selects and falls back between LLM providers.

    Usage:
        router = ModelRouter()
        provider = await router.select_provider()
        async for token in provider.stream_complete(messages):
            ...
    """

    def __init__(self, policy: str | None = None) -> None:
        self._policy = (policy or os.getenv("LLM_ROUTING_POLICY", "cost")).lower()
        self._providers: dict[ProviderName, LLMProvider] = {
            ProviderName.OLLAMA: OllamaProvider(),
            ProviderName.OPENAI: OpenAIProvider(),
            ProviderName.GROQ: GroqProvider(),
            ProviderName.MISTRAL: MistralProvider(),
        }
        # Health cache: ProviderName → (is_healthy, checked_at)
        self._health_cache: dict[ProviderName, tuple[bool, float]] = {}

    async def select_provider(self, force: ProviderName | None = None) -> LLMProvider:
        """
        Return the first healthy provider in the policy chain.

        Args:
            force: Override the policy and use this provider directly.

        Returns:
            A healthy LLMProvider, or StubProvider if all fail.
        """
        if force and force in self._providers:
            return self._providers[force]

        chain = _POLICY_CHAINS.get(self._policy, _POLICY_CHAINS["cost"])

        for provider_name in chain:
            provider = self._providers.get(provider_name)
            if provider is None:
                continue
            if await self._is_healthy(provider):
                logger.debug("ModelRouter selected %r (policy=%s)", provider_name, self._policy)
                return provider

        logger.error("All LLM providers failed health check — using StubProvider.")
        return StubProvider()

    async def _is_healthy(self, provider: LLMProvider) -> bool:
        """Return cached health status or probe the provider."""
        now = time.monotonic()
        cached = self._health_cache.get(provider.name)
        if cached:
            healthy, checked_at = cached
            if now - checked_at < _HEALTH_CACHE_TTL:
                return healthy

        try:
            healthy = await asyncio.wait_for(provider.health_check(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            healthy = False

        self._health_cache[provider.name] = (healthy, now)
        if not healthy:
            logger.debug("Provider %r is unhealthy — skipping.", provider.name)
        return healthy

    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Convenience: select provider + stream in one call.
        Falls back to StubProvider if all providers fail.
        """
        provider = await self.select_provider()
        async for token in provider.stream_complete(messages, config):
            yield token

    def invalidate_cache(self) -> None:
        """Force re-probe of all providers on next request."""
        self._health_cache.clear()
