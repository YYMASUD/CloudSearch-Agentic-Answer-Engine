"""
LLM Provider ABC — common interface for all language model backends.

Every LLM provider (Ollama, OpenAI, Groq, Mistral) must implement this
interface. The ModelRouter uses it to select and fall back between providers.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class ProviderName(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    MISTRAL = "mistral"
    STUB = "stub"


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    """Per-request LLM configuration overrides."""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    stream: bool = True
    extra: dict = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """
    Abstract LLM backend.

    Providers MUST implement:
        - stream_complete() for token streaming
        - complete() for one-shot completion (can delegate to stream_complete)
        - health_check() for router liveness probing

    Providers MUST handle their own authentication and retry logic.
    They MUST NOT raise for transient errors — instead yield/return empty
    and log the error so the router can fall back.
    """

    @property
    @abc.abstractmethod
    def name(self) -> ProviderName:
        ...

    @property
    @abc.abstractmethod
    def default_model(self) -> str:
        ...

    @abc.abstractmethod
    async def stream_complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion token-by-token.

        Yields:
            String tokens as they are generated.
        """
        ...

    async def complete(
        self,
        messages: list[LLMMessage],
        config: LLMConfig | None = None,
    ) -> str:
        """
        Non-streaming completion. Collects the full response.
        Default implementation consumes stream_complete().
        """
        parts = []
        async for token in self.stream_complete(messages, config):
            parts.append(token)
        return "".join(parts)

    async def health_check(self) -> bool:
        """Return True if the provider endpoint is reachable."""
        return True
