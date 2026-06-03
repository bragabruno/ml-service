from __future__ import annotations

import os

from .base import LLMClient, LLMMessage, LLMResponse
from .mock_client import MockLLM


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()

    if provider == "mock":
        return MockLLM()

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or api_key == "your-anthropic-api-key-here":
            raise ValueError("ANTHROPIC_API_KEY must be set when LLM_PROVIDER=anthropic")
        from .anthropic_client import AnthropicClient
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        return AnthropicClient(api_key=api_key, model=model)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'mock' or 'anthropic'.")


__all__ = ["LLMClient", "LLMMessage", "LLMResponse", "MockLLM", "get_llm_client"]
