from __future__ import annotations

from typing import Any

import anthropic

from .base import LLMMessage, LLMResponse


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        api_messages: list[anthropic.types.MessageParam] = [
            {"role": m.role, "content": m.content}  # type: ignore[typeddict-item]
            for m in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        for block in response.content:
            if isinstance(block, anthropic.types.TextBlock):
                text_parts.append(block.text)
        content = "\n".join(text_parts)

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            stop_reason=response.stop_reason or "end_turn",
        )
