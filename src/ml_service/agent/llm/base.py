from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "end_turn"


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...
