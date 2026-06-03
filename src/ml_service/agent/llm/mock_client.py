from __future__ import annotations

import hashlib
import json

from .base import LLMMessage, LLMResponse

_MOCK_INVESTIGATION = {
    "disposition": "REVIEW",
    "confidence": 0.72,
    "summary": "Transaction exhibits several fraud indicators: new device combined with foreign country origin and elevated velocity. However, account history shows prior legitimate activity from the same merchant category. Recommend manual review before final disposition.",
    "evidence": [
        {"tool": "get_transaction_features", "finding": "new_device=true, country=NG, amount=4850.00"},
        {"tool": "get_velocity", "finding": "5 transactions in last 24h (threshold: 3)"},
        {"tool": "get_rule_hits", "finding": "rules triggered: new_device_foreign_country, high_velocity_24h"},
        {"tool": "get_customer_history", "finding": "account_age=342 days, 2 prior chargebacks, 47 legitimate transactions"},
    ],
    "reason_codes": ["NEW_DEVICE_FOREIGN_COUNTRY", "HIGH_VELOCITY_24H", "ELEVATED_AMOUNT"],
}

_MOCK_EXPLANATION = (
    "This transaction scored as high-risk primarily due to three factors: "
    "(1) it originated from a **new device** not previously associated with the account, "
    "(2) the transaction country (**Nigeria**) differs from the account's home country, and "
    "(3) the velocity of transactions in the past 24 hours (5) exceeds the typical pattern (1-2). "
    "The account does have a 342-day history with 47 legitimate transactions, which moderates the risk somewhat. "
    "The model assigned a fraud probability of 0.78, with the top contributing features being "
    "new_device (gain: 0.31), country_risk (gain: 0.24), and velocity_24h (gain: 0.18)."
)


def _deterministic_response(messages: list[LLMMessage], template_hint: str) -> str:
    combined = "".join(m.content for m in messages)
    seed = hashlib.sha256(combined.encode()).hexdigest()[:8]

    if "investigate" in template_hint or "investigation" in template_hint:
        payload = {**_MOCK_INVESTIGATION, "trace_id": seed}
        return json.dumps(payload)
    if "explain" in template_hint or "explanation" in template_hint:
        return _MOCK_EXPLANATION
    if "judge" in template_hint or "groundedness" in template_hint or "hallucination" in template_hint:
        return json.dumps({"score": 4, "reasoning": f"Mock judge: generally grounded (seed={seed})", "pass": True})

    return json.dumps({"response": f"Mock LLM response (seed={seed})", "content": combined[:200]})


class MockLLM:
    def __init__(self, model_name: str = "mock-llm-v1") -> None:
        self._model = model_name

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        hint = system.lower()
        content = _deterministic_response(messages, hint)
        return LLMResponse(
            content=content,
            model=self._model,
            usage={"input_tokens": sum(len(m.content) for m in messages), "output_tokens": len(content)},
            stop_reason="end_turn",
        )
