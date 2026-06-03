from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .base import LLMMessage, LLMResponse

# Country risk buckets used by the deterministic mock heuristic. These mirror the
# synthetic-data generator and the dbt country_risk seed so the mock's reasoning is
# consistent with the rest of the pipeline.
_HIGH_RISK_COUNTRIES = {"NG", "RU", "CN", "ZA", "PK", "VE", "IR"}
_MEDIUM_RISK_COUNTRIES = {"BR", "MX", "IN", "TR", "ID"}


@dataclass(frozen=True)
class _Risk:
    amount: float
    country: str
    new_device: bool
    failed: int
    score: float
    disposition: str
    confidence: float
    reason_codes: list[str]


def _risk_from_signals(text: str) -> _Risk:
    """Derive a disposition from the ``CASE_SIGNALS`` line in the prompt.

    The mock is a *transparent, deterministic risk heuristic* standing in for a real
    LLM so that evals run offline, free, and reproducibly. Real reasoning happens with
    ``LLM_PROVIDER=anthropic``. The thresholds are tuned to the documented golden-case
    semantics (clean US profile → APPROVE, high-amount foreign + new device + failed
    attempts → DECLINE, mixed signals → REVIEW).
    """
    amount_match = re.search(r"amount=([0-9]+(?:\.[0-9]+)?)", text)
    amount = float(amount_match.group(1)) if amount_match else 0.0
    country_match = re.search(r"country=([A-Za-z]{2})", text)
    country = country_match.group(1).upper() if country_match else "US"
    new_device = bool(re.search(r"new_device=(?:true|1)", text, re.IGNORECASE))
    failed_match = re.search(r"failed_attempts=([0-9]+)", text)
    failed = int(failed_match.group(1)) if failed_match else 0

    score = 0.0
    if amount > 5000:
        score += 0.30
    elif amount > 2000:
        score += 0.15
    elif amount > 1000:
        score += 0.08
    if country in _HIGH_RISK_COUNTRIES:
        score += 0.30
    elif country in _MEDIUM_RISK_COUNTRIES:
        score += 0.12
    if new_device:
        score += 0.15
    if failed >= 5:
        score += 0.25
    elif failed >= 3:
        score += 0.15
    elif failed >= 1:
        score += 0.05

    if score >= 0.70:
        disposition, confidence = "DECLINE", 0.86
    elif score >= 0.20:
        disposition, confidence = "REVIEW", 0.55
    else:
        disposition, confidence = "APPROVE", 0.85

    reason_codes: list[str] = []
    if new_device and country in _HIGH_RISK_COUNTRIES:
        reason_codes.append("NEW_DEVICE_FOREIGN_COUNTRY")
    if failed >= 3:
        reason_codes.append("HIGH_FAILED_ATTEMPTS")
    if amount > 5000:
        reason_codes.append("HIGH_AMOUNT")
    if country in _HIGH_RISK_COUNTRIES or country in _MEDIUM_RISK_COUNTRIES:
        reason_codes.append("HIGH_RISK_GEOGRAPHY")
    if not reason_codes:
        reason_codes = ["NO_SIGNIFICANT_RISK_SIGNALS"]

    return _Risk(
        amount=amount,
        country=country,
        new_device=new_device,
        failed=failed,
        score=round(score, 3),
        disposition=disposition,
        confidence=confidence,
        reason_codes=reason_codes,
    )


def _mock_investigation(text: str, seed: str) -> str:
    r = _risk_from_signals(text)
    summary = (
        f"Risk assessment scored {r.score} from amount={r.amount}, country={r.country}, "
        f"new_device={r.new_device}, failed_attempts={r.failed}. Recommended disposition: {r.disposition}."
    )
    evidence = [
        {
            "tool": "get_transaction_features",
            "finding": (
                f"amount={r.amount}, country={r.country}, new_device={r.new_device}, failed_attempts={r.failed}"
            ),
        },
        {"tool": "get_velocity", "finding": "velocity metrics retrieved for the user's recent activity windows"},
        {"tool": "get_rule_hits", "finding": "rules evaluated: " + ", ".join(r.reason_codes)},
    ]
    payload = {
        "disposition": r.disposition,
        "confidence": r.confidence,
        "summary": summary,
        "evidence": evidence,
        "reason_codes": r.reason_codes,
        "trace_id": seed,
    }
    return json.dumps(payload)


_MOCK_EXPLANATION = (
    "This transaction scored as high-risk primarily due to three factors: "
    "(1) it originated from a **new device** not previously associated with the account, "
    "(2) the transaction country differs from the account's home country, and "
    "(3) the velocity of transactions in the past 24 hours exceeds the typical pattern. "
    "The account history moderates the risk somewhat. The top contributing features were "
    "new_device, country_risk, and velocity_24h."
)


def _deterministic_response(messages: list[LLMMessage], template_hint: str) -> str:
    combined = "".join(m.content for m in messages)
    seed = hashlib.sha256(combined.encode()).hexdigest()[:8]
    hint = template_hint

    # Judge prompts are uniquely marked by "impartial evaluator"; check first because
    # they also contain the word "investigation".
    if "impartial evaluator" in hint:
        return json.dumps(
            {
                "score": 4,
                "reasoning": f"Mock judge: report claims are grounded in cited tool evidence (seed={seed}).",
                "pass": True,
            }
        )
    if "narrator" in hint:
        return _MOCK_EXPLANATION
    if "investigat" in hint:
        return _mock_investigation(combined, seed)

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
