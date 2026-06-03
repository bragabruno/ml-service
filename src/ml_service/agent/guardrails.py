from __future__ import annotations

import re
from dataclasses import dataclass

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD_NUMBER", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]

_TOKEN_PREFIX = "[REDACTED_"


@dataclass
class RedactionResult:
    text: str
    token_map: dict[str, str]
    redaction_count: int


def redact_pii(text: str) -> RedactionResult:
    token_map: dict[str, str] = {}
    redacted = text
    count = 0

    for label, pattern in _PII_PATTERNS:
        matches = pattern.findall(redacted)
        for match in matches:
            token = f"{_TOKEN_PREFIX}{label}_{count}]"
            token_map[token] = match
            redacted = redacted.replace(match, token, 1)
            count += 1

    return RedactionResult(text=redacted, token_map=token_map, redaction_count=count)


def rehydrate(text: str, token_map: dict[str, str]) -> str:
    result = text
    for token, original in token_map.items():
        result = result.replace(token, original)
    return result


_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"disregard\s+(your|all|the)\s+(instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"</?(?:system|user|assistant)>", re.IGNORECASE),
]


@dataclass
class InjectionCheck:
    detected: bool
    patterns_matched: list[str]


def check_prompt_injection(text: str) -> InjectionCheck:
    matched: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern)
    return InjectionCheck(detected=bool(matched), patterns_matched=matched)


def validate_output(content: str, required_keys: list[str]) -> tuple[bool, list[str]]:
    import json

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False, ["Output is not valid JSON"]

    if not isinstance(parsed, dict):
        return False, ["Output is not a JSON object"]

    missing = [k for k in required_keys if k not in parsed]
    if missing:
        return False, [f"Missing required key: {k}" for k in missing]

    return True, []


def should_refuse(text: str) -> tuple[bool, str]:
    injection = check_prompt_injection(text)
    if injection.detected:
        return True, f"Refusing: prompt injection detected ({len(injection.patterns_matched)} patterns matched)"

    redaction = redact_pii(text)
    if redaction.redaction_count > 10:
        return True, f"Refusing: excessive PII detected ({redaction.redaction_count} instances)"

    return False, ""
