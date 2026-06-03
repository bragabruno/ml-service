from __future__ import annotations

from typing import Any

from ml_service.agent.llm.base import LLMClient, LLMMessage


def narrate_explanation(
    llm: LLMClient,
    *,
    model_version: str,
    fraud_probability: float,
    risk_level: str,
    features: list[dict[str, Any]],
    rule_hits: list[dict[str, str]] | None = None,
) -> str:
    """Turn SHAP/importance contributions + rule hits into an analyst-readable narrative.

    ``features`` items should expose ``name``/``value``/``contribution`` (e.g. the dicts
    produced from ``FeatureContribution``). Uses the versioned ``explanation`` prompt so the
    wording is reviewable and pinned to eval results (Prompt Engineering, FRAUD-084/156).
    """
    from prompts.registry import load_prompt

    rendered = load_prompt(
        "explanation",
        "v1",
        model_version=model_version,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        features=features,
        rule_hits=rule_hits or [],
    )
    response = llm.complete(
        [LLMMessage(role="user", content=rendered.user)],
        system=rendered.system,
        temperature=rendered.metadata.temperature,
    )
    return str(response.content)
