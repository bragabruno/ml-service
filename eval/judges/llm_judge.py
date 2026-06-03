from __future__ import annotations

import json
from typing import Any

from ml_service.agent.llm.base import LLMClient, LLMMessage
from ml_service.schemas.investigation import InvestigationReport, JudgeVerdict

from prompts.registry import load_prompt


def _parse_judge_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        brace_start = content.find("{")
        brace_end = content.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            return json.loads(content[brace_start:brace_end])
        return {"score": 3, "reasoning": "Could not parse judge response", "pass": True}


def judge_groundedness(
    report: InvestigationReport,
    llm: LLMClient,
) -> JudgeVerdict:
    rendered = load_prompt(
        "judges",
        "groundedness",
        report=report.model_dump(),
        evidence=[e.model_dump() for e in report.evidence],
    )

    messages = [LLMMessage(role="user", content=rendered.user)]
    response = llm.complete(messages, system=rendered.system, temperature=0.0)
    parsed = _parse_judge_response(response.content)

    return JudgeVerdict(
        metric="groundedness",
        score=int(parsed.get("score", 3)),
        reasoning=parsed.get("reasoning", ""),
        passed=bool(parsed.get("pass", True)),
    )


def judge_hallucination(
    report: InvestigationReport,
    llm: LLMClient,
) -> JudgeVerdict:
    rendered = load_prompt(
        "judges",
        "hallucination",
        report=report.model_dump(),
        evidence=[e.model_dump() for e in report.evidence],
    )

    messages = [LLMMessage(role="user", content=rendered.user)]
    response = llm.complete(messages, system=rendered.system, temperature=0.0)
    parsed = _parse_judge_response(response.content)

    return JudgeVerdict(
        metric="hallucination",
        score=int(parsed.get("score", 3)),
        reasoning=parsed.get("reasoning", ""),
        passed=bool(parsed.get("pass", True)),
    )


def judge_disposition(
    report: InvestigationReport,
    llm: LLMClient,
    expected_disposition: str | None = None,
) -> JudgeVerdict:
    rendered = load_prompt(
        "judges",
        "disposition",
        report=report.model_dump(),
        evidence=[e.model_dump() for e in report.evidence],
        expected_disposition=expected_disposition or "Not provided",
    )

    messages = [LLMMessage(role="user", content=rendered.user)]
    response = llm.complete(messages, system=rendered.system, temperature=0.0)
    parsed = _parse_judge_response(response.content)

    return JudgeVerdict(
        metric="disposition_accuracy",
        score=int(parsed.get("score", 3)),
        reasoning=parsed.get("reasoning", ""),
        passed=bool(parsed.get("pass", True)),
    )


def run_all_judges(
    report: InvestigationReport,
    llm: LLMClient,
    expected_disposition: str | None = None,
) -> list[JudgeVerdict]:
    return [
        judge_groundedness(report, llm),
        judge_hallucination(report, llm),
        judge_disposition(report, llm, expected_disposition),
    ]
