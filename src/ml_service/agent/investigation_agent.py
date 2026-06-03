from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ml_service.agent.llm.base import LLMClient, LLMMessage
from ml_service.agent.tools import TOOL_REGISTRY, ToolResult
from ml_service.schemas.investigation import (
    Decision,
    Evidence,
    InvestigateRequest,
    InvestigationReport,
)


@dataclass
class ToolCall:
    tool: str
    result: ToolResult
    latency_ms: float


@dataclass
class AgentTrace:
    transaction_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    model: str = ""


def _parse_json_response(content: str) -> dict[str, Any]:
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        brace_start = content.find("{")
        brace_end = content.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            return json.loads(content[brace_start:brace_end])
        raise


def investigate(
    request: InvestigateRequest,
    llm: LLMClient,
    *,
    max_steps: int = 6,
) -> tuple[InvestigationReport, AgentTrace]:
    t_start = time.monotonic()
    trace = AgentTrace(transaction_id=request.transaction_id)

    tools_to_call = [
        "get_transaction_features",
        "get_velocity",
        "get_rule_hits",
        "get_customer_history",
        "get_shap_explanation",
    ]

    tool_results: list[ToolResult] = []
    for tool_name in tools_to_call[:max_steps]:
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            continue
        t_tool = time.monotonic()
        result = fn(request.transaction_id)
        elapsed = (time.monotonic() - t_tool) * 1000
        trace.tool_calls.append(ToolCall(tool=tool_name, result=result, latency_ms=elapsed))
        tool_results.append(result)

    from prompts.registry import load_prompt

    rendered = load_prompt(
        "investigation",
        "v1",
        transaction_id=request.transaction_id,
        amount="unknown",
        currency="USD",
        country="unknown",
        device_id="unknown",
        new_device="unknown",
        user_id="unknown",
        merchant_id="unknown",
        timestamp="unknown",
    )

    evidence_text = "\n".join(f"- **{r.tool}**: {r.finding}" for r in tool_results)
    user_content = f"{rendered.user}\n\n## Tool Results\n\n{evidence_text}"

    messages = [LLMMessage(role="user", content=user_content)]

    t_llm = time.monotonic()
    response = llm.complete(
        messages,
        system=rendered.system,
        temperature=rendered.metadata.temperature,
    )

    trace.model = response.model
    trace.total_tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
    trace.total_latency_ms = (time.monotonic() - t_start) * 1000

    try:
        parsed = _parse_json_response(response.content)
    except (json.JSONDecodeError, ValueError):
        parsed = {
            "disposition": "REVIEW",
            "confidence": 0.5,
            "summary": "Agent could not parse LLM response. Manual review required.",
            "evidence": [],
            "reason_codes": ["PARSE_ERROR"],
        }

    evidence_list = [
        Evidence(tool=e.get("tool", "unknown"), finding=e.get("finding", ""))
        for e in parsed.get("evidence", [])
    ]

    disposition_str = parsed.get("disposition", "REVIEW")
    try:
        disposition = Decision(disposition_str)
    except ValueError:
        disposition = Decision.REVIEW

    report = InvestigationReport(
        transaction_id=request.transaction_id,
        disposition=disposition,
        confidence=float(parsed.get("confidence", 0.5)),
        summary=parsed.get("summary", ""),
        evidence=evidence_list,
        reason_codes=parsed.get("reason_codes", []),
        trace_id=parsed.get("trace_id"),
        model=response.model,
        tokens_used=trace.total_tokens,
        latency_ms=trace.total_latency_ms,
    )

    return report, trace
