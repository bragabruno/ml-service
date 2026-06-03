from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_service.agent.investigation_agent import investigate
from ml_service.agent.llm.base import LLMClient
from ml_service.schemas.investigation import InvestigateRequest, InvestigationReport, JudgeVerdict

from .judges.llm_judge import run_all_judges
from .metrics.deterministic import (
    citation_correctness,
    confidence_calibration,
    disposition_accuracy,
    tool_selection_correctness,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "datasets" / "golden_cases.jsonl"


@dataclass
class CaseResult:
    transaction_id: str
    report: InvestigationReport
    expected_disposition: str
    difficulty: str
    deterministic_metrics: dict[str, float] = field(default_factory=dict)
    judge_verdicts: list[JudgeVerdict] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class EvalRunResult:
    run_id: str
    model: str
    prompt_version: str
    timestamp: str
    total_cases: int
    case_results: list[CaseResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


def load_golden_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    dataset_path = path or GOLDEN_DATASET_PATH
    cases: list[dict[str, Any]] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _compute_aggregates(case_results: list[CaseResult]) -> dict[str, float]:
    if not case_results:
        return {}

    n = len(case_results)

    disp_correct = sum(cr.deterministic_metrics.get("disposition_correct", 0) for cr in case_results) / n
    cit_corr = sum(cr.deterministic_metrics.get("citation_correctness", 0) for cr in case_results) / n
    tool_sel = sum(cr.deterministic_metrics.get("tool_selection", 0) for cr in case_results) / n
    conf_cal = sum(cr.deterministic_metrics.get("confidence_calibration", 0) for cr in case_results) / n
    avg_latency = sum(cr.latency_ms for cr in case_results) / n

    agg: dict[str, float] = {
        "disposition_accuracy": round(disp_correct, 4),
        "citation_correctness": round(cit_corr, 4),
        "tool_selection_correctness": round(tool_sel, 4),
        "confidence_calibration": round(conf_cal, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }

    all_verdicts = [v for cr in case_results for v in cr.judge_verdicts]
    if all_verdicts:
        for metric_name in ("groundedness", "hallucination", "disposition_accuracy"):
            metric_verdicts = [v for v in all_verdicts if v.metric == metric_name]
            if metric_verdicts:
                avg_score = sum(v.score for v in metric_verdicts) / len(metric_verdicts)
                pass_rate = sum(1 for v in metric_verdicts if v.passed) / len(metric_verdicts)
                agg[f"judge_{metric_name}_avg"] = round(avg_score, 2)
                agg[f"judge_{metric_name}_pass_rate"] = round(pass_rate, 4)

    return agg


def run_eval(
    llm: LLMClient,
    *,
    dataset_path: Path | None = None,
    run_judges: bool = True,
    prompt_version: str = "v1",
) -> EvalRunResult:
    import datetime

    run_id = f"eval-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    cases = load_golden_dataset(dataset_path)
    result = EvalRunResult(
        run_id=run_id,
        model=getattr(llm, "_model", "unknown"),
        prompt_version=prompt_version,
        timestamp=timestamp,
        total_cases=len(cases),
    )

    for case in cases:
        t_start = time.monotonic()
        request = InvestigateRequest(
            transaction_id=case["transaction_id"],
            amount=float(case.get("amount", 0.0)),
            currency=case.get("currency", "USD"),
            country=case.get("country", "US"),
            new_device=bool(case.get("new_device", False)),
            failed_attempts=int(case.get("failed_attempts", 0)),
            user_id=case.get("user_id"),
            merchant_id=case.get("merchant_id"),
            device_id=case.get("device_id"),
            untrusted_notes=case.get("injection_payload"),
        )
        report, trace = investigate(request, llm)
        latency = (time.monotonic() - t_start) * 1000

        expected = case.get("expected_disposition", "REVIEW")
        difficulty = case.get("difficulty", "unknown")

        det_metrics = {
            "disposition_correct": float(disposition_accuracy(report, expected)),
            "citation_correctness": citation_correctness(report),
            "tool_selection": tool_selection_correctness(report),
            "confidence_calibration": confidence_calibration(report),
        }

        judge_verdicts: list[JudgeVerdict] = []
        if run_judges:
            judge_verdicts = run_all_judges(report, llm, expected)

        case_result = CaseResult(
            transaction_id=case["transaction_id"],
            report=report,
            expected_disposition=expected,
            difficulty=difficulty,
            deterministic_metrics=det_metrics,
            judge_verdicts=judge_verdicts,
            latency_ms=latency,
        )
        result.case_results.append(case_result)

    result.aggregate_metrics = _compute_aggregates(result.case_results)

    inj_cases = [c for c in cases if c.get("injection_payload")]
    if inj_cases:
        blocked = sum(
            1 for cr in result.case_results if "PROMPT_INJECTION_BLOCKED" in cr.report.safety_flags
        )
        result.aggregate_metrics["injection_resistance"] = round(blocked / len(inj_cases), 4)

    return result


def main() -> int:
    from ml_service.agent.llm.factory import get_llm_client

    from .report import generate_report

    llm = get_llm_client()
    result = run_eval(llm, run_judges=True)
    print(generate_report(result))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
