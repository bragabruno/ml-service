from __future__ import annotations

from ml_service.schemas.investigation import Decision, InvestigationReport


def disposition_accuracy(report: InvestigationReport, expected: str) -> bool:
    try:
        expected_decision = Decision(expected)
    except ValueError:
        return False
    return report.disposition == expected_decision


def disposition_accuracy_batch(
    reports: list[InvestigationReport], expected: list[str]
) -> float:
    if not reports:
        return 0.0
    correct = sum(
        1 for r, e in zip(reports, expected) if disposition_accuracy(r, e)
    )
    return correct / len(reports)


def citation_correctness(report: InvestigationReport) -> float:
    if not report.evidence:
        return 0.0
    valid_tools = {
        "get_transaction_features",
        "get_velocity",
        "get_similar_cases",
        "get_shap_explanation",
        "get_rule_hits",
        "get_customer_history",
    }
    valid = sum(1 for e in report.evidence if e.tool in valid_tools and e.finding)
    return valid / len(report.evidence)


def citation_correctness_batch(reports: list[InvestigationReport]) -> float:
    if not reports:
        return 0.0
    scores = [citation_correctness(r) for r in reports]
    return sum(scores) / len(scores)


def tool_selection_correctness(report: InvestigationReport) -> float:
    expected_tools = {
        "get_transaction_features",
        "get_velocity",
        "get_rule_hits",
    }
    used_tools = {e.tool for e in report.evidence}
    if not expected_tools:
        return 1.0
    overlap = expected_tools & used_tools
    return len(overlap) / len(expected_tools)


def tool_selection_correctness_batch(reports: list[InvestigationReport]) -> float:
    if not reports:
        return 0.0
    scores = [tool_selection_correctness(r) for r in reports]
    return sum(scores) / len(scores)


def confidence_calibration(report: InvestigationReport) -> float:
    if report.disposition == Decision.DECLINE and report.confidence >= 0.7:
        return 1.0
    if report.disposition == Decision.APPROVE and report.confidence >= 0.6:
        return 1.0
    if report.disposition == Decision.REVIEW and 0.3 <= report.confidence <= 0.8:
        return 1.0
    return 0.5


def confidence_calibration_batch(reports: list[InvestigationReport]) -> float:
    if not reports:
        return 0.0
    scores = [confidence_calibration(r) for r in reports]
    return sum(scores) / len(scores)
