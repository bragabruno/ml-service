from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ml_service.app.config import get_settings
from ml_service.app.observability import get_logger

if TYPE_CHECKING:
    from ml_service.schemas.investigation import InvestigationReport, LabelType

logger = get_logger("case_management")


@runtime_checkable
class CaseManagementClient(Protocol):
    """Sink for agent output flowing into the HITL case-management loop.

    Mirrors the LLM client abstraction (ml_service.agent.llm): a deterministic mock for
    offline/test, a real HTTP client for production — selected by `get_case_client()`.
    """

    def attach_report(self, case_id: str, report: InvestigationReport) -> None:
        """Attach an auto-drafted investigation report to an existing FraudCase."""
        ...

    def record_label(self, case_id: str, transaction_id: str, label: LabelType) -> None:
        """Record an analyst-derived ground-truth label against a case."""
        ...


@dataclass
class _AttachedReport:
    case_id: str
    report: InvestigationReport


@dataclass
class _RecordedLabel:
    case_id: str
    transaction_id: str
    label: LabelType


@dataclass
class MockCaseClient:
    """Deterministic, offline case client — records in memory and logs. Default provider."""

    attached: list[_AttachedReport] = field(default_factory=list)
    labels: list[_RecordedLabel] = field(default_factory=list)

    def attach_report(self, case_id: str, report: InvestigationReport) -> None:
        self.attached.append(_AttachedReport(case_id=case_id, report=report))
        logger.info(
            "case_report_attached",
            case_id=case_id,
            txn_id=report.transaction_id,
            disposition=report.disposition.value,
        )

    def record_label(self, case_id: str, transaction_id: str, label: LabelType) -> None:
        self.labels.append(_RecordedLabel(case_id=case_id, transaction_id=transaction_id, label=label))
        logger.info("case_label_recorded", case_id=case_id, txn_id=transaction_id, label=label.value)


class HttpCaseClient:
    """Production case client — POSTs reports/labels to the case-management service."""

    def __init__(self, base_url: str) -> None:
        if not base_url:
            raise ValueError("case_mgmt_url must be set when CASE_MGMT_PROVIDER=http")
        self._base_url = base_url.rstrip("/")

    def attach_report(self, case_id: str, report: InvestigationReport) -> None:
        import httpx

        url = f"{self._base_url}/cases/{case_id}/reports"
        resp = httpx.post(url, json=report.model_dump(mode="json"), timeout=10.0)
        resp.raise_for_status()
        logger.info("case_report_attached", case_id=case_id, txn_id=report.transaction_id, status=resp.status_code)

    def record_label(self, case_id: str, transaction_id: str, label: LabelType) -> None:
        import httpx

        url = f"{self._base_url}/cases/{case_id}/labels"
        resp = httpx.post(url, json={"transaction_id": transaction_id, "label": label.value}, timeout=10.0)
        resp.raise_for_status()
        logger.info("case_label_recorded", case_id=case_id, txn_id=transaction_id, label=label.value)


_client: CaseManagementClient | None = None


def get_case_client() -> CaseManagementClient:
    """Return the configured case-management client (mock by default).

    Fail-fast: an `http` provider without a URL raises, matching the LLM factory's shape.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    provider = settings.case_mgmt_provider.lower()
    if provider == "mock":
        _client = MockCaseClient()
    elif provider == "http":
        _client = HttpCaseClient(settings.case_mgmt_url)
    else:
        raise ValueError(f"Unknown CASE_MGMT_PROVIDER: {provider!r}. Use 'mock' or 'http'.")
    return _client


__all__ = [
    "CaseManagementClient",
    "HttpCaseClient",
    "MockCaseClient",
    "get_case_client",
]
