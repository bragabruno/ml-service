from __future__ import annotations

from typing import Any

from ml_service.app.observability import get_logger
from ml_service.schemas.retraining import PromotionDecision

logger = get_logger("promotion")


def evaluate_promotion(
    candidate_metrics: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
    *,
    gate_passed: bool,
    min_pr_auc_improvement: float = 0.0,
) -> PromotionDecision:
    """Champion/challenger guardrail: is a candidate eligible to replace the incumbent?

    Eligibility is necessary but NOT sufficient — promotion always requires human approval
    (FRAUD-080); this never deploys. A candidate is eligible only if it clears the absolute
    gate and does not regress against the current production model on PR-AUC or cost.
    """
    candidate_pr_auc = _get_float(candidate_metrics, "pr_auc")
    reasons: list[str] = []

    if not gate_passed:
        reasons.append("candidate failed the absolute quality gate")
        return _decision(False, reasons, candidate_pr_auc, None)

    if incumbent_metrics is None:
        reasons.append("no incumbent model in production (cold start)")
        return _decision(True, reasons, candidate_pr_auc, None)

    incumbent_pr_auc = _get_float(incumbent_metrics, "pr_auc")
    required = (incumbent_pr_auc or 0.0) + min_pr_auc_improvement
    eligible = True

    if candidate_pr_auc is None or candidate_pr_auc < required:
        eligible = False
        reasons.append(f"PR-AUC {candidate_pr_auc} < required {required:.4f} (incumbent {incumbent_pr_auc})")

    cand_cost = _get_float(candidate_metrics, "cost")
    inc_cost = _get_float(incumbent_metrics, "cost")
    if cand_cost is not None and inc_cost is not None and cand_cost > inc_cost:
        eligible = False
        reasons.append(f"cost {cand_cost} worse than incumbent {inc_cost}")

    if eligible:
        reasons.append("candidate meets or beats incumbent on PR-AUC and cost")

    return _decision(eligible, reasons, candidate_pr_auc, incumbent_pr_auc)


def _decision(
    eligible: bool,
    reasons: list[str],
    candidate_pr_auc: float | None,
    incumbent_pr_auc: float | None,
) -> PromotionDecision:
    logger.info("promotion_evaluated", eligible=eligible, reasons=reasons)
    return PromotionDecision(
        eligible=eligible,
        reasons=reasons,
        candidate_pr_auc=candidate_pr_auc,
        incumbent_pr_auc=incumbent_pr_auc,
        requires_human_approval=True,
    )


def _get_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else None
