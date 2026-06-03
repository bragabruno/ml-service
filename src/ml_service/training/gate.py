from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateThresholds:
    min_pr_auc: float = 0.60
    min_roc_auc: float = 0.80
    min_recall: float = 0.70
    max_fp_rate: float = 0.15
    max_cost_per_txn: float = 50.0


@dataclass
class GateResult:
    passed: bool
    violations: list[str]
    metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "metrics": self.metrics,
        }


def check_gate(
    pr_auc: float,
    roc_auc: float,
    recall: float,
    fp_rate: float,
    cost: float,
    n_test: int,
    thresholds: GateThresholds | None = None,
) -> GateResult:
    t = thresholds or GateThresholds()
    violations: list[str] = []
    metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall": recall,
        "fp_rate": fp_rate,
        "cost": cost,
        "cost_per_txn": cost / max(n_test, 1),
    }

    if pr_auc < t.min_pr_auc:
        violations.append(f"PR-AUC {pr_auc:.4f} < {t.min_pr_auc}")
    if roc_auc < t.min_roc_auc:
        violations.append(f"ROC-AUC {roc_auc:.4f} < {t.min_roc_auc}")
    if recall < t.min_recall:
        violations.append(f"Recall {recall:.4f} < {t.min_recall}")
    if fp_rate > t.max_fp_rate:
        violations.append(f"FP rate {fp_rate:.4f} > {t.max_fp_rate}")
    if n_test > 0 and cost / n_test > t.max_cost_per_txn:
        violations.append(f"Cost/txn {cost / n_test:.2f} > {t.max_cost_per_txn}")

    return GateResult(passed=len(violations) == 0, violations=violations, metrics=metrics)
