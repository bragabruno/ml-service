from __future__ import annotations

import json
import sys
from pathlib import Path

from .runner import EvalRunResult

BASELINES_DIR = Path(__file__).parent / "baselines"
BASELINE_FILE = BASELINES_DIR / "current.json"


def load_baselines(path: Path | None = None) -> dict[str, float]:
    baseline_path = path or BASELINE_FILE
    if not baseline_path.exists():
        return {}
    return json.loads(baseline_path.read_text())


def save_baselines(metrics: dict[str, float], path: Path | None = None) -> None:
    baseline_path = path or BASELINE_FILE
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(metrics, indent=2))


def check_gate(
    result: EvalRunResult,
    *,
    min_disposition_accuracy: float = 0.5,
    min_groundedness: float = 3.0,
    max_hallucination_rate: float = 0.5,
    baselines: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    agg = result.aggregate_metrics

    disp_acc = agg.get("disposition_accuracy", 0.0)
    if disp_acc < min_disposition_accuracy:
        failures.append(
            f"Disposition accuracy {disp_acc:.4f} < threshold {min_disposition_accuracy}"
        )

    groundedness = agg.get("judge_groundedness_avg", 5.0)
    if groundedness < min_groundedness:
        failures.append(
            f"Groundedness avg {groundedness:.2f} < threshold {min_groundedness}"
        )

    hallucination_pass_rate = agg.get("judge_hallucination_pass_rate", 1.0)
    hallucination_fail_rate = 1.0 - hallucination_pass_rate
    if hallucination_fail_rate > max_hallucination_rate:
        failures.append(
            f"Hallucination fail rate {hallucination_fail_rate:.2%} > threshold {max_hallucination_rate:.2%}"
        )

    if baselines:
        # Only quality metrics are regression-checked. Latency is "lower is better",
        # so a decrease is an improvement, not a regression.
        for metric, baseline_value in baselines.items():
            if metric.endswith("_ms"):
                continue
            current_value = agg.get(metric)
            if current_value is not None and current_value < baseline_value * 0.9:
                failures.append(
                    f"Regression: {metric} dropped from baseline {baseline_value:.4f} "
                    f"to {current_value:.4f} (>10% regression)"
                )

    passed = len(failures) == 0
    return passed, failures


def main() -> int:
    from .runner import load_golden_dataset, run_eval
    from .report import generate_report, save_report
    from ml_service.agent.llm.factory import get_llm_client

    llm = get_llm_client()
    result = run_eval(llm, run_judges=True)

    report_text = generate_report(result)
    print(report_text)

    report_path = save_report(result)
    print(f"\nReport saved to: {report_path}")

    baselines = load_baselines()
    passed, failures = check_gate(result, baselines=baselines)

    if not passed:
        print("\n❌ EVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n✅ EVAL GATE PASSED")

    if not baselines:
        save_baselines(result.aggregate_metrics)
        print("  (saved current metrics as new baselines)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
