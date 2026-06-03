from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .runner import EvalRunResult


def generate_report(result: EvalRunResult) -> str:
    lines: list[str] = []
    lines.append(f"# Evaluation Report: {result.run_id}")
    lines.append("")
    lines.append(f"- **Model:** {result.model}")
    lines.append(f"- **Prompt Version:** {result.prompt_version}")
    lines.append(f"- **Timestamp:** {result.timestamp}")
    lines.append(f"- **Total Cases:** {result.total_cases}")
    lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for metric, value in sorted(result.aggregate_metrics.items()):
        lines.append(f"| {metric} | {value} |")
    lines.append("")

    lines.append("## Per-Case Results")
    lines.append("")
    lines.append("| Transaction | Expected | Actual | Correct | Difficulty | Latency (ms) |")
    lines.append("|-------------|----------|--------|---------|------------|--------------|")
    for cr in result.case_results:
        correct = "✅" if cr.deterministic_metrics.get("disposition_correct", 0) == 1.0 else "❌"
        lines.append(
            f"| {cr.transaction_id} | {cr.expected_disposition} | "
            f"{cr.report.disposition.value} | {correct} | {cr.difficulty} | "
            f"{cr.latency_ms:.0f} |"
        )
    lines.append("")

    judge_verdicts = [v for cr in result.case_results for v in cr.judge_verdicts]
    if judge_verdicts:
        lines.append("## Judge Verdicts")
        lines.append("")
        for metric_name in ("groundedness", "hallucination", "disposition_accuracy"):
            metric_verdicts = [v for v in judge_verdicts if v.metric == metric_name]
            if metric_verdicts:
                avg = sum(v.score for v in metric_verdicts) / len(metric_verdicts)
                pass_rate = sum(1 for v in metric_verdicts if v.passed) / len(metric_verdicts) * 100
                lines.append(f"### {metric_name.replace('_', ' ').title()}")
                lines.append(f"- Average score: {avg:.1f}/5")
                lines.append(f"- Pass rate: {pass_rate:.0f}%")
                lines.append("")

    lines.append("---")
    lines.append(f"*Generated at {datetime.now(timezone.utc).isoformat()}*")
    return "\n".join(lines)


def save_report(result: EvalRunResult, output_dir: Path | None = None) -> Path:
    output = output_dir or Path("eval/reports")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{result.run_id}.md"
    report_path.write_text(generate_report(result))
    return report_path
