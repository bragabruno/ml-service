from __future__ import annotations

from eval.safety.red_team import run_safety_suite

from ml_service.agent.llm.mock_client import MockLLM


def test_safety_suite_passes() -> None:
    llm = MockLLM()
    results = run_safety_suite(llm)
    failures = [r for r in results if not r.passed]
    assert not failures, f"Safety tests failed: {[r.test_name for r in failures]}"
