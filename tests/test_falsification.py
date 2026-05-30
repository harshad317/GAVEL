from __future__ import annotations

from pact_el.falsification import evaluate_falsification_gate
from pact_el.schemas import CanaryResult, CanaryStatus, GateDecision


def test_gate_accepts_when_all_canaries_pass(sample_patch, four_canaries):
    results = [
        CanaryResult(
            canary_id=canary.canary_id,
            kind=canary.kind,
            status=CanaryStatus.PASSED,
            passed=True,
        )
        for canary in four_canaries
    ]

    outcome = evaluate_falsification_gate(
        "original",
        "rendered",
        sample_patch,
        four_canaries,
        results,
    )

    assert outcome.accepted
    assert outcome.decision == GateDecision.ACCEPTED


def test_gate_rejects_regression_failure(sample_patch, four_canaries):
    results = []
    for canary in four_canaries:
        passed = canary.canary_id != "regression"
        results.append(
            CanaryResult(
                canary_id=canary.canary_id,
                kind=canary.kind,
                status=CanaryStatus.PASSED if passed else CanaryStatus.FAILED,
                passed=passed,
                failure_reason="" if passed else "regression broke",
            )
        )

    outcome = evaluate_falsification_gate(
        "original",
        "rendered",
        sample_patch,
        four_canaries,
        results,
    )

    assert not outcome.accepted
    assert outcome.decision == GateDecision.REJECTED_REGRESSION

