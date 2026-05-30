"""PACT-EL falsification gate and rollback logic."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from pydantic import Field

from pact_el.renderer import estimate_token_delta
from pact_el.schemas import (
    AxiomPatch,
    Canary,
    CanaryKind,
    CanaryResult,
    GateDecision,
    NoPatchDiagnosis,
    Severity,
    StrictModel,
)


class FalsificationGateConfig(StrictModel):
    max_absolute_token_delta: int = 800
    max_relative_token_delta: float = 0.35
    require_fix_canary: bool = True
    require_format_canary: bool = True
    treat_boundary_failure_as_regression: bool = True
    critical_severities: List[Severity] = Field(
        default_factory=lambda: [Severity.CRITICAL]
    )


class GateOutcome(StrictModel):
    accepted: bool
    decision: GateDecision
    reasons: List[str] = Field(default_factory=list)
    failed_canary_ids: List[str] = Field(default_factory=list)
    critical_regression: bool = False
    fix_claim_supported: bool = False
    output_contract_preserved: bool = False
    token_delta_ok: bool = True
    token_delta: int = 0
    no_patch_diagnosis: Optional[NoPatchDiagnosis] = None


def evaluate_falsification_gate(
    original_prompt: str,
    rendered_prompt: str,
    patch: AxiomPatch,
    canaries: Sequence[Canary],
    results: Sequence[CanaryResult],
    no_patch_diagnosis: Optional[NoPatchDiagnosis] = None,
    config: Optional[FalsificationGateConfig] = None,
) -> GateOutcome:
    config = config or FalsificationGateConfig()
    canary_by_id: Dict[str, Canary] = {canary.canary_id: canary for canary in canaries}
    failed = [result for result in results if not result.passed]
    failed_ids = [result.canary_id for result in failed]

    token_delta = patch.token_delta
    if token_delta == 0:
        token_delta = estimate_token_delta(original_prompt, rendered_prompt)
    token_delta_ok = _token_delta_ok(
        original_prompt,
        token_delta,
        config.max_absolute_token_delta,
        config.max_relative_token_delta,
    )

    fix_claim_supported = any(
        result.passed
        and canary_by_id.get(result.canary_id) is not None
        and canary_by_id[result.canary_id].kind == CanaryKind.FIX
        for result in results
    )
    output_contract_preserved = all(
        result.passed
        for result in results
        if canary_by_id.get(result.canary_id) is not None
        and canary_by_id[result.canary_id].kind == CanaryKind.FORMAT_SCHEMA
    )
    has_format = any(canary.kind == CanaryKind.FORMAT_SCHEMA for canary in canaries)
    if config.require_format_canary and not has_format:
        output_contract_preserved = False

    critical_regression = _has_critical_regression(
        failed,
        canary_by_id,
        config,
    )

    reasons: List[str] = []
    if no_patch_diagnosis is not None:
        reasons.append(no_patch_diagnosis.reason)
        return GateOutcome(
            accepted=False,
            decision=GateDecision.REJECTED_NO_PATCH,
            reasons=reasons,
            failed_canary_ids=failed_ids,
            critical_regression=critical_regression,
            fix_claim_supported=fix_claim_supported,
            output_contract_preserved=output_contract_preserved,
            token_delta_ok=token_delta_ok,
            token_delta=token_delta,
            no_patch_diagnosis=no_patch_diagnosis,
        )

    if not token_delta_ok:
        reasons.append(
            f"token delta {token_delta} exceeds configured absolute/relative bounds"
        )
    if config.require_fix_canary and not fix_claim_supported:
        reasons.append("fix canary did not support the patch claim")
    if not output_contract_preserved:
        reasons.append("format/schema canary did not preserve the output contract")
    if critical_regression:
        reasons.append("critical regression detected")
    if failed:
        reasons.append("one or more canaries failed: " + ", ".join(failed_ids))

    accepted = (
        token_delta_ok
        and (fix_claim_supported or not config.require_fix_canary)
        and (output_contract_preserved or not config.require_format_canary)
        and not critical_regression
        and not failed
    )
    if accepted:
        return GateOutcome(
            accepted=True,
            decision=GateDecision.ACCEPTED,
            reasons=["patch passed deterministic falsification gate"],
            failed_canary_ids=[],
            critical_regression=False,
            fix_claim_supported=fix_claim_supported,
            output_contract_preserved=output_contract_preserved,
            token_delta_ok=True,
            token_delta=token_delta,
        )

    return GateOutcome(
        accepted=False,
        decision=(
            GateDecision.REJECTED_REGRESSION
            if critical_regression
            else GateDecision.REJECTED_VALIDATION
        ),
        reasons=reasons,
        failed_canary_ids=failed_ids,
        critical_regression=critical_regression,
        fix_claim_supported=fix_claim_supported,
        output_contract_preserved=output_contract_preserved,
        token_delta_ok=token_delta_ok,
        token_delta=token_delta,
    )


def should_attempt_repair(outcome: GateOutcome, already_repaired: bool = False) -> bool:
    return (
        not already_repaired
        and not outcome.accepted
        and outcome.decision != GateDecision.REJECTED_NO_PATCH
        and bool(outcome.failed_canary_ids)
    )


def make_no_patch_diagnosis(
    outcome: GateOutcome,
    reason: Optional[str] = None,
) -> NoPatchDiagnosis:
    return NoPatchDiagnosis(
        reason=reason or "; ".join(outcome.reasons) or "patch failed falsification",
        evidence_ids=outcome.failed_canary_ids,
        recommended_owner="prompt" if not outcome.critical_regression else "human_review",
        confidence=0.7 if outcome.critical_regression else 0.55,
        metadata={"gate_outcome": outcome.model_dump(mode="json")},
    )


def _token_delta_ok(
    original_prompt: str,
    token_delta: int,
    max_absolute: int,
    max_relative: float,
) -> bool:
    original_tokens = max(1, round(len(original_prompt) / 4))
    absolute_ok = abs(token_delta) <= max_absolute
    relative_ok = abs(token_delta) / original_tokens <= max_relative
    return absolute_ok or relative_ok


def _has_critical_regression(
    failed_results: Sequence[CanaryResult],
    canary_by_id: Dict[str, Canary],
    config: FalsificationGateConfig,
) -> bool:
    critical = set(config.critical_severities)
    for result in failed_results:
        canary = canary_by_id.get(result.canary_id)
        if canary is None:
            continue
        if canary.kind == CanaryKind.REGRESSION:
            return True
        if (
            config.treat_boundary_failure_as_regression
            and canary.kind == CanaryKind.BOUNDARY
        ):
            return True
        failing_validator_ids = {
            validator.validator_id
            for validator in result.validator_outputs
            if not validator.passed
        }
        for spec in canary.validator_specs:
            if spec.validator_id in failing_validator_ids and spec.severity in critical:
                return True
    return False
