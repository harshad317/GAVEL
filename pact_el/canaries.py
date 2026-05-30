"""Canary construction, validation, and execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pact_el.clients import TargetClient
from pact_el.schemas import (
    CallLedger,
    Canary,
    CanaryKind,
    CanaryResult,
    CanaryStatus,
    ContractArea,
    PromptAxiomGraph,
    ValidatorKind,
    ValidatorSpec,
)
from pact_el.validators import ValidatorRegistry, validate_canary_output


REQUIRED_CANARY_KINDS = {
    CanaryKind.FIX,
    CanaryKind.BOUNDARY,
    CanaryKind.REGRESSION,
    CanaryKind.FORMAT_SCHEMA,
}


def missing_required_canary_kinds(canaries: Sequence[Canary]) -> List[CanaryKind]:
    present = {canary.kind for canary in canaries}
    return sorted(REQUIRED_CANARY_KINDS - present, key=lambda kind: kind.value)


def require_four_canary_families(canaries: Sequence[Canary]) -> None:
    missing = missing_required_canary_kinds(canaries)
    if missing:
        labels = ", ".join(kind.value for kind in missing)
        raise ValueError(f"compiler output is missing required canary kinds: {labels}")


def guarantee_validators_from_graph(graph: PromptAxiomGraph) -> List[ValidatorSpec]:
    validators: List[ValidatorSpec] = []
    for clause in graph.guarantee_script.clauses:
        validators.append(
            ValidatorSpec(
                validator_id=clause.guarantee_id,
                kind=ValidatorKind.JSON_LOGIC,
                config={"rule": clause.predicate},
                severity=clause.severity,
                description=clause.description,
            )
        )
    return validators


def construct_default_canaries(
    graph: PromptAxiomGraph,
    task_spec: str,
    rubric: str,
) -> List[Canary]:
    """Construct canary shells when running deterministic offline experiments.

    The compiler path should normally provide concrete canaries. This helper is
    for reproducible ablations and cached tests where a minimal deterministic
    canary suite is needed without another optimizer call.
    """

    validators = guarantee_validators_from_graph(graph)
    return [
        Canary(
            canary_id="default_fix_canary",
            kind=CanaryKind.FIX,
            input={
                "case": "fix",
                "instruction": task_spec,
                "stress": "Exercise the repaired behavior directly.",
            },
            expected_behavior="The patched behavior succeeds.",
            target_contract_area=ContractArea.TASK_INTENT,
            validator_specs=validators,
        ),
        Canary(
            canary_id="default_boundary_canary",
            kind=CanaryKind.BOUNDARY,
            input={
                "case": "boundary",
                "instruction": task_spec,
                "rubric": rubric,
                "stress": "Ambiguous, incomplete, or unsafe boundary.",
            },
            expected_behavior="The prompt boundary is respected without invention.",
            target_contract_area=ContractArea.SAFETY_BOUNDARY,
            validator_specs=validators,
        ),
        Canary(
            canary_id="default_regression_canary",
            kind=CanaryKind.REGRESSION,
            input={
                "case": "regression",
                "instruction": task_spec,
                "stress": "Simple representative case that should not regress.",
            },
            expected_behavior="Previously easy behavior remains correct.",
            target_contract_area=ContractArea.TASK_INTENT,
            validator_specs=validators,
        ),
        Canary(
            canary_id="default_format_canary",
            kind=CanaryKind.FORMAT_SCHEMA,
            input={
                "case": "format_schema",
                "instruction": task_spec,
                "stress": "Exact output contract and formatting.",
            },
            expected_behavior="The output contract is preserved exactly.",
            target_contract_area=ContractArea.OUTPUT_SCHEMA,
            validator_specs=validators,
        ),
    ]


async def run_canary(
    rendered_prompt: str,
    canary: Canary,
    target_client: TargetClient,
    call_ledger: CallLedger,
    registry: Optional[ValidatorRegistry] = None,
) -> CanaryResult:
    started = time.perf_counter()
    try:
        response = await target_client.complete(
            rendered_prompt,
            canary.input,
            metadata={"canary_id": canary.canary_id, "canary_kind": canary.kind.value},
        )
        call_ledger.add(response.call_record)
        result = validate_canary_output(
            canary,
            response.output,
            metadata={"target_call_id": response.call_record.call_id},
            registry=registry,
        )
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.target_call_id = response.call_record.call_id
        return result
    except Exception as exc:
        return CanaryResult(
            canary_id=canary.canary_id,
            kind=canary.kind,
            status=CanaryStatus.ERROR,
            passed=False,
            input=canary.input,
            output=None,
            failure_reason=str(exc),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


async def run_canaries(
    rendered_prompt: str,
    canaries: Sequence[Canary],
    target_client: TargetClient,
    call_ledger: CallLedger,
    concurrency: int = 4,
    registry: Optional[ValidatorRegistry] = None,
) -> List[CanaryResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(canary: Canary) -> CanaryResult:
        async with semaphore:
            return await run_canary(
                rendered_prompt,
                canary,
                target_client,
                call_ledger,
                registry=registry,
            )

    return list(await asyncio.gather(*(guarded(canary) for canary in canaries)))


def select_retest_canaries(
    canaries: Sequence[Canary],
    results: Sequence[CanaryResult],
) -> List[Canary]:
    """Return failed canaries plus one regression canary for targeted repair."""

    failed_ids = {result.canary_id for result in results if not result.passed}
    failed = [canary for canary in canaries if canary.canary_id in failed_ids]
    regression = next(
        (canary for canary in canaries if canary.kind == CanaryKind.REGRESSION),
        None,
    )
    if regression is not None and regression.canary_id not in failed_ids:
        failed.append(regression)
    return failed[:2] if len(failed) > 2 else failed


def canary_results_by_id(results: Sequence[CanaryResult]) -> Dict[str, CanaryResult]:
    return {result.canary_id: result for result in results}

