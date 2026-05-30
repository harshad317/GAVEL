"""End-to-end PACT-EL orchestration."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Optional, Sequence

from pact_el.canaries import run_canaries, select_retest_canaries
from pact_el.clients import OptimizerClient, TargetClient
from pact_el.compiler import CompilerOutput, ContractCompiler
from pact_el.falsification import (
    FalsificationGateConfig,
    evaluate_falsification_gate,
    make_no_patch_diagnosis,
    should_attempt_repair,
)
from pact_el.ledger import (
    build_evidence_ledger,
    choose_four_probe_inputs,
    evidence_from_probe_outputs,
)
from pact_el.renderer import estimate_token_delta, render_prompt
from pact_el.schemas import (
    CallLedger,
    CanaryResult,
    EvidenceRow,
    GateDecision,
    OptimizationReport,
)
from pact_el.validators import ValidatorRegistry


async def pact_optimize(
    prompt: str,
    task_spec: str,
    rubric: str,
    optimizer_client: OptimizerClient,
    target_client: TargetClient,
    logs: Optional[Sequence[Mapping[str, Any]]] = None,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
    evidence_rows: Optional[Sequence[EvidenceRow]] = None,
    budget: int = 9,
    allow_one_repair: bool = True,
    canary_concurrency: int = 4,
    gate_config: Optional[FalsificationGateConfig] = None,
    validator_registry: Optional[ValidatorRegistry] = None,
) -> OptimizationReport:
    """Run the PACT-EL compile, canary, and optional targeted repair flow."""

    call_ledger = CallLedger()
    ledger = build_evidence_ledger(logs=logs, examples=examples, rows=evidence_rows)

    if not ledger.has_behavior_evidence():
        probes = choose_four_probe_inputs(task_spec, rubric)
        outputs = await _run_probe_calls(prompt, probes, target_client, call_ledger)
        ledger.extend(
            evidence_from_probe_outputs(
                probes,
                outputs,
                expected_behavior="Follow the task specification and rubric.",
            )
        )

    compiler = ContractCompiler(optimizer_client)
    compiled = await compiler.compile(
        prompt=prompt,
        task_spec=task_spec,
        rubric=rubric,
        ledger=ledger,
        call_ledger=call_ledger,
        budget=budget,
    )

    rendered_prompt = render_prompt(
        compiled.graph,
        source_prompt=prompt,
        patch=compiled.patch,
    )
    compiled.patch.token_delta = estimate_token_delta(prompt, rendered_prompt)
    canary_results = await run_canaries(
        rendered_prompt,
        compiled.canaries,
        target_client,
        call_ledger,
        concurrency=canary_concurrency,
        registry=validator_registry,
    )
    outcome = evaluate_falsification_gate(
        prompt,
        rendered_prompt,
        compiled.patch,
        compiled.canaries,
        canary_results,
        no_patch_diagnosis=compiled.no_patch_diagnosis,
        config=gate_config,
    )

    if outcome.accepted:
        return OptimizationReport(
            original_prompt=prompt,
            rendered_prompt=rendered_prompt,
            accepted=True,
            decision=GateDecision.ACCEPTED,
            graph=compiled.graph,
            defect_posterior=compiled.defect_posterior,
            patch=compiled.patch,
            canaries=compiled.canaries,
            canary_results=canary_results,
            call_ledger=call_ledger,
            deterministic_validator_passed=True,
            notes=outcome.reasons,
            metadata={"gate_outcome": outcome.model_dump(mode="json")},
        )

    if allow_one_repair and should_attempt_repair(outcome):
        repair_report = await _attempt_one_repair(
            prompt=prompt,
            task_spec=task_spec,
            rubric=rubric,
            compiled=compiled,
            rendered_prompt=rendered_prompt,
            canary_results=canary_results,
            compiler=compiler,
            target_client=target_client,
            call_ledger=call_ledger,
            gate_config=gate_config,
            validator_registry=validator_registry,
            canary_concurrency=canary_concurrency,
        )
        if repair_report.accepted:
            return repair_report
        return repair_report

    diagnosis = outcome.no_patch_diagnosis or make_no_patch_diagnosis(outcome)
    return OptimizationReport(
        original_prompt=prompt,
        rendered_prompt=prompt,
        accepted=False,
        decision=outcome.decision,
        graph=compiled.graph,
        defect_posterior=compiled.defect_posterior,
        patch=compiled.patch,
        canaries=compiled.canaries,
        canary_results=canary_results,
        call_ledger=call_ledger,
        deterministic_validator_passed=False,
        no_patch_diagnosis=diagnosis,
        notes=outcome.reasons,
        metadata={"gate_outcome": outcome.model_dump(mode="json")},
    )


async def _run_probe_calls(
    prompt: str,
    probes: Sequence[Mapping[str, Any]],
    target_client: TargetClient,
    call_ledger: CallLedger,
) -> Sequence[Any]:
    async def run_one(probe: Mapping[str, Any]) -> Any:
        response = await target_client.complete(
            prompt,
            probe.get("input"),
            metadata={"phase": "cold_start_probe", "probe_id": probe.get("probe_id")},
        )
        call_ledger.add(response.call_record)
        return response.output

    return await asyncio.gather(*(run_one(probe) for probe in probes))


async def _attempt_one_repair(
    prompt: str,
    task_spec: str,
    rubric: str,
    compiled: CompilerOutput,
    rendered_prompt: str,
    canary_results: Sequence[CanaryResult],
    compiler: ContractCompiler,
    target_client: TargetClient,
    call_ledger: CallLedger,
    gate_config: Optional[FalsificationGateConfig],
    validator_registry: Optional[ValidatorRegistry],
    canary_concurrency: int,
) -> OptimizationReport:
    failed_results = [result for result in canary_results if not result.passed]
    repair = await compiler.repair(
        prompt,
        task_spec,
        rubric,
        prior=compiled,
        failed_results=failed_results,
        call_ledger=call_ledger,
    )
    repair_prompt = render_prompt(
        repair.graph,
        source_prompt=prompt,
        patch=repair.patch,
    )
    repair.patch.token_delta = estimate_token_delta(prompt, repair_prompt)
    retest_canaries = repair.retest_canaries or select_retest_canaries(
        compiled.canaries,
        canary_results,
    )
    retest_results = await run_canaries(
        repair_prompt,
        retest_canaries,
        target_client,
        call_ledger,
        concurrency=canary_concurrency,
        registry=validator_registry,
    )
    repair_gate_config = (gate_config or FalsificationGateConfig()).model_copy(
        update={"require_fix_canary": False, "require_format_canary": False}
    )
    repair_outcome = evaluate_falsification_gate(
        prompt,
        repair_prompt,
        repair.patch,
        retest_canaries,
        retest_results,
        no_patch_diagnosis=repair.no_patch_diagnosis,
        config=repair_gate_config,
    )

    if repair_outcome.accepted:
        return OptimizationReport(
            original_prompt=prompt,
            rendered_prompt=repair_prompt,
            accepted=True,
            decision=GateDecision.REPAIRED,
            graph=repair.graph,
            defect_posterior=compiled.defect_posterior,
            patch=repair.patch,
            canaries=[*compiled.canaries, *retest_canaries],
            canary_results=[*canary_results, *retest_results],
            call_ledger=call_ledger,
            deterministic_validator_passed=True,
            notes=repair_outcome.reasons,
            metadata={"gate_outcome": repair_outcome.model_dump(mode="json")},
        )

    diagnosis = repair_outcome.no_patch_diagnosis or make_no_patch_diagnosis(
        repair_outcome,
        reason="initial patch and one targeted repair failed falsification",
    )
    return OptimizationReport(
        original_prompt=prompt,
        rendered_prompt=prompt,
        accepted=False,
        decision=GateDecision.REJECTED_NO_PATCH,
        graph=repair.graph,
        defect_posterior=compiled.defect_posterior,
        patch=repair.patch,
        canaries=[*compiled.canaries, *retest_canaries],
        canary_results=[*canary_results, *retest_results],
        call_ledger=call_ledger,
        deterministic_validator_passed=False,
        no_patch_diagnosis=diagnosis,
        notes=repair_outcome.reasons,
        metadata={"gate_outcome": repair_outcome.model_dump(mode="json")},
    )
