"""Minimal deterministic PACT-EL run with replay clients."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact_el.clients import ReplayOptimizerClient, ReplayTargetClient
from pact_el.compiler import CompilerOutput
from pact_el.optimize import pact_optimize
from pact_el.schemas import (
    AxiomNode,
    AxiomNodeType,
    AxiomPatch,
    Canary,
    CanaryKind,
    ContractArea,
    DefectHypothesis,
    GuaranteeClause,
    GuaranteeScript,
    PromptAxiomGraph,
    ValidatorKind,
    ValidatorSpec,
)


def make_validator() -> ValidatorSpec:
    return ValidatorSpec(
        validator_id="priority_enum",
        kind=ValidatorKind.ENUM,
        config={
            "field": "parsed_output.priority",
            "allowed": ["low", "medium", "high", "urgent"],
        },
    )


def make_compiler_output() -> CompilerOutput:
    canaries = [
        Canary(
            canary_id="fix",
            kind=CanaryKind.FIX,
            input={"ticket": "Production outage for a paying customer."},
            expected_behavior="Return urgent priority.",
            target_contract_area=ContractArea.OUTPUT_SCHEMA,
            validator_specs=[make_validator()],
        ),
        Canary(
            canary_id="boundary",
            kind=CanaryKind.BOUNDARY,
            input={"ticket": "No details supplied."},
            expected_behavior="Return a supported priority without inventing details.",
            target_contract_area=ContractArea.UNCERTAINTY_POLICY,
            validator_specs=[make_validator()],
        ),
        Canary(
            canary_id="regression",
            kind=CanaryKind.REGRESSION,
            input={"ticket": "Cosmetic UI typo."},
            expected_behavior="Return low priority.",
            target_contract_area=ContractArea.TASK_INTENT,
            validator_specs=[make_validator()],
        ),
        Canary(
            canary_id="format",
            kind=CanaryKind.FORMAT_SCHEMA,
            input={"ticket": "Billing page unavailable."},
            expected_behavior="Return JSON with priority.",
            target_contract_area=ContractArea.OUTPUT_SCHEMA,
            validator_specs=[make_validator()],
        ),
    ]
    return CompilerOutput(
        graph=PromptAxiomGraph(
            nodes=[
                AxiomNode(
                    node_id="task",
                    node_type=AxiomNodeType.TASK_INTENT,
                    title="Classify support tickets",
                    statement="Classify the ticket into exactly one supported priority.",
                    priority=10,
                    guarantee_ids=["priority_enum"],
                )
            ],
            edges=[],
            guarantee_script=GuaranteeScript(
                clauses=[
                    GuaranteeClause(
                        guarantee_id="priority_enum",
                        description="Priority must be one of low, medium, high, or urgent.",
                        predicate={
                            "in": [
                                {"var": "parsed_output.priority"},
                                ["low", "medium", "high", "urgent"],
                            ]
                        },
                    )
                ]
            ),
        ),
        defect_posterior=[
            DefectHypothesis(
                defect_id="d1",
                suspected_contract_area=ContractArea.OUTPUT_SCHEMA,
                description="Priority format is underspecified.",
                supporting_evidence_ids=[],
                confidence=0.9,
                severity="high",
                expected_gain=0.7,
                regression_risk=0.1,
                prompt_fixability=0.95,
            )
        ],
        patch=AxiomPatch(
            patch_id="p1",
            summary="Make priority enum executable.",
            defect_id="d1",
            graph_edits=[],
            token_delta=30,
            expected_fixed_behaviors=["Priority is always in the enum."],
            expected_unchanged_behaviors=["Ticket classification stays semantically correct."],
            regression_risks=[],
            canary_ids=["fix", "boundary", "regression", "format"],
            rollback_rule="Rollback if enum validation fails.",
        ),
        canaries=canaries,
    )


async def main() -> None:
    compiler_output = make_compiler_output()
    report = await pact_optimize(
        prompt="Classify support tickets by priority.",
        task_spec="Return JSON with a priority field.",
        rubric="Valid JSON and correct priority.",
        optimizer_client=ReplayOptimizerClient([compiler_output.model_dump_json()]),
        target_client=ReplayTargetClient(
            [
                '{"priority": "low"}',
                '{"priority": "medium"}',
                '{"priority": "urgent"}',
                '{"priority": "medium"}',
                '{"priority": "low"}',
                '{"priority": "high"}',
                '{"priority": "urgent"}',
                '{"priority": "medium"}',
            ]
        ),
    )
    print(report.accepted)
    print(report.decision.value)


if __name__ == "__main__":
    asyncio.run(main())
