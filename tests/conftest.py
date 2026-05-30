from __future__ import annotations

import pytest

from pact_el.compiler import CompilerOutput
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


def make_priority_validator() -> ValidatorSpec:
    return ValidatorSpec(
        validator_id="priority_enum",
        kind=ValidatorKind.ENUM,
        config={
            "field": "parsed_output.priority",
            "allowed": ["low", "medium", "high", "urgent"],
        },
    )


def make_four_canaries() -> list[Canary]:
    return [
        Canary(
            canary_id="fix",
            kind=CanaryKind.FIX,
            input={"ticket": "prod outage"},
            expected_behavior="urgent priority",
            target_contract_area=ContractArea.OUTPUT_SCHEMA,
            validator_specs=[make_priority_validator()],
        ),
        Canary(
            canary_id="boundary",
            kind=CanaryKind.BOUNDARY,
            input={"ticket": ""},
            expected_behavior="supported priority without invented facts",
            target_contract_area=ContractArea.UNCERTAINTY_POLICY,
            validator_specs=[make_priority_validator()],
        ),
        Canary(
            canary_id="regression",
            kind=CanaryKind.REGRESSION,
            input={"ticket": "typo"},
            expected_behavior="low priority remains possible",
            target_contract_area=ContractArea.TASK_INTENT,
            validator_specs=[make_priority_validator()],
        ),
        Canary(
            canary_id="format",
            kind=CanaryKind.FORMAT_SCHEMA,
            input={"ticket": "billing outage"},
            expected_behavior="JSON with priority",
            target_contract_area=ContractArea.OUTPUT_SCHEMA,
            validator_specs=[make_priority_validator()],
        ),
    ]


def make_sample_graph() -> PromptAxiomGraph:
    return PromptAxiomGraph(
        nodes=[
            AxiomNode(
                node_id="task",
                node_type=AxiomNodeType.TASK_INTENT,
                title="Classify tickets",
                statement="Classify each ticket into exactly one priority.",
                priority=10,
                guarantee_ids=["priority_enum"],
            ),
            AxiomNode(
                node_id="format",
                node_type=AxiomNodeType.OUTPUT_SCHEMA,
                title="JSON priority",
                statement="Return a JSON object containing priority.",
                priority=20,
                guarantee_ids=["priority_enum"],
            ),
        ],
        guarantee_script=GuaranteeScript(
            clauses=[
                GuaranteeClause(
                    guarantee_id="priority_enum",
                    description="Priority must be in the supported enum.",
                    predicate={
                        "in": [
                            {"var": "parsed_output.priority"},
                            ["low", "medium", "high", "urgent"],
                        ]
                    },
                )
            ]
        ),
    )


def make_sample_patch() -> AxiomPatch:
    return AxiomPatch(
        patch_id="patch",
        summary="Make the priority enum explicit and executable.",
        defect_id="defect",
        graph_edits=[],
        token_delta=10,
        expected_fixed_behaviors=["priority is always in enum"],
        expected_unchanged_behaviors=["classification semantics remain"],
        regression_risks=[],
        canary_ids=["fix", "boundary", "regression", "format"],
        rollback_rule="Rollback if any canary fails.",
    )


def make_sample_compiler_output() -> CompilerOutput:
    return CompilerOutput(
        graph=make_sample_graph(),
        defect_posterior=[
            DefectHypothesis(
                defect_id="defect",
                suspected_contract_area=ContractArea.OUTPUT_SCHEMA,
                description="Output enum is underspecified.",
                supporting_evidence_ids=["log_0"],
                confidence=0.9,
                severity="high",
                expected_gain=0.8,
                regression_risk=0.1,
                prompt_fixability=0.95,
            )
        ],
        patch=make_sample_patch(),
        canaries=make_four_canaries(),
    )


@pytest.fixture
def priority_validator() -> ValidatorSpec:
    return make_priority_validator()


@pytest.fixture
def four_canaries() -> list[Canary]:
    return make_four_canaries()


@pytest.fixture
def sample_graph() -> PromptAxiomGraph:
    return make_sample_graph()


@pytest.fixture
def sample_patch() -> AxiomPatch:
    return make_sample_patch()


@pytest.fixture
def sample_compiler_output() -> CompilerOutput:
    return make_sample_compiler_output()
