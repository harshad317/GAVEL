"""Strict data models for PACT-EL.

The models in this module are intentionally narrow. Optimizer output is expected
to validate against these schemas directly; callers should not silently repair,
coerce, or infer missing contract fields from malformed LLM JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    """Base model that forbids accidental schema drift."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        use_enum_values=False,
    )


class EvidenceSource(str, Enum):
    LOG = "log"
    EXAMPLE = "example"
    PROBE = "probe"
    CANARY = "canary"
    JUDGE = "judge"
    MANUAL = "manual"


class EvidenceOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_WEIGHT: Dict[Severity, float] = {
    Severity.INFO: 0.05,
    Severity.LOW: 0.2,
    Severity.MEDIUM: 0.45,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.0,
}


class ContractArea(str, Enum):
    TASK_INTENT = "task_intent"
    OUTPUT_SCHEMA = "output_schema"
    DECISION_RULE = "decision_rule"
    REASONING_POLICY = "reasoning_policy"
    TOOL_CONTRACT = "tool_contract"
    RETRIEVAL_CONTRACT = "retrieval_contract"
    SAFETY_BOUNDARY = "safety_boundary"
    UNCERTAINTY_POLICY = "uncertainty_policy"
    FORMATTING_RULE = "formatting_rule"
    PRECEDENCE_RULE = "precedence_rule"
    DEMONSTRATION_ROLE = "demonstration_role"
    ANTI_PATTERN = "anti_pattern"
    MODEL_CEILING_NOTICE = "model_ceiling_notice"


class AxiomNodeType(str, Enum):
    TASK_INTENT = "task_intent"
    OUTPUT_SCHEMA = "output_schema"
    DECISION_RULE = "decision_rule"
    REASONING_POLICY = "reasoning_policy"
    TOOL_CONTRACT = "tool_contract"
    RETRIEVAL_CONTRACT = "retrieval_contract"
    SAFETY_BOUNDARY = "safety_boundary"
    UNCERTAINTY_POLICY = "uncertainty_policy"
    FORMATTING_RULE = "formatting_rule"
    PRECEDENCE_RULE = "precedence_rule"
    DEMONSTRATION_ROLE = "demonstration_role"
    ANTI_PATTERN = "anti_pattern"
    MODEL_CEILING_NOTICE = "model_ceiling_notice"


class AxiomEdgeType(str, Enum):
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"
    OVERRIDES = "overrides"
    CONTRADICTS = "contradicts"
    WEAKENS = "weakens"
    DUPLICATES = "duplicates"
    RISKS_REGRESSION_IN = "risks_regression_in"


class PatchOperation(str, Enum):
    ADD_NODE = "add_node"
    UPDATE_NODE = "update_node"
    REMOVE_NODE = "remove_node"
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    MERGE_NODES = "merge_nodes"
    ADD_GUARANTEE = "add_guarantee"
    UPDATE_GUARANTEE = "update_guarantee"
    REMOVE_GUARANTEE = "remove_guarantee"


class CanaryKind(str, Enum):
    FIX = "fix"
    BOUNDARY = "boundary"
    REGRESSION = "regression"
    FORMAT_SCHEMA = "format_schema"


class ValidatorKind(str, Enum):
    JSON_SCHEMA = "jsonschema"
    REGEX = "regex"
    EXACT_MATCH = "exact_match"
    ENUM = "enum"
    NUMERIC = "numeric"
    JSON_LOGIC = "jsonlogic"
    LLM_JUDGE = "llm_judge"


class CanaryStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class GateDecision(str, Enum):
    ACCEPTED = "accepted"
    REPAIRED = "repaired"
    REJECTED_NO_PATCH = "rejected_no_patch"
    REJECTED_REGRESSION = "rejected_regression"
    REJECTED_VALIDATION = "rejected_validation"


class CallRole(str, Enum):
    OPTIMIZER = "optimizer"
    TARGET = "target"
    JUDGE = "judge"
    VALIDATOR = "validator"


class EvidenceRow(StrictModel):
    evidence_id: str = Field(default_factory=lambda: _id("ev"))
    source: EvidenceSource
    input: Any = None
    expected_behavior: str
    observed_behavior: str
    output: Any = None
    success_or_failure: EvidenceOutcome
    severity: Severity = Severity.MEDIUM
    suspected_contract_area: Optional[ContractArea] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    prompt_fixability: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_behavior", "observed_behavior")
    @classmethod
    def _non_empty_behavior(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("behavior fields must be non-empty")
        return value


class GuaranteeClause(StrictModel):
    guarantee_id: str = Field(default_factory=lambda: _id("guarantee"))
    description: str
    predicate: Dict[str, Any]
    applies_to: List[str] = Field(default_factory=list)
    severity: Severity = Severity.HIGH
    violation_message: str = "Guarantee clause evaluated to false."
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("description")
    @classmethod
    def _description_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("guarantee description is required")
        return value


class GuaranteeScript(StrictModel):
    version: str = "pact-el.guarantee-script.v1"
    clauses: List[GuaranteeClause] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_clause_ids(self) -> "GuaranteeScript":
        ids = [clause.guarantee_id for clause in self.clauses]
        if len(ids) != len(set(ids)):
            raise ValueError("guarantee clause ids must be unique")
        return self


class AxiomNode(StrictModel):
    node_id: str
    node_type: AxiomNodeType
    title: str
    statement: str
    rationale: str = ""
    priority: int = Field(default=100, ge=0)
    required: bool = True
    guarantee_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("node_id", "title", "statement")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("node fields must be non-empty")
        return value


class AxiomEdge(StrictModel):
    edge_id: str = Field(default_factory=lambda: _id("edge"))
    source_id: str
    target_id: str
    edge_type: AxiomEdgeType
    rationale: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "target_id")
    @classmethod
    def _edge_endpoint_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("edge endpoints must be non-empty")
        return value


class PromptAxiomGraph(StrictModel):
    nodes: List[AxiomNode]
    edges: List[AxiomEdge] = Field(default_factory=list)
    guarantee_script: GuaranteeScript = Field(default_factory=GuaranteeScript)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_references(self) -> "PromptAxiomGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("axiom node ids must be unique")
        node_id_set = set(node_ids)
        for edge in self.edges:
            if edge.source_id not in node_id_set or edge.target_id not in node_id_set:
                raise ValueError(
                    f"edge {edge.edge_id} references unknown node "
                    f"{edge.source_id!r}->{edge.target_id!r}"
                )
            if edge.source_id == edge.target_id:
                raise ValueError(f"edge {edge.edge_id} cannot point to itself")

        guarantee_ids = {
            clause.guarantee_id for clause in self.guarantee_script.clauses
        }
        for node in self.nodes:
            missing = set(node.guarantee_ids) - guarantee_ids
            if missing:
                raise ValueError(
                    f"node {node.node_id} references unknown guarantees: {sorted(missing)}"
                )
        return self


class DefectHypothesis(StrictModel):
    defect_id: str = Field(default_factory=lambda: _id("defect"))
    suspected_contract_area: ContractArea
    description: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity
    expected_gain: float = Field(ge=0.0, le=1.0)
    regression_risk: float = Field(ge=0.0, le=1.0)
    prompt_fixability: float = Field(ge=0.0, le=1.0)
    affected_node_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AxiomEdit(StrictModel):
    operation: PatchOperation
    node: Optional[AxiomNode] = None
    edge: Optional[AxiomEdge] = None
    guarantee: Optional[GuaranteeClause] = None
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    guarantee_id: Optional[str] = None
    before: Optional[Any] = None
    after: Optional[Any] = None
    rationale: str = ""

    @model_validator(mode="after")
    def _operation_has_target(self) -> "AxiomEdit":
        has_target = any(
            [
                self.node is not None,
                self.edge is not None,
                self.guarantee is not None,
                self.node_id is not None,
                self.edge_id is not None,
                self.guarantee_id is not None,
            ]
        )
        if not has_target:
            raise ValueError("axiom edit must include a target object or id")
        return self


class AxiomPatch(StrictModel):
    patch_id: str = Field(default_factory=lambda: _id("patch"))
    summary: str
    defect_id: Optional[str] = None
    graph_edits: List[AxiomEdit] = Field(default_factory=list)
    rendered_prompt_diff: List[str] = Field(default_factory=list)
    clauses_added: List[str] = Field(default_factory=list)
    clauses_removed: List[str] = Field(default_factory=list)
    clauses_merged: List[str] = Field(default_factory=list)
    token_delta: int = 0
    expected_fixed_behaviors: List[str] = Field(default_factory=list)
    expected_unchanged_behaviors: List[str] = Field(default_factory=list)
    regression_risks: List[str] = Field(default_factory=list)
    canary_ids: List[str] = Field(default_factory=list)
    rollback_rule: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary", "rollback_rule")
    @classmethod
    def _patch_text_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("patch summary and rollback rule are required")
        return value


class ValidatorSpec(StrictModel):
    validator_id: str = Field(default_factory=lambda: _id("validator"))
    kind: ValidatorKind
    config: Dict[str, Any] = Field(default_factory=dict)
    severity: Severity = Severity.HIGH
    description: str = ""


class ValidatorResult(StrictModel):
    validator_id: str
    kind: ValidatorKind
    passed: bool
    message: str
    observed: Any = None
    expected: Any = None
    details: Dict[str, Any] = Field(default_factory=dict)


class Canary(StrictModel):
    canary_id: str = Field(default_factory=lambda: _id("canary"))
    kind: CanaryKind
    input: Any
    expected_behavior: str
    target_contract_area: ContractArea
    validator_specs: List[ValidatorSpec] = Field(default_factory=list)
    max_target_calls: int = Field(default=1, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_behavior")
    @classmethod
    def _expected_behavior_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("canary expected_behavior is required")
        return value


class CanaryResult(StrictModel):
    canary_id: str
    kind: CanaryKind
    status: CanaryStatus
    passed: bool
    input: Any = None
    output: Any = None
    validator_outputs: List[ValidatorResult] = Field(default_factory=list)
    failure_reason: str = ""
    latency_ms: Optional[float] = None
    target_call_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _status_matches_passed(self) -> "CanaryResult":
        if self.status == CanaryStatus.PASSED and not self.passed:
            raise ValueError("passed canary status requires passed=True")
        if self.status == CanaryStatus.FAILED and self.passed:
            raise ValueError("failed canary status requires passed=False")
        return self


class CallRecord(StrictModel):
    call_id: str = Field(default_factory=lambda: _id("call"))
    role: CallRole
    name: str
    model: str = ""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    finished_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CallLedger(StrictModel):
    records: List[CallRecord] = Field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.records.append(record)

    def calls_by_role(self, role: CallRole) -> List[CallRecord]:
        return [record for record in self.records if record.role == role]

    def count_role(self, role: CallRole) -> int:
        return len(self.calls_by_role(role))

    @property
    def optimizer_calls(self) -> int:
        return self.count_role(CallRole.OPTIMIZER)

    @property
    def target_calls(self) -> int:
        return self.count_role(CallRole.TARGET)

    @property
    def judge_calls(self) -> int:
        return self.count_role(CallRole.JUDGE)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.records)

    @property
    def token_totals(self) -> Dict[str, int]:
        return {
            "prompt_tokens": sum(record.prompt_tokens for record in self.records),
            "completion_tokens": sum(
                record.completion_tokens for record in self.records
            ),
        }


class NoPatchDiagnosis(StrictModel):
    reason: str
    non_prompt_fixable_area: Optional[ContractArea] = None
    evidence_ids: List[str] = Field(default_factory=list)
    recommended_owner: str = "prompt"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizationReport(StrictModel):
    run_id: str = Field(default_factory=lambda: _id("run"))
    original_prompt: str
    rendered_prompt: str
    accepted: bool
    decision: GateDecision
    graph: Optional[PromptAxiomGraph] = None
    defect_posterior: List[DefectHypothesis] = Field(default_factory=list)
    patch: Optional[AxiomPatch] = None
    canaries: List[Canary] = Field(default_factory=list)
    canary_results: List[CanaryResult] = Field(default_factory=list)
    call_ledger: CallLedger = Field(default_factory=CallLedger)
    deterministic_validator_passed: bool = False
    no_patch_diagnosis: Optional[NoPatchDiagnosis] = None
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

