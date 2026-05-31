"""Optimizer prompt construction and strict compiler-output parsing."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, Field, model_validator

from pact_el.canaries import require_four_canary_families
from pact_el.clients import OptimizerClient
from pact_el.ledger import EvidenceLedger
from pact_el.schemas import (
    AxiomPatch,
    CallLedger,
    Canary,
    CanaryResult,
    DefectHypothesis,
    NoPatchDiagnosis,
    PromptAxiomGraph,
    StrictModel,
)


class CompilerOutput(StrictModel):
    graph: PromptAxiomGraph
    defect_posterior: List[DefectHypothesis]
    patch: AxiomPatch
    canaries: List[Canary] = Field(min_length=4)
    rendered_prompt: Optional[str] = None
    no_patch_diagnosis: Optional[NoPatchDiagnosis] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_canaries_and_patch(self) -> "CompilerOutput":
        require_four_canary_families(self.canaries)
        canary_ids = {canary.canary_id for canary in self.canaries}
        unknown = set(self.patch.canary_ids) - canary_ids
        if unknown:
            raise ValueError(
                f"patch references unknown canary ids: {sorted(unknown)}"
            )
        return self


class RepairOutput(StrictModel):
    graph: PromptAxiomGraph
    patch: AxiomPatch
    retest_canaries: List[Canary] = Field(min_length=1, max_length=2)
    rendered_prompt: Optional[str] = None
    no_patch_diagnosis: Optional[NoPatchDiagnosis] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


ModelT = TypeVar("ModelT", bound=BaseModel)


def strict_parse_json_model(model: Type[ModelT], payload: Any) -> ModelT:
    """Parse model JSON without markdown stripping, retries, or repair."""

    if isinstance(payload, model):
        return payload
    if isinstance(payload, Mapping):
        return model.model_validate(payload)
    if not isinstance(payload, str):
        raise TypeError(f"expected JSON string or mapping, got {type(payload).__name__}")
    stripped = payload.strip()
    if not stripped.startswith("{"):
        raise ValueError("optimizer output must be a bare JSON object, not fenced text")
    return model.model_validate_json(stripped)


def compiler_response_schema() -> Dict[str, Any]:
    return CompilerOutput.model_json_schema()


def repair_response_schema() -> Dict[str, Any]:
    return RepairOutput.model_json_schema()


def build_compiler_messages(
    prompt: str,
    task_spec: str,
    rubric: str,
    ledger: EvidenceLedger,
    budget: int = 9,
) -> List[Dict[str, str]]:
    schema = json.dumps(compiler_response_schema(), indent=2, sort_keys=True)
    ledger_payload = json.dumps(ledger.to_optimizer_table(), indent=2, sort_keys=True)
    summary_payload = json.dumps(ledger.compact_summary(), indent=2, sort_keys=True)
    system = (
        "You are the PACT-EL optimizer compiler. Your job is not to rewrite "
        "the prompt broadly. Infer the smallest transferable behavioral contract "
        "that explains the evidence, represent it as a typed Prompt Axiom Graph, "
        "apply exactly one minimal AxiomPatch, and design four falsification "
        "canaries: fix, boundary, regression, and format_schema. Return only "
        "valid JSON matching the supplied schema."
    )
    user = f"""
PACT-EL compile request

Objective:
- Produce a contract patch that transfers to unseen validation and test examples.
- Improve behavior by adding operational rules, precedence rules, output-shape rules, or uncertainty policies that the target model can execute.
- Use training labels only to infer reusable behavior. Do not put per-example answers, example ids, memorized labels, or copied training inputs into graph node statements, patch summaries, or rendered-prompt diffs.

Hard rules:
- Use one patch, not a candidate-search loop.
- Preserve behavior that is not implicated by the defect posterior.
- Treat success rows as regression constraints and failure rows as evidence for a general defect.
- Prefer defects with repeated support, high prompt-fixability, high expected gain, and low regression risk.
- Never convert a constraint observed in only some evidence rows into an unconditional global runtime requirement. Express it as a conditional policy such as "when the current prompt requests this constraint, verify it this way."
- Graph node statements must generalize across unseen inputs. Do not make every future answer use a specific bracket pattern, vowel set, word count, schema, answer style, or topic unless the task specification itself always requires it.
- For heterogeneous benchmark tasks, prefer meta-rules for detecting and satisfying current-input constraints over narrow rules copied from failed examples.
- Distinguish prompt-fixable failures from model-knowledge, tool, retrieval, evaluator, or impossible-constraint failures.
- Put hard clauses in GuaranteeScript predicates when deterministic validation is possible.
- GuaranteeScript predicates must use the supported JSONLogic subset only: var, if, and, or, !/not, !!, ==, !=, <, <=, >, >=, in, missing, missing_some, cat, substr, +, -, *, /, %, max, min, all, some, none. Do not invent keys such as type, all_of, clauses, regex, or description inside predicates.
- Every canary must have at least one deterministic validator, and canary inputs must be fresh stress cases rather than copied evidence rows.
- Canary inputs, expected_behavior, and validators must be internally consistent. Do not make a canary whose prompt asks for one output while the validator expects a different output.
- Regex validators must use Python-compatible regular expressions. The flags field, when needed, must be a list such as ["dotall", "ignorecase"].
- Exact-match validators are only appropriate when the canary prompt fully determines a single exact output. Otherwise use schema, enum, numeric, or regex validators that check the contract without over-constraining content.
- Do not claim to fix retrieval gaps, missing knowledge, bad tools, invalid upstream data, or impossible constraints.
- If the evidence is not prompt-fixable, include a no_patch_diagnosis and make the patch a conservative diagnostic patch.
- Do not output markdown fences or explanatory prose.
- Optimization budget target: {budget} total calls including target probes/canaries.

Original prompt:
{prompt}

Task specification:
{task_spec}

Rubric:
{rubric}

Evidence Ledger summary:
{summary_payload}

Evidence Ledger rows:
{ledger_payload}

Required JSON schema:
{schema}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    prompt: str,
    task_spec: str,
    rubric: str,
    prior: CompilerOutput,
    failed_results: Sequence[CanaryResult],
) -> List[Dict[str, str]]:
    schema = json.dumps(repair_response_schema(), indent=2, sort_keys=True)
    prior_payload = prior.model_dump_json(indent=2)
    failed_payload = json.dumps(
        [result.model_dump(mode="json") for result in failed_results],
        indent=2,
        sort_keys=True,
    )
    system = (
        "You are the PACT-EL targeted repair compiler. Edit only the falsified "
        "axiom, edge, or guarantee clause. Return one repair patch and one or "
        "two retest canaries. Return only valid JSON matching the schema."
    )
    user = f"""
PACT-EL targeted repair request

Hard rules:
- Do not perform a broad rewrite.
- Edit only the falsified axiom, edge, or guarantee.
- Retest only the failed canary plus one regression canary.
- Keep repaired graph clauses conditional on the current user prompt. Do not make evidence-specific constraints apply to unrelated future inputs.
- If the failure is not prompt-fixable, return a no_patch_diagnosis.
- Do not output markdown fences or explanatory prose.

Original prompt:
{prompt}

Task specification:
{task_spec}

Rubric:
{rubric}

Prior compiler output:
{prior_payload}

Failed canary results:
{failed_payload}

Required JSON schema:
{schema}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class ContractCompiler:
    """High-level compiler facade around an OptimizerClient."""

    def __init__(self, optimizer_client: OptimizerClient):
        self.optimizer_client = optimizer_client

    async def compile(
        self,
        prompt: str,
        task_spec: str,
        rubric: str,
        ledger: EvidenceLedger,
        call_ledger: Optional[CallLedger] = None,
        budget: int = 9,
    ) -> CompilerOutput:
        messages = build_compiler_messages(prompt, task_spec, rubric, ledger, budget)
        response = await self.optimizer_client.complete(
            messages,
            response_schema=compiler_response_schema(),
            metadata={"phase": "compile"},
        )
        if call_ledger is not None:
            call_ledger.add(response.call_record)
        return strict_parse_json_model(CompilerOutput, response.output)

    async def repair(
        self,
        prompt: str,
        task_spec: str,
        rubric: str,
        prior: CompilerOutput,
        failed_results: Sequence[CanaryResult],
        call_ledger: Optional[CallLedger] = None,
    ) -> RepairOutput:
        messages = build_repair_messages(
            prompt,
            task_spec,
            rubric,
            prior,
            failed_results,
        )
        response = await self.optimizer_client.complete(
            messages,
            response_schema=repair_response_schema(),
            metadata={"phase": "repair"},
        )
        if call_ledger is not None:
            call_ledger.add(response.call_record)
        return strict_parse_json_model(RepairOutput, response.output)
