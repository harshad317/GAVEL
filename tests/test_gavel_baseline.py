from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import pytest
import asyncio

from pact_el.baselines.gavel import (
    GavelConfig,
    default_base_prompt,
    evaluate_prompt,
    run_gavel_baseline,
)
from pact_el.benchmarks.schemas import BenchmarkExample, BenchmarkSpec, BenchmarkTaskType, MetricKind
from pact_el.clients import ClientResponse
from pact_el.schemas import CallRecord, CallRole


class FakeOptimizerClient:
    def __init__(self, output: Any):
        self.outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        self.calls = 0
        self.model = "fake-optimizer"

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del messages, response_schema
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.OPTIMIZER, "optimizer_compile", metadata),
        )


class FakeTargetClient:
    model = "fake-target"

    def __init__(self):
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del prompt
        self.calls += 1
        if isinstance(input, Mapping):
            output = '{"priority": "urgent"}'
        elif "2+2" in str(input):
            output = "4"
        else:
            output = "A"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class SlowTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        await asyncio.sleep(0.05)
        return await super().complete(prompt, input, metadata=metadata)


class ValidationRegressionTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del prompt
        self.calls += 1
        phase = (metadata or {}).get("phase")
        if (metadata or {}).get("canary_id"):
            output = '{"priority": "urgent"}'
        elif str(phase).startswith("validation_gate_") and phase != "validation_gate_base":
            output = "5"
        else:
            output = "4" if "2+2" in str(input) else "A"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class RejectedCandidateValidationTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        self.calls += 1
        phase = (metadata or {}).get("phase")
        if (metadata or {}).get("canary_id"):
            output = "not valid json"
        elif phase == "validation_gate_base":
            output = "5"
        elif phase == "validation_gate_optimized" and "## Constraints" in prompt:
            output = "4"
        elif "2+2" in str(input):
            output = "4" if "## Constraints" in prompt else "5"
        else:
            output = "A"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class PromptPortfolioTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        self.calls += 1
        phase = (metadata or {}).get("phase")
        if (metadata or {}).get("canary_id"):
            output = '{"priority": "urgent"}'
        elif phase == "validation_gate_task_strategy":
            output = "4"
        elif phase == "final_test" and "deterministic solve-and-verify loop" in prompt:
            output = "4"
        elif "2+2" in str(input):
            output = "5"
        else:
            output = "A"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class ParetoTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        self.calls += 1
        phase = str((metadata or {}).get("phase"))
        if (metadata or {}).get("canary_id"):
            output = '{"priority": "urgent"}'
        elif phase.startswith("validation_gate_pareto_"):
            output = "4"
        elif phase == "final_test" and "pareto arithmetic contract" in prompt:
            output = "4"
        elif "2+2" in str(input):
            output = "5"
        else:
            output = "A"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class SelfRefineTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del prompt
        self.calls += 1
        phase = (metadata or {}).get("phase")
        if phase == "test_self_refine_1":
            output = "4"
        else:
            output = "5"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class PlanTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del input
        self.calls += 1
        phase = str((metadata or {}).get("phase"))
        if (metadata or {}).get("canary_id"):
            output = '{"priority": "urgent"}'
        elif phase.endswith("_plan_contract"):
            output = '{"goal":"answer arithmetic","answer_shape":"number","hard_constraints":[],"solve_plan":["compute"],"final_checks":["number only"]}'
        elif phase.endswith("_plan_answer"):
            output = "5" if "Classify tickets" in prompt else "4"
        elif phase.endswith("_plan_refine_1"):
            output = "4"
        else:
            output = "5"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


class PortfolioTargetClient(FakeTargetClient):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del prompt, input
        self.calls += 1
        phase = str((metadata or {}).get("phase"))
        if phase.endswith("_portfolio_select"):
            output = "4"
        elif phase.endswith("_plan_contract"):
            output = '{"goal":"answer arithmetic","answer_shape":"number","hard_constraints":[],"solve_plan":["compute"],"final_checks":["number only"]}'
        elif phase.endswith("_plan_answer"):
            output = "4"
        else:
            output = "5"
        return ClientResponse(
            output=output,
            raw=output,
            call_record=_record(CallRole.TARGET, "target_complete", metadata),
        )


def _record(
    role: CallRole,
    name: str,
    metadata: Optional[Mapping[str, Any]],
) -> CallRecord:
    return CallRecord(
        role=role,
        name=name,
        model="fake",
        prompt_tokens=1,
        completion_tokens=1,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        metadata=dict(metadata or {}),
    )


def _numeric_example(example_id: str) -> BenchmarkExample:
    return BenchmarkExample(
        benchmark_id="gsm8k",
        example_id=example_id,
        split="test",
        prompt="What is 2+2?",
        expected_answer="4",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="gsm8k",
        display_name="GSM8K",
        task_type=BenchmarkTaskType.MATH,
        adapter="gsm8k",
        default_split="test",
        metrics=[MetricKind.NUMERIC_EXACT],
        official_url="official",
        source_url="official",
        evaluator="numeric exact",
        sources=[],
        description="Grade-school math.",
    )


@pytest.mark.asyncio
async def test_gavel_baseline_compiles_and_reports_splits(tmp_path, sample_compiler_output):
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        temperature=0.2,
        optimizer_temperature=0.3,
        cache=False,
        workers=2,
        show_progress=False,
        self_refine_rounds=0,
        execution_modes=("direct",),
        pareto_candidates=0,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0"), _numeric_example("gsm8k:train:1")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(sample_compiler_output.model_dump_json()),
        target_client=FakeTargetClient(),
    )

    assert result.summary["optimizer"] == "gavel"
    assert result.summary["method"] == "gavel"
    assert result.summary["temperature"] == 0.2
    assert result.summary["optimizer_temperature"] == 0.3
    assert result.summary["accepted"] is False
    assert result.summary["split_results"]["train"]["score"] == 1.0
    assert result.summary["split_results"]["val"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0
    assert result.predictions_path.exists()
    assert result.scores_path.exists()
    assert result.report_path.exists()
    assert result.prompt_path.exists()


@pytest.mark.asyncio
async def test_gavel_validation_gate_rolls_back_regressing_prompt(tmp_path, sample_compiler_output):
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        cache=False,
        workers=2,
        show_progress=False,
        validation_gate=True,
        self_refine_rounds=0,
        execution_modes=("direct",),
        pareto_candidates=0,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(sample_compiler_output.model_dump_json()),
        target_client=ValidationRegressionTargetClient(),
    )

    gate = result.summary["validation_gate"]
    assert result.summary["selected_prompt"] == "base"
    assert result.summary["accepted"] is False
    assert result.summary["decision"] == "validation_rollback"
    assert gate["base"]["score"] == 1.0
    assert gate["candidate"]["score"] == 0.0
    assert result.summary["split_results"]["test"]["score"] == 1.0


@pytest.mark.asyncio
async def test_gavel_validation_gate_can_override_synthetic_canary_rejection(
    tmp_path,
    sample_compiler_output,
):
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        cache=False,
        workers=2,
        show_progress=False,
        allow_one_repair=False,
        validation_gate=True,
        validate_rejected_candidates=True,
        prompt_portfolio=False,
        self_refine_rounds=0,
        execution_modes=("direct",),
        pareto_candidates=0,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(sample_compiler_output.model_dump_json()),
        target_client=RejectedCandidateValidationTargetClient(),
    )

    gate = result.summary["validation_gate"]
    assert result.summary["selected_prompt"] == "optimized"
    assert result.summary["accepted"] is True
    assert result.summary["decision"] == "validation_override"
    assert gate["candidate_source"] == "rejected"
    assert gate["base"]["score"] == 0.0
    assert gate["candidate"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0


@pytest.mark.asyncio
async def test_gavel_validation_gate_can_select_task_strategy_from_portfolio(
    tmp_path,
    sample_compiler_output,
):
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        cache=False,
        workers=2,
        show_progress=False,
        validation_gate=True,
        prompt_portfolio=True,
        self_refine_rounds=0,
        execution_modes=("direct",),
        pareto_candidates=0,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(sample_compiler_output.model_dump_json()),
        target_client=PromptPortfolioTargetClient(),
    )

    gate = result.summary["validation_gate"]
    assert result.summary["selected_prompt"] == "task_strategy"
    assert result.summary["accepted"] is True
    assert result.summary["decision"] == "validation_override"
    assert "task_strategy" in gate["candidates"]
    assert gate["base"]["score"] == 0.0
    assert gate["candidate"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0


@pytest.mark.asyncio
async def test_gavel_pareto_candidate_can_win_validation(
    tmp_path,
    sample_compiler_output,
):
    pareto_prompt = (
        default_base_prompt(_spec())
        + "\n\n## Pareto Mutation\n"
        + "- pareto arithmetic contract: compute exact arithmetic and return only the normalized final answer."
    )
    mutation_batch = {
        "mutations": [
            {
                "mutation_id": "arith_contract",
                "title": "Arithmetic contract",
                "strategy": "decision_rule",
                "prompt": pareto_prompt,
                "expected_fixed_behaviors": ["Arithmetic answers use the computed value."],
                "expected_unchanged_behaviors": ["Return only the requested final answer."],
                "risk_notes": ["May be too terse for tasks that ask for explanations."],
                "evidence_ids": ["train_gsm8k:train:0"],
                "confidence": 0.8,
                "expected_gain": 0.9,
                "regression_risk": 0.1,
            }
        ],
        "no_mutation_reason": None,
    }
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        cache=False,
        workers=2,
        show_progress=False,
        validation_gate=True,
        prompt_portfolio=True,
        self_refine_rounds=0,
        execution_modes=("direct",),
        pareto_candidates=3,
        pareto_frontier_size=2,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(
            [sample_compiler_output.model_dump_json(), json.dumps(mutation_batch)]
        ),
        target_client=ParetoTargetClient(),
    )

    gate = result.summary["validation_gate"]
    assert result.summary["method"] == "gavel_pareto"
    assert result.summary["optimizer"] == "gavel_pareto"
    assert result.summary["selected_prompt"].startswith("pareto_")
    assert result.summary["decision"] == "validation_selected_pareto"
    assert result.summary["pareto_search"]["selected"] == 1
    assert gate["candidate_source"] == "pareto:decision_rule"
    assert result.summary["split_results"]["test"]["score"] == 1.0


def test_default_base_prompt_is_benchmark_specific():
    prompt = default_base_prompt(_spec())
    assert "benchmark-solving" in prompt
    assert "final answer" in prompt
    for heading in (
        "## Goal",
        "## Context",
        "## Role",
        "## Input",
        "## Task",
        "## Constraints",
        "## Output Format",
        "## Quality Bar",
    ):
        assert heading in prompt


def test_gavel_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        GavelConfig(temperature=2.1).validate()

    with pytest.raises(ValueError, match="optimizer_temperature must be between 0 and 2"):
        GavelConfig(optimizer_temperature=-0.1).validate()

    with pytest.raises(ValueError, match="validation_margin must be between 0 and 1"):
        GavelConfig(validation_margin=1.1).validate()

    with pytest.raises(ValueError, match="self_refine_rounds must be non-negative"):
        GavelConfig(self_refine_rounds=-1).validate()

    with pytest.raises(ValueError, match="pareto_candidates must be non-negative"):
        GavelConfig(pareto_candidates=-1).validate()

    with pytest.raises(ValueError, match="execution mode must be one of"):
        GavelConfig(execution_modes=("unsupported",)).validate()


@pytest.mark.asyncio
async def test_gavel_evaluation_uses_worker_concurrency():
    examples = [_numeric_example(f"gsm8k:test:{index}") for index in range(6)]

    _predictions, _scores, stats = await evaluate_prompt(
        prompt="answer",
        examples=examples,
        target_client=SlowTargetClient(),
        allow_code_execution=False,
        show_progress=False,
        workers=3,
        description="test",
        phase="test",
    )

    assert stats.requested_workers == 3
    assert stats.max_in_flight == 3


@pytest.mark.asyncio
async def test_gavel_evaluation_can_self_refine_without_scorer_feedback():
    predictions, scores, stats = await evaluate_prompt(
        prompt="answer",
        examples=[_numeric_example("gsm8k:test:0")],
        target_client=SelfRefineTargetClient(),
        allow_code_execution=False,
        show_progress=False,
        workers=1,
        description="test",
        phase="test",
        self_refine_rounds=1,
    )

    assert predictions[0]["prediction"] == "4"
    assert scores[0].score == 1.0
    assert stats.api_calls == 2


@pytest.mark.asyncio
async def test_gavel_evaluation_can_plan_then_answer_without_scorer_feedback():
    predictions, scores, stats = await evaluate_prompt(
        prompt="answer",
        examples=[_numeric_example("gsm8k:test:0")],
        target_client=PlanTargetClient(),
        allow_code_execution=False,
        show_progress=False,
        workers=1,
        description="test",
        phase="test",
        execution_mode="plan",
    )

    assert predictions[0]["prediction"] == "4"
    assert predictions[0]["execution_mode"] == "plan"
    assert scores[0].score == 1.0
    assert stats.api_calls == 2


@pytest.mark.asyncio
async def test_gavel_evaluation_can_plan_answer_then_refine_without_scorer_feedback():
    predictions, scores, stats = await evaluate_prompt(
        prompt="Classify tickets",
        examples=[_numeric_example("gsm8k:test:0")],
        target_client=PlanTargetClient(),
        allow_code_execution=False,
        show_progress=False,
        workers=1,
        description="test",
        phase="test",
        self_refine_rounds=1,
        execution_mode="plan_refine",
    )

    assert predictions[0]["prediction"] == "4"
    assert predictions[0]["execution_mode"] == "plan_refine"
    assert scores[0].score == 1.0
    assert stats.api_calls == 3


@pytest.mark.asyncio
async def test_gavel_evaluation_can_select_from_label_free_portfolio():
    predictions, scores, stats = await evaluate_prompt(
        prompt="answer",
        examples=[_numeric_example("gsm8k:test:0")],
        target_client=PortfolioTargetClient(),
        allow_code_execution=False,
        show_progress=False,
        workers=1,
        description="test",
        phase="test",
        self_refine_rounds=1,
        execution_mode="portfolio_select",
    )

    assert predictions[0]["prediction"] == "4"
    assert predictions[0]["execution_mode"] == "portfolio_select"
    assert scores[0].score == 1.0
    assert stats.api_calls == 6


@pytest.mark.asyncio
async def test_gavel_validation_gate_can_select_execution_mode(
    tmp_path,
    sample_compiler_output,
):
    config = GavelConfig(
        model="fake-target",
        optimizer_model="fake-optimizer",
        output_dir=tmp_path,
        cache=False,
        workers=2,
        show_progress=False,
        validation_gate=True,
        prompt_portfolio=False,
        self_refine_rounds=0,
        execution_modes=("direct", "plan"),
        pareto_candidates=0,
    )
    result = await run_gavel_baseline(
        train_examples=[_numeric_example("gsm8k:train:0")],
        val_examples=[_numeric_example("gsm8k:validation:0")],
        test_examples=[_numeric_example("gsm8k:test:0")],
        config=config,
        benchmark_spec=_spec(),
        optimizer_client=FakeOptimizerClient(sample_compiler_output.model_dump_json()),
        target_client=PlanTargetClient(),
    )

    gate = result.summary["validation_gate"]
    assert result.summary["selected_prompt"] == "base"
    assert result.summary["selected_execution_mode"] == "plan"
    assert result.summary["accepted"] is True
    assert result.summary["decision"] == "validation_selected_execution_mode"
    assert gate["base"]["score"] == 0.0
    assert gate["candidate"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0
