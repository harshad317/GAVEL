from __future__ import annotations

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
    def __init__(self, output: str):
        self.output = output
        self.calls = 0
        self.model = "fake-optimizer"

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        del messages, response_schema
        self.calls += 1
        return ClientResponse(
            output=self.output,
            raw=self.output,
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
        cache=False,
        workers=2,
        show_progress=False,
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
    assert result.summary["accepted"] is True
    assert result.summary["split_results"]["train"]["score"] == 1.0
    assert result.summary["split_results"]["val"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0
    assert result.predictions_path.exists()
    assert result.scores_path.exists()
    assert result.report_path.exists()
    assert result.prompt_path.exists()


def test_default_base_prompt_is_benchmark_specific():
    prompt = default_base_prompt(_spec())
    assert "benchmark-solving" in prompt
    assert "final answer" in prompt


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
