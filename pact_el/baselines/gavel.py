"""Live GAVEL benchmark runner over normalized benchmark examples."""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from pact_el.benchmarks.scoring import ScoreAccumulator, score_prediction, summarize_scores
from pact_el.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSpec,
    BenchmarkTaskType,
    MetricKind,
    ScoreResult,
)
from pact_el.clients import (
    ClientResponse,
    LiteLLMOptimizerClient,
    LiteLLMTargetClient,
    OptimizerClient,
    TargetClient,
)
from pact_el.optimize import pact_optimize
from pact_el.renderer import estimate_token_count
from pact_el.schemas import CallRecord, CallRole, ContractArea


@dataclass
class GavelConfig:
    """Configuration for one live GAVEL benchmark run."""

    model: str = "openai/gpt-4o-mini"
    optimizer_model: Optional[str] = None
    output_dir: Path = Path("output/baselines/gavel")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    optimizer_temperature: float = 0.0
    max_tokens: Optional[int] = None
    optimizer_max_tokens: Optional[int] = None
    target_kwargs: Dict[str, Any] = field(default_factory=dict)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    cache: bool = True
    workers: int = 1
    budget: int = 9
    allow_one_repair: bool = True
    allow_code_execution: bool = False
    show_progress: bool = True
    base_prompt: Optional[str] = None

    def validate(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.budget < 1:
            raise ValueError("budget must be at least 1")


@dataclass
class GavelRunResult:
    """File paths and aggregate metrics from a GAVEL run."""

    summary: Dict[str, Any]
    predictions_path: Path
    scores_path: Path
    report_path: Path
    prompt_path: Path


@dataclass
class EvaluationStats:
    """Concurrency stats captured during prompt evaluation."""

    requested_workers: int
    max_in_flight: int = 0


async def run_gavel_baseline(
    *,
    train_examples: Sequence[BenchmarkExample],
    val_examples: Sequence[BenchmarkExample],
    test_examples: Sequence[BenchmarkExample],
    config: GavelConfig,
    benchmark_spec: Optional[BenchmarkSpec] = None,
    selection_summary: Optional[Mapping[str, Any]] = None,
    optimizer_client: Optional[OptimizerClient] = None,
    target_client: Optional[TargetClient] = None,
) -> GavelRunResult:
    """Compile a GAVEL prompt from train evidence, then evaluate all splits."""

    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = _run_name(config, benchmark_spec)
    base_prompt = config.base_prompt or default_base_prompt(benchmark_spec)
    task_spec = _task_spec(benchmark_spec)
    rubric = _rubric(benchmark_spec)

    optimizer_client = optimizer_client or _build_optimizer_client(config)
    target_client = target_client or _build_target_client(config)
    if config.cache:
        cache_dir = config.output_dir / ".cache"
        optimizer_client = CachedOptimizerClient(
            optimizer_client,
            DiskJsonCache(cache_dir / "optimizer.json"),
        )
        target_client = CachedTargetClient(
            target_client,
            DiskJsonCache(cache_dir / "target.json"),
        )

    evidence_predictions, evidence_scores, evidence_stats = await evaluate_prompt(
        prompt=base_prompt,
        examples=train_examples,
        target_client=target_client,
        allow_code_execution=config.allow_code_execution,
        show_progress=config.show_progress and bool(train_examples),
        workers=config.workers,
        description="GAVEL evidence",
        phase="evidence",
    )
    logs = _evidence_logs(train_examples, evidence_predictions, evidence_scores)

    report = await pact_optimize(
        prompt=base_prompt,
        task_spec=task_spec,
        rubric=rubric,
        logs=logs,
        optimizer_client=optimizer_client,
        target_client=target_client,
        budget=config.budget,
        allow_one_repair=config.allow_one_repair,
        canary_concurrency=config.workers,
    )
    optimized_prompt = report.rendered_prompt

    split_evaluations = await _evaluate_report_splits(
        prompt=optimized_prompt,
        run_name=run_name,
        output_dir=config.output_dir,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        target_client=target_client,
        allow_code_execution=config.allow_code_execution,
        show_progress=config.show_progress,
        workers=config.workers,
    )
    test_eval = split_evaluations["test"]
    scores = test_eval["scores"]
    predictions_path = Path(test_eval["predictions_path"])
    scores_path = Path(test_eval["scores_path"])
    report_path = config.output_dir / f"{run_name}.report.json"
    report_path.write_text(report.model_dump_json(indent=2))
    prompt_path = config.output_dir / f"{run_name}.prompt.txt"
    prompt_path.write_text(optimized_prompt)

    split_results = {
        split_name: split_eval["summary"]
        for split_name, split_eval in split_evaluations.items()
    }
    optimization_api_calls = len(train_examples) + report.call_ledger.total_calls
    split_results["optimization"] = {
        "score": None,
        "stddev": None,
        "api_calls": optimization_api_calls,
        "examples": len(train_examples),
        "scored": len([score for score in evidence_scores if score.score is not None]),
        "workers": evidence_stats.requested_workers,
        "max_in_flight": evidence_stats.max_in_flight,
    }
    max_in_flight = max(
        [evidence_stats.max_in_flight]
        + [
            int(split_eval["summary"].get("max_in_flight") or 0)
            for split_eval in split_evaluations.values()
        ]
    )
    summary = {
        **summarize_scores(scores),
        "method": "gavel",
        "optimizer": "gavel",
        "model": config.model,
        "optimizer_model": _optimizer_model(config),
        "benchmark": benchmark_spec.benchmark_id if benchmark_spec else None,
        "eval_examples": len(test_examples),
        "test_examples": len(test_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "workers": config.workers,
        "max_in_flight": max_in_flight,
        "cache": config.cache,
        "budget": config.budget,
        "accepted": report.accepted,
        "decision": report.decision.value,
        "optimization_api_calls": optimization_api_calls,
        "predictions_path": str(predictions_path),
        "scores_path": str(scores_path),
        "report_path": str(report_path),
        "prompt_path": str(prompt_path),
        "split_results": split_results,
        "split_paths": {
            split_name: {
                "predictions_path": split_eval["predictions_path"],
                "scores_path": split_eval["scores_path"],
            }
            for split_name, split_eval in split_evaluations.items()
        },
    }
    if selection_summary is not None:
        summary["selection"] = dict(selection_summary)
    summary_path = config.output_dir / f"{run_name}.summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return GavelRunResult(
        summary=summary,
        predictions_path=predictions_path,
        scores_path=scores_path,
        report_path=report_path,
        prompt_path=prompt_path,
    )


async def evaluate_prompt(
    *,
    prompt: str,
    examples: Sequence[BenchmarkExample],
    target_client: TargetClient,
    allow_code_execution: bool,
    show_progress: bool,
    workers: int,
    description: str,
    phase: str,
) -> tuple[List[Dict[str, Any]], List[ScoreResult], EvaluationStats]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    predictions: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    scores: List[Optional[ScoreResult]] = [None] * len(examples)
    progress = tqdm(
        total=len(examples),
        desc=description,
        unit="ex",
        colour="cyan",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    tracker = ScoreAccumulator()
    stats = EvaluationStats(requested_workers=workers)
    active = 0
    active_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(workers)

    async def run_one(index: int, example: BenchmarkExample) -> Tuple[int, Dict[str, Any], ScoreResult]:
        async with semaphore:
            nonlocal active
            async with active_lock:
                active += 1
                stats.max_in_flight = max(stats.max_in_flight, active)
                progress.set_postfix(
                    workers=workers,
                    in_flight=active,
                    max_in_flight=stats.max_in_flight,
                    refresh=False,
                )
            try:
                response = await target_client.complete(
                    prompt,
                    example.prompt,
                    metadata={
                        "phase": phase,
                        "example_id": example.example_id,
                        "benchmark_id": example.benchmark_id,
                    },
                )
            finally:
                async with active_lock:
                    active -= 1
            prediction = "" if response.output is None else str(response.output)
            score = score_prediction(
                example,
                prediction,
                allow_code_execution=allow_code_execution,
            )
            return (
                index,
                {
                    "example_id": example.example_id,
                    "benchmark_id": example.benchmark_id,
                    "prediction": prediction,
                    "raw_prediction": response.output,
                },
                score,
            )

    with progress:
        progress.set_postfix(
            workers=workers,
            in_flight=0,
            max_in_flight=0,
            refresh=False,
        )
        tasks = [
            asyncio.create_task(run_one(index, example))
            for index, example in enumerate(examples)
        ]
        for task in asyncio.as_completed(tasks):
            index, prediction, score = await task
            predictions[index] = prediction
            scores[index] = score
            if tracker.add(score):
                progress.set_postfix(
                    **tracker.progress_postfix(),
                    workers=workers,
                    in_flight=active,
                    max_in_flight=stats.max_in_flight,
                    refresh=False,
                )
            progress.update(1)
    return (
        [prediction for prediction in predictions if prediction is not None],
        [score for score in scores if score is not None],
        stats,
    )


def default_base_prompt(spec: Optional[BenchmarkSpec] = None) -> str:
    task_type = spec.task_type if spec else None
    lines = [
        "You are a precise benchmark-solving assistant.",
        "Follow the user's instruction exactly.",
        "Do not add extra commentary unless the user explicitly asks for it.",
    ]
    if task_type == BenchmarkTaskType.MATH:
        lines.append("Solve the problem carefully and put the final answer at the end.")
    elif task_type == BenchmarkTaskType.INSTRUCTION_FOLLOWING:
        lines.append("Satisfy every explicit output constraint in the instruction.")
    elif task_type == BenchmarkTaskType.MULTIPLE_CHOICE:
        lines.append("Return the single best answer choice letter.")
    elif task_type == BenchmarkTaskType.CODE_GENERATION:
        lines.append("Return only executable code with no markdown fences.")
    elif task_type == BenchmarkTaskType.QUESTION_ANSWERING:
        lines.append("Answer using only the provided context when context is present.")
    elif task_type == BenchmarkTaskType.TRUTHFULNESS:
        lines.append("If a premise is false or unsupported, answer truthfully and avoid imitation of common falsehoods.")
    return "\n".join(lines)


class DiskJsonCache:
    """Small JSON cache used to make repeated GAVEL runs resumable."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}
        else:
            self._data = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True, default=str))
            tmp.replace(self.path)


class CachedTargetClient:
    """Target client wrapper with deterministic prompt/input cache keys."""

    def __init__(self, inner: TargetClient, cache: DiskJsonCache):
        self.inner = inner
        self.cache = cache
        self.model = getattr(inner, "model", "")

    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        key = _cache_key(
            {
                "kind": "target",
                "model": self.model,
                "prompt": prompt,
                "input": input,
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            output = cached.get("output")
            return ClientResponse(
                output=output,
                raw=output,
                call_record=_cached_call_record(
                    role=CallRole.TARGET,
                    name="target_cached",
                    model=self.model,
                    prompt_payload={"prompt": prompt, "input": input},
                    output=output,
                    metadata=metadata,
                ),
            )
        response = await self.inner.complete(prompt, input, metadata=metadata)
        self.cache.set(key, {"output": response.output})
        return response


class CachedOptimizerClient:
    """Optimizer client wrapper with deterministic message/schema cache keys."""

    def __init__(self, inner: OptimizerClient, cache: DiskJsonCache):
        self.inner = inner
        self.cache = cache
        self.model = getattr(inner, "model", "")

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        key = _cache_key(
            {
                "kind": "optimizer",
                "model": self.model,
                "messages": list(messages),
                "response_schema": response_schema,
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            output = cached.get("output")
            return ClientResponse(
                output=output,
                raw=output,
                call_record=_cached_call_record(
                    role=CallRole.OPTIMIZER,
                    name="optimizer_cached",
                    model=self.model,
                    prompt_payload={"messages": list(messages)},
                    output=output,
                    metadata=metadata,
                ),
            )
        response = await self.inner.complete(
            messages,
            response_schema=response_schema,
            metadata=metadata,
        )
        self.cache.set(key, {"output": response.output})
        return response


async def _evaluate_report_splits(
    *,
    prompt: str,
    run_name: str,
    output_dir: Path,
    train_examples: Sequence[BenchmarkExample],
    val_examples: Sequence[BenchmarkExample],
    test_examples: Sequence[BenchmarkExample],
    target_client: TargetClient,
    allow_code_execution: bool,
    show_progress: bool,
    workers: int,
) -> Dict[str, Dict[str, Any]]:
    split_examples = {
        "train": list(train_examples),
        "val": list(val_examples),
        "test": list(test_examples),
    }
    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name, examples in split_examples.items():
        predictions, scores, stats = await evaluate_prompt(
            prompt=prompt,
            examples=examples,
            target_client=target_client,
            allow_code_execution=allow_code_execution,
            show_progress=show_progress and bool(examples),
            workers=workers,
            description=f"GAVEL {split_name}",
            phase=f"final_{split_name}",
        )
        predictions_path, scores_path = _write_split_outputs(
            output_dir,
            run_name,
            split_name,
            predictions,
            scores,
        )
        evaluations[split_name] = {
            "predictions": predictions,
            "scores": scores,
            "predictions_path": str(predictions_path),
            "scores_path": str(scores_path),
            "summary": _split_result_summary(
                scores,
                examples=len(examples),
                api_calls=len(examples),
                stats=stats,
            ),
        }
    return evaluations


def _build_optimizer_client(config: GavelConfig) -> OptimizerClient:
    kwargs = dict(config.optimizer_kwargs)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.optimizer_max_tokens is not None:
        kwargs["max_tokens"] = config.optimizer_max_tokens
    return LiteLLMOptimizerClient(
        model=_optimizer_model(config),
        temperature=config.optimizer_temperature,
        extra_kwargs=kwargs,
    )


def _build_target_client(config: GavelConfig) -> TargetClient:
    kwargs = dict(config.target_kwargs)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return LiteLLMTargetClient(
        model=config.model,
        temperature=config.temperature,
        extra_kwargs=kwargs,
    )


def _evidence_logs(
    examples: Sequence[BenchmarkExample],
    predictions: Sequence[Mapping[str, Any]],
    scores: Sequence[ScoreResult],
) -> List[Dict[str, Any]]:
    prediction_by_id = {row["example_id"]: row.get("prediction") for row in predictions}
    score_by_id = {score.example_id: score for score in scores}
    logs: List[Dict[str, Any]] = []
    for example in examples:
        score = score_by_id.get(example.example_id)
        prediction = prediction_by_id.get(example.example_id)
        passed = bool(score and score.passed is True)
        logs.append(
            {
                "evidence_id": f"train_{example.example_id}",
                "input": example.prompt,
                "expected": _expected_behavior(example),
                "observed": prediction,
                "output": prediction,
                "passed": passed,
                "severity": "low" if passed else "high",
                "area": _contract_area(example).value,
                "confidence": 0.8 if score and score.score is not None else 0.5,
                "prompt_fixability": 0.4 if passed else 0.8,
                "metadata": {
                    "benchmark_id": example.benchmark_id,
                    "example_id": example.example_id,
                    "metric": example.metric.value,
                    "score": None if score is None else score.model_dump(mode="json"),
                },
            }
        )
    return logs


def _expected_behavior(example: BenchmarkExample) -> str:
    if example.metric == MetricKind.OFFICIAL_EVALUATOR:
        return "Satisfy the benchmark's official evaluator for this instruction."
    if example.metric == MetricKind.PASS_AT_1:
        return "Generate code that passes the official unit tests."
    if example.choices:
        return f"Return the correct answer choice. Expected: {example.expected_answer}"
    return f"Expected answer: {example.expected_answer}"


def _contract_area(example: BenchmarkExample) -> ContractArea:
    if example.metric == MetricKind.OFFICIAL_EVALUATOR:
        return ContractArea.FORMATTING_RULE
    if example.metric == MetricKind.PASS_AT_1:
        return ContractArea.TASK_INTENT
    if example.metric == MetricKind.MULTIPLE_CHOICE_ACCURACY:
        return ContractArea.DECISION_RULE
    return ContractArea.TASK_INTENT


def _task_spec(spec: Optional[BenchmarkSpec]) -> str:
    if spec is None:
        return "Solve each benchmark example according to its prompt."
    return (
        f"Benchmark: {spec.display_name} ({spec.benchmark_id}). "
        f"Task type: {spec.task_type.value}. {spec.description}"
    )


def _rubric(spec: Optional[BenchmarkSpec]) -> str:
    if spec is None:
        return "Score each answer with the normalized benchmark scorer."
    metrics = ", ".join(metric.value for metric in spec.metrics)
    return f"Metrics: {metrics}. Official evaluator: {spec.evaluator}"


def _write_split_outputs(
    output_dir: Path,
    run_name: str,
    split_name: str,
    predictions: Sequence[Mapping[str, Any]],
    scores: Sequence[ScoreResult],
) -> tuple[Path, Path]:
    stem = run_name if split_name == "test" else f"{run_name}.{split_name}"
    predictions_path = output_dir / f"{stem}.predictions.jsonl"
    scores_path = output_dir / f"{stem}.scores.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in predictions)
        + ("\n" if predictions else "")
    )
    scores_path.write_text(
        "\n".join(score.model_dump_json() for score in scores) + ("\n" if scores else "")
    )
    return predictions_path, scores_path


def _split_result_summary(
    scores: Sequence[ScoreResult],
    *,
    examples: int,
    api_calls: Optional[int],
    stats: Optional[EvaluationStats] = None,
) -> Dict[str, Any]:
    score_values = [float(score.score) for score in scores if score.score is not None]
    summary = summarize_scores(scores)
    return {
        "score": summary["mean_score"],
        "accuracy": summary["accuracy"],
        "stddev": _sample_stddev(score_values),
        "api_calls": api_calls,
        "examples": examples,
        "scored": summary["scored"],
        "passed": summary["passed"],
        "unscored": summary["unscored"],
        "workers": None if stats is None else stats.requested_workers,
        "max_in_flight": None if stats is None else stats.max_in_flight,
    }


def _sample_stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def _cache_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_call_record(
    *,
    role: CallRole,
    name: str,
    model: str,
    prompt_payload: Any,
    output: Any,
    metadata: Optional[Mapping[str, Any]],
) -> CallRecord:
    return CallRecord(
        role=role,
        name=name,
        model=model,
        prompt_tokens=estimate_token_count(json.dumps(prompt_payload, sort_keys=True, default=str)),
        completion_tokens=estimate_token_count("" if output is None else str(output)),
        cached=True,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        metadata=dict(metadata or {}),
    )


def _optimizer_model(config: GavelConfig) -> str:
    return config.optimizer_model or config.model


def _run_name(config: GavelConfig, spec: Optional[BenchmarkSpec]) -> str:
    benchmark = spec.benchmark_id if spec else "benchmark"
    model = config.model.replace("/", "_").replace(":", "_")
    optimizer_model = _optimizer_model(config).replace("/", "_").replace(":", "_")
    if optimizer_model == model:
        return f"gavel-{benchmark}-{model}"
    return f"gavel-{benchmark}-{model}-opt_{optimizer_model}"
