"""Live GAVEL benchmark runner over normalized benchmark examples."""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from pact_el.benchmarks.scoring import (
    ScoreAccumulator,
    score_prediction,
    summarize_scores,
    summarize_unscored_reasons,
)
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
from pact_el.renderer import estimate_token_count, render_prompt
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
    validation_gate: bool = True
    validate_rejected_candidates: bool = True
    prompt_portfolio: bool = True
    validation_margin: float = 0.0
    allow_code_execution: bool = False
    show_progress: bool = True
    base_prompt: Optional[str] = None

    def validate(self) -> None:
        _validate_temperature(self.temperature, "temperature")
        _validate_temperature(self.optimizer_temperature, "optimizer_temperature")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.budget < 1:
            raise ValueError("budget must be at least 1")
        if self.validation_margin < 0 or self.validation_margin > 1:
            raise ValueError("validation_margin must be between 0 and 1")


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


@dataclass
class PromptSelection:
    """Validation-gated prompt selected for final benchmark evaluation."""

    prompt: str
    selected_variant: str
    decision: str
    reason: str
    api_calls: int = 0
    base_summary: Optional[Dict[str, Any]] = None
    candidate_summary: Optional[Dict[str, Any]] = None
    candidate_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    margin: float = 0.0
    enabled: bool = True
    candidate_source: str = "accepted"

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selected_prompt": self.selected_variant,
            "decision": self.decision,
            "reason": self.reason,
            "api_calls": self.api_calls,
            "margin": self.margin,
            "candidate_source": self.candidate_source,
            "base": self.base_summary,
            "candidate": self.candidate_summary,
            "candidates": self.candidate_summaries,
        }


@dataclass(frozen=True)
class PromptCandidate:
    """A prompt variant competing in validation selection."""

    name: str
    prompt: str
    source: str
    report_decision: str
    report_accepted: bool = False


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
    prompt_candidates = _build_prompt_candidates(
        base_prompt=base_prompt,
        report=report,
        benchmark_spec=benchmark_spec,
        logs=logs,
        enabled=config.prompt_portfolio,
        validate_rejected_candidates=config.validate_rejected_candidates,
    )
    prompt_selection = await _select_prompt_with_validation(
        base_prompt=base_prompt,
        candidates=prompt_candidates,
        val_examples=val_examples,
        target_client=target_client,
        allow_code_execution=config.allow_code_execution,
        show_progress=config.show_progress,
        workers=config.workers,
        enabled=config.validation_gate,
        margin=config.validation_margin,
    )
    report.metadata = {
        **report.metadata,
        "validation_gate": prompt_selection.summary(),
    }
    optimized_prompt = prompt_selection.prompt

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
    optimization_api_calls = (
        len(train_examples)
        + report.call_ledger.total_calls
        + prompt_selection.api_calls
    )
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
        "temperature": config.temperature,
        "optimizer_temperature": config.optimizer_temperature,
        "benchmark": benchmark_spec.benchmark_id if benchmark_spec else None,
        "eval_examples": len(test_examples),
        "test_examples": len(test_examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "workers": config.workers,
        "max_in_flight": max_in_flight,
        "cache": config.cache,
        "budget": config.budget,
        "prompt_portfolio": config.prompt_portfolio,
        "accepted": prompt_selection.selected_variant != "base",
        "decision": prompt_selection.decision,
        "selected_prompt": prompt_selection.selected_variant,
        "validation_gate": prompt_selection.summary(),
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
                _set_progress_postfix(
                    progress,
                    tracker,
                    workers=workers,
                    in_flight=active,
                    max_in_flight=stats.max_in_flight,
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
        _set_progress_postfix(
            progress,
            tracker,
            workers=workers,
            in_flight=0,
            max_in_flight=0,
        )
        tasks = [
            asyncio.create_task(run_one(index, example))
            for index, example in enumerate(examples)
        ]
        for task in asyncio.as_completed(tasks):
            index, prediction, score = await task
            predictions[index] = prediction
            scores[index] = score
            tracker.add(score)
            _set_progress_postfix(
                progress,
                tracker,
                workers=workers,
                in_flight=active,
                max_in_flight=stats.max_in_flight,
            )
            progress.update(1)
    return (
        [prediction for prediction in predictions if prediction is not None],
        [score for score in scores if score is not None],
        stats,
    )


def _set_progress_postfix(
    progress: Any,
    tracker: ScoreAccumulator,
    *,
    workers: Optional[int],
    in_flight: Optional[int],
    max_in_flight: Optional[int],
) -> None:
    postfix = tracker.progress_postfix()
    if workers is not None:
        postfix["workers"] = workers
    if in_flight is not None:
        postfix["in_flight"] = in_flight
    if max_in_flight is not None:
        postfix["max_in_flight"] = max_in_flight
    progress.set_postfix(**postfix, refresh=False)


def default_base_prompt(spec: Optional[BenchmarkSpec] = None) -> str:
    return _render_structured_sections(_default_prompt_sections(spec))


def task_strategy_prompt(spec: Optional[BenchmarkSpec] = None) -> str:
    sections = _default_prompt_sections(spec)
    _extend_section(
        sections,
        "Task",
        [
            "Use a deterministic solve-and-verify loop: parse requirements, choose the simplest compliant structure, produce the answer, then check every requirement once more.",
            "When the task has multiple constraints, satisfy the most mechanically verifiable constraints first so the final answer is easy to audit.",
        ],
    )
    _extend_section(
        sections,
        "Quality Bar",
        [
            "Prefer a shorter, more controlled answer over a fluent answer that is hard to verify.",
            "Before finalizing, ensure the answer would still be correct if scored by a strict exact-match, format, unit-test, or rule-based evaluator.",
        ],
    )
    task_type = spec.task_type if spec else None
    if task_type == BenchmarkTaskType.MATH:
        _extend_section(
            sections,
            "Task",
            [
                "Privately compute through the problem step by step, then independently check the final arithmetic.",
                "If the answer is a number, normalize signs, fractions, decimals, and units according to the prompt before returning it.",
            ],
        )
    elif task_type == BenchmarkTaskType.INSTRUCTION_FOLLOWING:
        _extend_section(
            sections,
            "Task",
            [
                "For exact counts, preselect the number of words, sentences, lines, list items, or repetitions before writing.",
                "For ratios or balances, choose a simple symmetric structure and verify each side of the ratio explicitly.",
                "For word-property constraints, use conservative vocabulary and avoid words whose spelling, vowels, consonants, syllables, or first/last letters you cannot verify.",
                "For sentence-local constraints, build each sentence separately and check the required keyword, punctuation, alliteration, length, or ordering before moving on.",
            ],
        )
        _extend_section(
            sections,
            "Constraints",
            [
                "If satisfying all constraints requires sacrificing style or richness, sacrifice style and keep the constraints exact.",
            ],
        )
    elif task_type == BenchmarkTaskType.MULTIPLE_CHOICE:
        _extend_section(
            sections,
            "Task",
            [
                "Map each choice label to its answer text before deciding.",
                "Eliminate distractors using the question, then return the selected label exactly.",
            ],
        )
    elif task_type == BenchmarkTaskType.CODE_GENERATION:
        _extend_section(
            sections,
            "Task",
            [
                "Infer edge cases from the prompt and public tests.",
                "Return a complete implementation with required imports and helper functions included.",
            ],
        )
    elif task_type == BenchmarkTaskType.QUESTION_ANSWERING:
        _extend_section(
            sections,
            "Task",
            [
                "Locate the shortest answer span or normalized answer supported by the provided context.",
                "Avoid explanatory wrapping when the evaluator expects a concise final answer.",
            ],
        )
    elif task_type == BenchmarkTaskType.TRUTHFULNESS:
        _extend_section(
            sections,
            "Task",
            [
                "Check whether the question contains a false presupposition before answering directly.",
                "When uncertain, give a calibrated truthful answer instead of a popular but unsupported claim.",
            ],
        )
    metric_items = _metric_strategy_items(spec.metrics if spec else [])
    if metric_items:
        _extend_section(sections, "Task", metric_items)
    return _render_structured_sections(sections)


def evidence_strategy_prompt(
    spec: Optional[BenchmarkSpec],
    logs: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    summary = _summarize_evidence_patterns(logs)
    if not summary:
        return None
    sections = _default_prompt_sections(spec)
    _extend_section(
        sections,
        "Context",
        [
            "Training evidence revealed recurring prompt-fixable failure modes. Use these as transferable risk signals, not as memorized examples.",
            *summary,
        ],
    )
    _extend_section(
        sections,
        "Task",
        [
            "Before drafting, identify whether the current input resembles any recurring failure mode from the evidence summary.",
            "If it does, apply the corresponding stricter verification pattern before finalizing.",
        ],
    )
    _extend_section(
        sections,
        "Quality Bar",
        [
            "The answer is incomplete if it repeats a high-frequency training failure mode from the evidence summary.",
        ],
    )
    return _render_structured_sections(sections)


def _default_prompt_sections(
    spec: Optional[BenchmarkSpec] = None,
) -> List[tuple[str, List[str]]]:
    task_type = spec.task_type if spec else None
    goal = [
        "Solve each benchmark example correctly while satisfying every explicit instruction in the user prompt.",
    ]
    context = [
        "The user prompt is the authoritative task input. It may contain content requirements, formatting requirements, hidden evaluator constraints, or distractor wording.",
    ]
    if spec is not None:
        context.append(
            f"Benchmark: {spec.display_name} ({spec.benchmark_id}). Task type: {spec.task_type.value}. {spec.description}"
        )
    role = [
        "Act as a precise benchmark-solving assistant with strong constraint-following discipline.",
    ]
    input_section = [
        "Use the user's message as the raw input to solve. Do not assume missing data unless the prompt allows it.",
    ]
    task = [
        "Privately identify all explicit constraints before drafting.",
        "Draft the simplest answer that can satisfy the task.",
        "Privately verify the answer against every constraint before finalizing.",
    ]
    constraints = [
        "Follow the user's instruction exactly.",
        "Satisfy the requested output shape before adding any optional explanation.",
        "Do not add extra commentary unless the user explicitly asks for it.",
        "Treat output format, length, ordering, required content, and prohibited content as hard constraints.",
    ]
    output_format = [
        "Return only the final answer requested by the user.",
        "Do not show private reasoning, checklists, or verification steps.",
    ]
    quality_bar = [
        "The answer is complete only if it solves the task, satisfies every explicit constraint, and avoids unsupported extra content.",
    ]
    if task_type == BenchmarkTaskType.MATH:
        task.append("Solve the problem carefully and compute the final value before answering.")
        output_format.append("Put the final result on its own final line as `Answer: <answer>`.")
    elif task_type == BenchmarkTaskType.INSTRUCTION_FOLLOWING:
        constraints.append("Satisfy every explicit output constraint, including counts, casing, delimiters, required words, forbidden words, and ordering.")
        task.append("Privately convert the instruction into a checklist; treat word, sentence, line, ratio, repetition, and position constraints as hard requirements.")
        task.append("For count or ratio constraints, choose a simple structure that makes counting easy, then revise until the requested counts and proportions are exact.")
        task.append("For word-property constraints such as vowels, consonants, syllables, first/last letters, alphabetical order, or repeated spans, use only words you can verify against that property.")
        output_format.append("For formatting constraints such as quotes, indentation, options, title case, whitespace, newlines, bullets, or emoji, make the final answer match the requested surface form exactly.")
    elif task_type == BenchmarkTaskType.MULTIPLE_CHOICE:
        task.append("Select the single best answer choice.")
        output_format.append("Return only the answer choice letter unless the prompt explicitly requests explanation.")
    elif task_type == BenchmarkTaskType.CODE_GENERATION:
        task.append("Write complete executable Python code that satisfies the specification.")
        output_format.append("Return code only, with no markdown fences, no prose, and no copied tests unless the prompt asks for tests.")
    elif task_type == BenchmarkTaskType.QUESTION_ANSWERING:
        constraints.append("Use only the provided context when context is present.")
        quality_bar.append("Names, numbers, dates, and units are preserved exactly.")
    elif task_type == BenchmarkTaskType.TRUTHFULNESS:
        constraints.append("If a premise is false or unsupported, answer truthfully and avoid imitation of common falsehoods.")

    sections = [
        ("Goal", goal),
        ("Context", context),
        ("Role", role),
        ("Input", input_section),
        ("Task", task),
        ("Constraints", constraints),
        ("Output Format", output_format),
        ("Quality Bar", quality_bar),
    ]
    return sections


def _render_structured_sections(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    lines: List[str] = []
    for title, items in sections:
        lines.append(f"## {title}")
        for item in items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines).strip()


def _extend_section(
    sections: List[tuple[str, List[str]]],
    title: str,
    items: Sequence[str],
) -> None:
    for section_title, section_items in sections:
        if section_title == title:
            section_items.extend(items)
            return
    sections.append((title, list(items)))


def _summarize_evidence_patterns(logs: Sequence[Mapping[str, Any]]) -> List[str]:
    failure_count = 0
    area_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    for row in logs:
        if row.get("passed") is not False:
            continue
        failure_count += 1
        area = row.get("area")
        if area:
            area_counts[str(area)] += 1
        metadata = row.get("metadata") or {}
        metric = metadata.get("metric")
        if metric:
            metric_counts[str(metric)] += 1
        for instruction_id in metadata.get("failed_instruction_ids") or []:
            instruction = str(instruction_id)
            instruction_counts[instruction] += 1
            family_counts[instruction.split(":", 1)[0]] += 1

    if failure_count == 0:
        return []
    summary = [f"Training evidence failures: {failure_count}."]
    if area_counts:
        summary.append("Most affected contract areas: " + _format_counter(area_counts, limit=4) + ".")
    if metric_counts:
        summary.append("Most affected scorer metrics: " + _format_counter(metric_counts, limit=4) + ".")
    if family_counts:
        summary.append("Most affected instruction families: " + _format_counter(family_counts, limit=6) + ".")
        family_items = _instruction_family_strategy_items(family_counts)
        if family_items:
            summary.append("Transferable instruction checks: " + "; ".join(family_items[:6]) + ".")
    if instruction_counts:
        summary.append("Most frequent failed instruction ids: " + _format_counter(instruction_counts, limit=8) + ".")
    if metric_counts:
        metric_items = _metric_strategy_items(metric_counts.keys())
        if metric_items:
            summary.append("Transferable metric checks: " + "; ".join(item.rstrip(".") for item in metric_items[:5]) + ".")
    return summary


def _format_counter(counter: Counter[str], *, limit: int) -> str:
    return ", ".join(f"{key}={count}" for key, count in counter.most_common(limit))


def _metric_strategy_items(metrics: Sequence[Any]) -> List[str]:
    values = {
        metric.value if isinstance(metric, MetricKind) else str(metric)
        for metric in metrics
    }
    items: List[str] = []
    if MetricKind.OFFICIAL_EVALUATOR.value in values:
        items.append("For official evaluators, optimize for verifier-visible constraints: exact format, explicit inclusions/exclusions, counts, positions, and required transformations.")
    if MetricKind.NUMERIC_EXACT.value in values:
        items.append("For numeric exact-match scoring, make the final answer contain the normalized number the scorer should extract, with no competing numbers after it.")
    if MetricKind.MULTIPLE_CHOICE_ACCURACY.value in values:
        items.append("For multiple-choice scoring, return one unambiguous choice label and avoid mentioning alternate labels in the final answer.")
    if MetricKind.EXACT_MATCH.value in values:
        items.append("For exact-match scoring, produce the shortest normalized answer string that matches the expected entity, value, or phrase.")
    if MetricKind.TOKEN_F1.value in values:
        items.append("For token-F1 scoring, include all essential answer tokens while avoiding unsupported surrounding explanation.")
    if MetricKind.PASS_AT_1.value in values:
        items.append("For pass-at-1 code scoring, return executable code only, include needed imports, and handle edge cases implied by the tests.")
    return items


def _instruction_family_strategy_items(family_counts: Counter[str]) -> List[str]:
    strategy_by_family = {
        "words": "for word-property constraints, use words whose spelling, vowels, consonants, syllables, first letters, and last letters are easy to verify",
        "count": "for count constraints, decide the exact number of words, lines, list items, or occurrences before writing",
        "ratio": "for ratio constraints, use a simple repeated structure and verify each side of the ratio explicitly",
        "sentence": "for sentence constraints, construct and verify one sentence at a time before joining the answer",
        "format": "for format constraints, match requested delimiters, casing, whitespace, bullets, quotes, and line breaks exactly",
        "repeat": "for repetition constraints, count each required repeated word, phrase, or span after drafting",
        "custom": "for custom transformations, perform the requested transformation literally and avoid paraphrasing it away",
        "keywords": "for keyword constraints, include required keywords exactly and remove forbidden keywords completely",
        "language": "for language constraints, keep the entire final answer in the requested language",
        "startend": "for start/end constraints, verify the first and last visible tokens exactly",
        "punctuation": "for punctuation constraints, count and place the requested punctuation marks deliberately",
    }
    items: List[str] = []
    for family, _count in family_counts.most_common():
        strategy = strategy_by_family.get(str(family))
        if strategy is not None:
            items.append(strategy)
    return items


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


async def _select_prompt_with_validation(
    *,
    base_prompt: str,
    candidates: Sequence[PromptCandidate],
    val_examples: Sequence[BenchmarkExample],
    target_client: TargetClient,
    allow_code_execution: bool,
    show_progress: bool,
    workers: int,
    enabled: bool,
    margin: float,
) -> PromptSelection:
    unique_candidates = _dedupe_candidates(candidates, base_prompt=base_prompt)
    if not enabled:
        candidate = _preferred_candidate(unique_candidates)
        selected_variant = candidate.name if candidate is not None and candidate.report_accepted else "base"
        return PromptSelection(
            prompt=candidate.prompt if candidate is not None and selected_variant != "base" else base_prompt,
            selected_variant=selected_variant,
            decision=candidate.report_decision if candidate is not None else "base",
            reason="validation gate disabled",
            margin=margin,
            enabled=False,
            candidate_source=candidate.source if candidate is not None else "base",
        )
    if not unique_candidates:
        return PromptSelection(
            prompt=base_prompt,
            selected_variant="base",
            decision="base",
            reason="no validation candidates are available",
            margin=margin,
            candidate_source="base",
        )
    if not val_examples:
        candidate = _preferred_candidate(unique_candidates)
        return PromptSelection(
            prompt=candidate.prompt,
            selected_variant=candidate.name,
            decision=candidate.report_decision,
            reason="no validation examples available",
            margin=margin,
            candidate_source=candidate.source,
        )

    base_predictions, base_scores, base_stats = await evaluate_prompt(
        prompt=base_prompt,
        examples=val_examples,
        target_client=target_client,
        allow_code_execution=allow_code_execution,
        show_progress=show_progress,
        workers=workers,
        description="GAVEL validation gate/base",
        phase="validation_gate_base",
    )
    del base_predictions

    base_summary = _split_result_summary(
        base_scores,
        examples=len(val_examples),
        api_calls=len(val_examples),
        stats=base_stats,
    )
    candidate_summaries: Dict[str, Dict[str, Any]] = {}
    evaluated_candidates: List[tuple[PromptCandidate, Dict[str, Any]]] = []
    for candidate in unique_candidates:
        candidate_predictions, candidate_scores, candidate_stats = await evaluate_prompt(
            prompt=candidate.prompt,
            examples=val_examples,
            target_client=target_client,
            allow_code_execution=allow_code_execution,
            show_progress=show_progress,
            workers=workers,
            description=f"GAVEL validation gate/{candidate.name}",
            phase=f"validation_gate_{candidate.name}",
        )
        del candidate_predictions
        candidate_summary = _split_result_summary(
            candidate_scores,
            examples=len(val_examples),
            api_calls=len(val_examples),
            stats=candidate_stats,
        )
        candidate_summaries[candidate.name] = candidate_summary
        evaluated_candidates.append((candidate, candidate_summary))

    api_calls = len(val_examples) * (1 + len(unique_candidates))
    base_metric = _selection_metric(base_summary)
    scored_candidates = [
        (candidate, summary, metric)
        for candidate, summary in evaluated_candidates
        for metric in [_selection_metric(summary)]
        if metric is not None
    ]
    if base_metric is None or not scored_candidates:
        fallback = _preferred_candidate(unique_candidates)
        selected_variant = fallback.name if base_metric is None and fallback.report_accepted else "base"
        return PromptSelection(
            prompt=fallback.prompt if selected_variant != "base" else base_prompt,
            selected_variant=selected_variant,
            decision=fallback.report_decision if selected_variant != "base" else "validation_unscored_rollback",
            reason="validation scorer produced no comparable metric",
            api_calls=api_calls,
            base_summary=base_summary,
            candidate_summary=candidate_summaries.get(fallback.name) if fallback is not None else None,
            candidate_summaries=candidate_summaries,
            margin=margin,
            candidate_source=fallback.source if fallback is not None else "base",
        )

    best_candidate, best_summary, best_metric = max(
        scored_candidates,
        key=lambda item: (
            item[2],
            1 if item[0].report_accepted else 0,
            _candidate_priority(item[0]),
        ),
    )
    if best_metric + 1e-12 >= base_metric + margin:
        decision = (
            best_candidate.report_decision
            if best_candidate.report_accepted
            else "validation_override"
        )
        return PromptSelection(
            prompt=best_candidate.prompt,
            selected_variant=best_candidate.name,
            decision=decision,
            reason=(
                f"{best_candidate.name} validation metric {best_metric:.4f} met "
                f"base {base_metric:.4f} with margin {margin:.4f}"
            ),
            api_calls=api_calls,
            base_summary=base_summary,
            candidate_summary=best_summary,
            candidate_summaries=candidate_summaries,
            margin=margin,
            candidate_source=best_candidate.source,
        )
    return PromptSelection(
        prompt=base_prompt,
        selected_variant="base",
        decision="validation_rollback",
        reason=(
            f"best candidate validation metric {best_metric:.4f} fell below "
            f"base {base_metric:.4f} with margin {margin:.4f}"
        ),
        api_calls=api_calls,
        base_summary=base_summary,
        candidate_summary=best_summary,
        candidate_summaries=candidate_summaries,
        margin=margin,
        candidate_source=best_candidate.source,
    )


def _build_prompt_candidates(
    *,
    base_prompt: str,
    report: Any,
    benchmark_spec: Optional[BenchmarkSpec],
    logs: Sequence[Mapping[str, Any]],
    enabled: bool,
    validate_rejected_candidates: bool,
) -> List[PromptCandidate]:
    if not enabled:
        candidate_prompt, candidate_source = _selection_candidate_prompt(
            report,
            base_prompt=base_prompt,
            validate_rejected_candidates=validate_rejected_candidates,
        )
        if candidate_source == "none":
            return []
        return [
            PromptCandidate(
                name="optimized",
                prompt=candidate_prompt,
                source=candidate_source,
                report_decision=report.decision.value,
                report_accepted=report.accepted,
            )
        ]

    candidates = [
        PromptCandidate(
            name="task_strategy",
            prompt=task_strategy_prompt(benchmark_spec),
            source="deterministic_task_strategy",
            report_decision="validation_selected_task_strategy",
        )
    ]
    evidence_prompt = evidence_strategy_prompt(benchmark_spec, logs)
    if evidence_prompt is not None:
        candidates.append(
            PromptCandidate(
                name="evidence_strategy",
                prompt=evidence_prompt,
                source="deterministic_evidence_strategy",
                report_decision="validation_selected_evidence_strategy",
            )
        )
    candidate_prompt, candidate_source = _selection_candidate_prompt(
        report,
        base_prompt=base_prompt,
        validate_rejected_candidates=validate_rejected_candidates,
    )
    if candidate_source != "none":
        candidates.append(
            PromptCandidate(
                name="optimized",
                prompt=candidate_prompt,
                source=candidate_source,
                report_decision=report.decision.value,
                report_accepted=report.accepted,
            )
        )
    return _dedupe_candidates(candidates, base_prompt=base_prompt)


def _selection_candidate_prompt(
    report: Any,
    *,
    base_prompt: str,
    validate_rejected_candidates: bool,
) -> tuple[str, str]:
    if report.accepted:
        return report.rendered_prompt, "accepted"
    if not validate_rejected_candidates or report.graph is None or report.patch is None:
        return base_prompt, "none"
    return render_prompt(report.graph, source_prompt=base_prompt, patch=report.patch), "rejected"


def _dedupe_candidates(
    candidates: Sequence[PromptCandidate],
    *,
    base_prompt: str,
) -> List[PromptCandidate]:
    seen = {base_prompt.strip()}
    unique: List[PromptCandidate] = []
    for candidate in candidates:
        normalized = candidate.prompt.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def _preferred_candidate(candidates: Sequence[PromptCandidate]) -> Optional[PromptCandidate]:
    accepted = [candidate for candidate in candidates if candidate.report_accepted]
    if accepted:
        return max(accepted, key=_candidate_priority)
    return max(candidates, key=_candidate_priority) if candidates else None


def _candidate_priority(candidate: PromptCandidate) -> int:
    priorities = {
        "optimized": 3,
        "evidence_strategy": 2,
        "task_strategy": 1,
    }
    return priorities.get(candidate.name, 0)


def _selection_metric(summary: Mapping[str, Any]) -> Optional[float]:
    score = summary.get("score")
    if score is not None:
        return float(score)
    accuracy = summary.get("accuracy")
    if accuracy is not None:
        return float(accuracy)
    return None


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
        passed = None if score is None or score.passed is None else bool(score.passed)
        metadata = _evidence_metadata(example, score)
        logs.append(
            {
                "evidence_id": f"train_{example.example_id}",
                "input": example.prompt,
                "expected": _expected_behavior(example, score),
                "observed": _observed_behavior(prediction, score),
                "output": prediction,
                "passed": passed,
                "severity": _evidence_severity(score),
                "area": _contract_area(example, score).value,
                "confidence": _evidence_confidence(score),
                "prompt_fixability": _prompt_fixability(example, score),
                "metadata": metadata,
            }
        )
    return logs


def _expected_behavior(example: BenchmarkExample, score: Optional[ScoreResult]) -> str:
    if example.metric == MetricKind.OFFICIAL_EVALUATOR:
        parts = ["Satisfy every official-evaluator instruction constraint for this prompt."]
        constraints = _instruction_constraints(example)
        failed_constraints = _failed_instruction_ids(score)
        if constraints:
            parts.append("Instruction constraints: " + "; ".join(constraints))
        if failed_constraints:
            parts.append("Observed output failed these constraints: " + ", ".join(failed_constraints))
        return " ".join(parts)
    if example.metric == MetricKind.PASS_AT_1:
        tests = _public_tests(example)
        if tests:
            return (
                "Generate complete Python code that satisfies the specification and "
                "passes the public unit tests: " + " | ".join(tests)
            )
        return "Generate complete Python code that passes the official unit tests."
    if example.choices:
        choices = "; ".join(_choice_lines(example))
        return (
            f"Return exactly one answer choice label. Expected: {example.expected_answer}. "
            f"Choices: {choices}"
        )
    aliases = _answer_aliases(example)
    if aliases:
        return "Expected answer aliases: " + "; ".join(aliases)
    return f"Expected answer: {example.expected_answer}"


def _observed_behavior(prediction: Any, score: Optional[ScoreResult]) -> str:
    output = _truncate_text("" if prediction is None else str(prediction), max_chars=1600)
    if score is None:
        return f"Observed output: {output}"
    if score.passed is True:
        return f"Observed output passed scorer: {output}"
    diagnostics = _score_failure_summary(score)
    if diagnostics:
        return f"Observed output: {output}\nScorer diagnostics: {diagnostics}"
    return f"Observed output: {output}\nScorer marked the output as failing."


def _evidence_metadata(
    example: BenchmarkExample,
    score: Optional[ScoreResult],
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "benchmark_id": example.benchmark_id,
        "example_id": example.example_id,
        "metric": example.metric.value,
        "score": None if score is None else score.score,
        "passed": None if score is None else score.passed,
        "diagnostics": _score_diagnostics(score),
    }
    constraints = _instruction_constraints(example)
    if constraints:
        metadata["instruction_constraints"] = constraints
    failed_constraints = _failed_instruction_ids(score)
    if failed_constraints:
        metadata["failed_instruction_ids"] = failed_constraints
    if example.choices:
        metadata["choices"] = _choice_lines(example)
    if example.metric != MetricKind.PASS_AT_1:
        aliases = _answer_aliases(example)
        if aliases:
            metadata["answer_aliases"] = aliases
    tests = _public_tests(example)
    if tests:
        metadata["public_tests"] = tests
    return metadata


def _score_diagnostics(score: Optional[ScoreResult]) -> Dict[str, Any]:
    if score is None:
        return {"reason": "score missing"}
    diagnostics: Dict[str, Any] = {
        "score": score.score,
        "passed": score.passed,
    }
    summary = _score_failure_summary(score)
    if summary:
        diagnostics["summary"] = summary
    details = _compact_score_details(score.details)
    if details:
        diagnostics["details"] = details
    return diagnostics


def _score_failure_summary(score: Optional[ScoreResult]) -> str:
    if score is None:
        return ""
    details = score.details or {}
    parts: List[str] = []
    if details.get("reason"):
        parts.append(str(details["reason"]))
    if "observed_number" in details or "expected_number" in details:
        parts.append(
            f"observed_number={details.get('observed_number')!r}, "
            f"expected_number={details.get('expected_number')!r}"
        )
    if "observed_choice" in details or "expected_choice" in details:
        parts.append(
            f"observed_choice={details.get('observed_choice')!r}, "
            f"expected_choice={details.get('expected_choice')!r}"
        )
    if "best_f1" in details:
        parts.append(f"best_f1={details.get('best_f1')!r}")
    failed_constraints = _failed_instruction_ids(score)
    if failed_constraints:
        parts.append("failed_instruction_ids=" + ", ".join(failed_constraints))
    if details.get("stderr"):
        parts.append("stderr=" + _truncate_text(str(details["stderr"]), max_chars=600))
    if details.get("stdout"):
        parts.append("stdout=" + _truncate_text(str(details["stdout"]), max_chars=300))
    if score.score is not None and not parts:
        parts.append(f"score={score.score}")
    return "; ".join(parts)


def _compact_score_details(details: Mapping[str, Any]) -> Dict[str, Any]:
    keep = (
        "reason",
        "observed_number",
        "expected_number",
        "observed_choice",
        "expected_choice",
        "aliases",
        "best_f1",
        "instruction_id_list",
        "follow_instruction_list",
        "returncode",
        "stderr",
        "stdout",
        "evaluator",
    )
    compact: Dict[str, Any] = {}
    for key in keep:
        if key in details:
            compact[key] = _compact_value(details[key])
    return compact


def _compact_value(value: Any, *, max_chars: int = 800) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, max_chars=max_chars)
    if isinstance(value, list):
        return [_compact_value(item, max_chars=max_chars) for item in value[:20]]
    if isinstance(value, tuple):
        return [_compact_value(item, max_chars=max_chars) for item in value[:20]]
    if isinstance(value, Mapping):
        return {
            str(key): _compact_value(item, max_chars=max_chars)
            for key, item in list(value.items())[:20]
        }
    return value


def _instruction_constraints(example: BenchmarkExample) -> List[str]:
    ids = list(example.metadata.get("instruction_id_list") or [])
    kwargs_list = list(example.metadata.get("kwargs") or [])
    official_row = example.metadata.get("official_input_row") or {}
    if not ids:
        ids = list(official_row.get("instruction_id_list") or [])
    if not kwargs_list:
        kwargs_list = list(official_row.get("kwargs") or [])
    constraints: List[str] = []
    for index, instruction_id in enumerate(ids[:12]):
        raw_kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        kwargs = {
            str(key): value
            for key, value in dict(raw_kwargs or {}).items()
            if value is not None
        }
        if kwargs:
            rendered_kwargs = ", ".join(
                f"{key}={_truncate_text(str(value), max_chars=120)!r}"
                for key, value in sorted(kwargs.items())
            )
            constraints.append(f"{instruction_id}({rendered_kwargs})")
        else:
            constraints.append(str(instruction_id))
    return constraints


def _failed_instruction_ids(score: Optional[ScoreResult]) -> List[str]:
    if score is None:
        return []
    details = score.details or {}
    ids = list(details.get("instruction_id_list") or [])
    followed = list(details.get("follow_instruction_list") or [])
    failed: List[str] = []
    for instruction_id, did_follow in zip(ids, followed):
        if did_follow is False:
            failed.append(str(instruction_id))
    return failed


def _choice_lines(example: BenchmarkExample) -> List[str]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [
        f"{labels[index]}: {_truncate_text(str(choice), max_chars=240)}"
        for index, choice in enumerate(example.choices)
    ]


def _answer_aliases(example: BenchmarkExample) -> List[str]:
    aliases = example.metadata.get("answer_aliases") or example.metadata.get("correct_answers")
    if aliases:
        return [_truncate_text(str(alias), max_chars=240) for alias in aliases[:20]]
    if example.expected_answer is not None:
        return [_truncate_text(str(example.expected_answer), max_chars=240)]
    return []


def _public_tests(example: BenchmarkExample) -> List[str]:
    tests = list(example.metadata.get("test_imports", [])) + list(
        example.metadata.get("test_list", [])
    )
    return [_truncate_text(str(test), max_chars=300) for test in tests[:8]]


def _evidence_severity(score: Optional[ScoreResult]) -> str:
    if score is None or score.score is None:
        return "medium"
    if score.passed is True:
        return "low"
    if float(score.score) > 0:
        return "medium"
    return "high"


def _evidence_confidence(score: Optional[ScoreResult]) -> float:
    if score is None or score.score is None:
        return 0.45
    return 0.9


def _prompt_fixability(
    example: BenchmarkExample,
    score: Optional[ScoreResult],
) -> float:
    if score is None or score.passed is None:
        return 0.3
    if score.passed is True:
        return 0.2
    if example.metric == MetricKind.OFFICIAL_EVALUATOR:
        return 0.9
    if example.metric == MetricKind.MULTIPLE_CHOICE_ACCURACY:
        return 0.75
    if example.metric == MetricKind.PASS_AT_1:
        return 0.7
    if example.metric in {MetricKind.EXACT_MATCH, MetricKind.TOKEN_F1}:
        return 0.65
    if example.metric == MetricKind.NUMERIC_EXACT:
        return 0.7
    return 0.6


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def _contract_area(
    example: BenchmarkExample,
    score: Optional[ScoreResult] = None,
) -> ContractArea:
    if example.metric == MetricKind.OFFICIAL_EVALUATOR:
        failed_families = {
            instruction_id.split(":", 1)[0]
            for instruction_id in _failed_instruction_ids(score)
        }
        all_families = {
            constraint.split(":", 1)[0]
            for constraint in _instruction_constraints(example)
            if ":" in constraint
        }
        families = failed_families or all_families
        if families and families <= {"format"}:
            return ContractArea.FORMATTING_RULE
        if families & {"count", "ratio", "words", "sentence", "repeat", "custom"}:
            return ContractArea.PRECEDENCE_RULE
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
        f"Task type: {spec.task_type.value}. {spec.description} "
        "The optimized prompt must generalize to held-out examples and must not memorize train labels."
    )


def _rubric(spec: Optional[BenchmarkSpec]) -> str:
    if spec is None:
        return "Score each answer with the normalized benchmark scorer."
    metrics = ", ".join(metric.value for metric in spec.metrics)
    return (
        f"Metrics: {metrics}. Official evaluator: {spec.evaluator}. "
        "Prefer rules that improve scorer-visible correctness while preserving the prompt's requested output format."
    )


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
        "unscored_reasons": summarize_unscored_reasons(scores),
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


def _validate_temperature(value: float, name: str) -> None:
    if value < 0 or value > 2:
        raise ValueError(f"{name} must be between 0 and 2")


def _run_name(config: GavelConfig, spec: Optional[BenchmarkSpec]) -> str:
    benchmark = spec.benchmark_id if spec else "benchmark"
    model = config.model.replace("/", "_").replace(":", "_")
    optimizer_model = _optimizer_model(config).replace("/", "_").replace(":", "_")
    if optimizer_model == model:
        return f"gavel-{benchmark}-{model}"
    return f"gavel-{benchmark}-{model}-opt_{optimizer_model}"
