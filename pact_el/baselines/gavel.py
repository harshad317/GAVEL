"""Live GAVEL benchmark runner over normalized benchmark examples."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import statistics
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import Field, ValidationError
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
    _openai_strict_response_schema,
)
from pact_el.compiler import strict_parse_json_model
from pact_el.optimize import pact_optimize
from pact_el.renderer import estimate_token_count, render_prompt
from pact_el.schemas import CallRecord, CallRole, ContractArea, StrictModel


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
    validation_confidence_z: float = 0.8
    rejected_candidate_margin: float = 0.08
    prompt_complexity_margin: float = 0.02
    self_refine_rounds: int = 1
    execution_modes: Tuple[str, ...] = ("direct", "plan", "plan_refine", "self_refine")
    pareto_candidates: int = 6
    pareto_frontier_size: int = 6
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
        if self.validation_confidence_z < 0:
            raise ValueError("validation_confidence_z must be non-negative")
        if self.rejected_candidate_margin < 0 or self.rejected_candidate_margin > 1:
            raise ValueError("rejected_candidate_margin must be between 0 and 1")
        if self.prompt_complexity_margin < 0 or self.prompt_complexity_margin > 1:
            raise ValueError("prompt_complexity_margin must be between 0 and 1")
        if self.self_refine_rounds < 0:
            raise ValueError("self_refine_rounds must be non-negative")
        if self.pareto_candidates < 0:
            raise ValueError("pareto_candidates must be non-negative")
        if self.pareto_frontier_size < 0:
            raise ValueError("pareto_frontier_size must be non-negative")
        _resolve_execution_modes(self.execution_modes, self.self_refine_rounds)


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
    api_calls: int = 0


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
    effective_margin: float = 0.0
    execution_mode: str = "direct"

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selected_prompt": self.selected_variant,
            "selected_execution_mode": self.execution_mode,
            "decision": self.decision,
            "reason": self.reason,
            "api_calls": self.api_calls,
            "margin": self.margin,
            "candidate_source": self.candidate_source,
            "base": self.base_summary,
            "candidate": self.candidate_summary,
            "candidates": self.candidate_summaries,
            "effective_margin": self.effective_margin,
        }


@dataclass(frozen=True)
class PromptCandidate:
    """A prompt variant competing in validation selection."""

    name: str
    prompt: str
    source: str
    report_decision: str
    report_accepted: bool = False


@dataclass(frozen=True)
class EvaluationCandidate:
    """One prompt/execution-mode option evaluated by the validation gate."""

    prompt_name: str
    prompt: str
    source: str
    report_decision: str
    report_accepted: bool
    execution_mode: str

    @property
    def name(self) -> str:
        return _evaluation_candidate_key(self.prompt_name, self.execution_mode)


class ParetoPromptMutation(StrictModel):
    """One optimizer-proposed prompt candidate for validation selection."""

    mutation_id: str
    title: str
    strategy: str
    prompt: str
    expected_fixed_behaviors: List[str]
    expected_unchanged_behaviors: List[str]
    risk_notes: List[str]
    evidence_ids: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    expected_gain: float = Field(ge=0.0, le=1.0)
    regression_risk: float = Field(ge=0.0, le=1.0)


class ParetoPromptMutationBatch(StrictModel):
    """Schema returned by the GAVEL-Pareto prompt mutation step."""

    mutations: List[ParetoPromptMutation]
    no_mutation_reason: Optional[str]


@dataclass
class ParetoSearchResult:
    """Optimizer-proposed prompt candidates and run metadata."""

    candidates: List[PromptCandidate]
    api_calls: int
    summary: Dict[str, Any]


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
    execution_modes = _resolve_execution_modes(config.execution_modes, config.self_refine_rounds)

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
        self_refine_rounds=config.self_refine_rounds,
        execution_mode="direct",
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
    pareto_search = await _generate_pareto_prompt_candidates(
        base_prompt=base_prompt,
        benchmark_spec=benchmark_spec,
        report=report,
        logs=logs,
        optimizer_client=optimizer_client,
        max_candidates=(
            config.pareto_candidates
            if config.prompt_portfolio and config.validation_gate
            else 0
        ),
        frontier_size=config.pareto_frontier_size,
    )
    prompt_candidates = _build_prompt_candidates(
        base_prompt=base_prompt,
        report=report,
        benchmark_spec=benchmark_spec,
        logs=logs,
        pareto_candidates=pareto_search.candidates,
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
        confidence_z=config.validation_confidence_z,
        rejected_candidate_margin=config.rejected_candidate_margin,
        prompt_complexity_margin=config.prompt_complexity_margin,
        self_refine_rounds=config.self_refine_rounds,
        execution_modes=execution_modes,
    )
    report.metadata = {
        **report.metadata,
        "pareto_search": pareto_search.summary,
        "validation_gate": prompt_selection.summary(),
    }
    optimized_prompt = prompt_selection.prompt
    reference_execution_mode = "direct" if "direct" in execution_modes else execution_modes[0]

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
        self_refine_rounds=config.self_refine_rounds,
        execution_mode=prompt_selection.execution_mode,
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
        evidence_stats.api_calls
        + report.call_ledger.total_calls
        + pareto_search.api_calls
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
    method_label = "gavel_pareto" if pareto_search.summary.get("enabled") else "gavel"
    summary = {
        **summarize_scores(scores),
        "method": method_label,
        "optimizer": method_label,
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
        "pareto_candidates": config.pareto_candidates,
        "pareto_frontier_size": config.pareto_frontier_size,
        "pareto_search": pareto_search.summary,
        "self_refine_rounds": config.self_refine_rounds,
        "execution_modes": list(execution_modes),
        "validation_confidence_z": config.validation_confidence_z,
        "rejected_candidate_margin": config.rejected_candidate_margin,
        "prompt_complexity_margin": config.prompt_complexity_margin,
        "accepted": (
            prompt_selection.selected_variant != "base"
            or prompt_selection.execution_mode != reference_execution_mode
        ),
        "decision": prompt_selection.decision,
        "selected_prompt": prompt_selection.selected_variant,
        "selected_execution_mode": prompt_selection.execution_mode,
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
    self_refine_rounds: int = 0,
    execution_mode: str = "auto",
) -> tuple[List[Dict[str, Any]], List[ScoreResult], EvaluationStats]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if self_refine_rounds < 0:
        raise ValueError("self_refine_rounds must be non-negative")
    resolved_execution_mode = _resolve_execution_mode(execution_mode, self_refine_rounds)
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
                output, raw_output, call_count = await _complete_with_execution_mode(
                    prompt=prompt,
                    user_input=example.prompt,
                    target_client=target_client,
                    metadata={
                        "phase": phase,
                        "example_id": example.example_id,
                        "benchmark_id": example.benchmark_id,
                    },
                    self_refine_rounds=self_refine_rounds,
                    execution_mode=resolved_execution_mode,
                )
                stats.api_calls += call_count
            finally:
                async with active_lock:
                    active -= 1
            prediction = "" if output is None else str(output)
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
                    "raw_prediction": raw_output,
                    "execution_mode": resolved_execution_mode,
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


async def _complete_with_execution_mode(
    *,
    prompt: str,
    user_input: Any,
    target_client: TargetClient,
    metadata: Mapping[str, Any],
    self_refine_rounds: int,
    execution_mode: str,
) -> tuple[Any, Any, int]:
    mode = _resolve_execution_mode(execution_mode, self_refine_rounds)
    if mode == "direct":
        response = await target_client.complete(prompt, user_input, metadata=metadata)
        return response.output, response.output, 1
    if mode == "self_refine":
        return await _complete_with_self_refinement(
            prompt=prompt,
            user_input=user_input,
            target_client=target_client,
            metadata=metadata,
            self_refine_rounds=max(1, self_refine_rounds),
        )
    if mode == "plan":
        return await _complete_with_planning(
            prompt=prompt,
            user_input=user_input,
            target_client=target_client,
            metadata=metadata,
        )
    if mode == "plan_refine":
        return await _complete_with_planning(
            prompt=prompt,
            user_input=user_input,
            target_client=target_client,
            metadata=metadata,
            refine_rounds=max(1, self_refine_rounds),
        )
    raise ValueError(f"unsupported execution mode: {execution_mode}")


async def _complete_with_planning(
    *,
    prompt: str,
    user_input: Any,
    target_client: TargetClient,
    metadata: Mapping[str, Any],
    refine_rounds: int = 0,
) -> tuple[Any, Any, int]:
    execution_mode = "plan_refine" if refine_rounds > 0 else "plan"
    plan_metadata = {
        **dict(metadata),
        "phase": f"{metadata.get('phase', 'target')}_plan_contract",
        "execution_mode": execution_mode,
    }
    plan_response = await target_client.complete(
        _planning_prompt(prompt),
        _planning_input(user_input),
        metadata=plan_metadata,
    )
    answer_metadata = {
        **dict(metadata),
        "phase": f"{metadata.get('phase', 'target')}_plan_answer",
        "execution_mode": execution_mode,
    }
    answer_response = await target_client.complete(
        _planned_answer_prompt(prompt),
        _planned_answer_input(user_input, plan_response.output),
        metadata=answer_metadata,
    )
    output = answer_response.output
    call_count = 2
    refinement_outputs: List[Any] = []
    for round_index in range(refine_rounds):
        refine_metadata = {
            **dict(metadata),
            "phase": f"{metadata.get('phase', 'target')}_plan_refine_{round_index + 1}",
            "execution_mode": execution_mode,
            "self_refine_round": round_index + 1,
        }
        refine_response = await target_client.complete(
            _planned_refinement_prompt(prompt),
            _planned_refinement_input(user_input, plan_response.output, output),
            metadata=refine_metadata,
        )
        output = refine_response.output
        refinement_outputs.append(output)
        call_count += 1
    raw_output = {
        "execution_mode": execution_mode,
        "task_contract": plan_response.output,
        "final_answer": answer_response.output,
        "refined_answer": output if refine_rounds > 0 else None,
        "refinement_outputs": refinement_outputs,
    }
    return output, raw_output, call_count


async def _complete_with_self_refinement(
    *,
    prompt: str,
    user_input: Any,
    target_client: TargetClient,
    metadata: Mapping[str, Any],
    self_refine_rounds: int,
) -> tuple[Any, Any, int]:
    response = await target_client.complete(prompt, user_input, metadata=metadata)
    output = response.output
    raw_output = response.output
    call_count = 1
    if self_refine_rounds <= 0:
        return output, raw_output, call_count

    refine_prompt = _self_refinement_prompt(prompt)
    for round_index in range(self_refine_rounds):
        refine_metadata = {
            **dict(metadata),
            "phase": f"{metadata.get('phase', 'target')}_self_refine_{round_index + 1}",
            "self_refine_round": round_index + 1,
        }
        response = await target_client.complete(
            refine_prompt,
            _self_refinement_input(user_input, output),
            metadata=refine_metadata,
        )
        output = response.output
        raw_output = response.output
        call_count += 1
    return output, raw_output, call_count


def _planning_prompt(runtime_prompt: str) -> str:
    clipped_prompt = _truncate_text(runtime_prompt.strip(), max_chars=9000)
    sections = [
        (
            "Goal",
            [
                "Compile the original user prompt into a compact task contract that can guide a strict final answer.",
            ],
        ),
        (
            "Context",
            [
                "This is a label-free planning pass. You do not have benchmark labels, hidden evaluator data, expected answers, or scorer feedback.",
                "The runtime instructions that govern the final answer are included below. Apply them as authoritative guidance.",
                clipped_prompt,
            ],
        ),
        (
            "Role",
            [
                "Act as a precise task analyst and constraint compiler.",
            ],
        ),
        (
            "Input",
            [
                "You will receive one original user prompt.",
            ],
        ),
        (
            "Task",
            [
                "Extract the user's goal, the required answer shape, and every explicit hard constraint.",
                "Identify counts, ordering, required text, forbidden text, casing, delimiters, language, units, choices, code requirements, and edge cases when present.",
                "For reasoning tasks, identify the shortest path to the final answer and the normalized output expected by the prompt.",
                "For creative or open-ended tasks, identify the mechanical constraints that must still be satisfied exactly.",
                "For word-property constraints (vowel start/end, consonant clusters, consonant counts, syllable parity, prime word lengths, alphabetical order, first/last letters): pre-select a verified pool of at least 10 candidate words that satisfy the property; record them in word_pools so the answer pass can draw from them without guessing. Verify each candidate explicitly: vowels are {a,e,i,o,u}; consonants are all other letters; syllables = number of vowel clusters; prime lengths are {2,3,5,7,11,13}.",
                "For position-chain constraints (last word of sentence/paragraph = first word of the next): write out the complete ordered list of boundary words before anything else.",
                "For ratio or overlap constraints: compute the target value numerically and record it (e.g., 'overlap ratio must be 0.5: out of N content words, M must appear in both sections').",
                "For repeated-span constraints: write the exact span once, then note how many copies and what modifications are allowed.",
            ],
        ),
        (
            "Constraints",
            [
                "Do not solve using hidden labels or evaluator metadata.",
                "Do not invent requirements that are not supported by the current user prompt.",
                "Keep the contract concise enough to be useful in a second pass.",
            ],
        ),
        (
            "Output Format",
            [
                "Return JSON with keys: goal, answer_shape, hard_constraints, word_pools, boundary_chain, ratio_targets, solve_plan, final_checks.",
                "word_pools: object mapping each word-property constraint name to an array of pre-verified candidate words (empty object if no word-property constraints).",
                "boundary_chain: ordered array of boundary words for chaining constraints (empty array if not applicable).",
                "ratio_targets: object mapping each ratio/overlap constraint to its numeric target and how to measure it (empty object if not applicable).",
                "Use short strings or arrays of short strings. Do not include markdown fences.",
            ],
        ),
        (
            "Quality Bar",
            [
                "The contract is complete only if a second pass can use it—including word_pools and boundary_chain—to produce an answer that a strict mechanical verifier could score without additional guessing.",
            ],
        ),
    ]
    return _render_structured_sections(sections)


def _planning_input(user_input: Any) -> str:
    input_text = user_input if isinstance(user_input, str) else json.dumps(user_input, sort_keys=True)
    return (
        "Original user prompt:\n"
        f"{input_text}\n\n"
        "Return only the task contract JSON."
    )


def _planned_answer_prompt(runtime_prompt: str) -> str:
    clipped_prompt = _truncate_text(runtime_prompt.strip(), max_chars=9000)
    sections = [
        (
            "Goal",
            [
                "Produce the final answer requested by the original user prompt using the task contract.",
            ],
        ),
        (
            "Context",
            [
                "A previous label-free pass extracted a task contract from the current user prompt.",
                "The runtime instructions that govern the final answer are included below. Apply them as authoritative guidance.",
                clipped_prompt,
            ],
        ),
        (
            "Role",
            [
                "Act as a strict solver and final-answer editor.",
            ],
        ),
        (
            "Input",
            [
                "You will receive the original user prompt and the task contract.",
            ],
        ),
        (
            "Task",
            [
                "Solve the original user prompt directly.",
                "For word-property constraints: use ONLY words from the contract's word_pools. Do not use any word outside the pool for a property-constrained position unless you can explicitly verify it letter-by-letter satisfies the property.",
                "For chaining constraints: follow the boundary_chain exactly; the last word of each sentence or paragraph must match the first word of the next.",
                "For ratio or overlap constraints: use ratio_targets as the numeric goal and verify it after drafting.",
                "Use the task contract to satisfy every hard constraint before optimizing style or elaboration.",
                "Privately run the final checks from the contract and minimally revise any failing part.",
            ],
        ),
        (
            "Constraints",
            [
                "Do not mention the task contract, planning pass, validation, benchmark, or hidden reasoning.",
                "Do not add wrappers, labels, markdown fences, explanations, or extra sections unless the original user prompt requires them.",
                "If the contract conflicts with the original user prompt, obey the original user prompt.",
            ],
        ),
        (
            "Output Format",
            [
                "Return only the final answer requested by the original user prompt.",
            ],
        ),
        (
            "Quality Bar",
            [
                "The final answer is acceptable only if it satisfies the original prompt and every explicit mechanical constraint.",
            ],
        ),
    ]
    return _render_structured_sections(sections)


def _planned_answer_input(user_input: Any, task_contract: Any) -> str:
    input_text = user_input if isinstance(user_input, str) else json.dumps(user_input, sort_keys=True)
    contract_text = "" if task_contract is None else str(task_contract)
    return (
        "Original user prompt:\n"
        f"{input_text}\n\n"
        "Task contract from label-free planning pass:\n"
        f"{contract_text}\n\n"
        "Return only the final answer."
    )


def _planned_refinement_prompt(runtime_prompt: str) -> str:
    clipped_prompt = _truncate_text(runtime_prompt.strip(), max_chars=9000)
    sections = [
        (
            "Goal",
            [
                "Repair the draft final answer using the original prompt and the extracted task contract.",
            ],
        ),
        (
            "Context",
            [
                "This is a label-free verification pass. You do not have benchmark labels, hidden evaluator data, expected answers, or scorer feedback.",
                "The runtime instructions that govern the final answer are included below. Apply them as authoritative guidance.",
                clipped_prompt,
            ],
        ),
        (
            "Role",
            [
                "Act as a strict mechanical verifier and final-answer editor.",
            ],
        ),
        (
            "Input",
            [
                "You will receive the original user prompt, the extracted task contract, and a draft answer.",
            ],
        ),
        (
            "Task",
            [
                "Privately verify the draft against every hard constraint in the original prompt and task contract.",
                "If a constraint is violated, rewrite only as much as needed to satisfy it.",
                "For exact counts, first choose the target count, then count the final answer before returning it.",
                "For no-repeat or uniqueness constraints, scan all words after lowercasing and stripping punctuation; replace repeats.",
                "For no-consecutive constraints, check every adjacent word pair and repair any pair with the forbidden relation.",
                "For word-property constraints, use only words you can verify letter-by-letter; vowels are {a,e,i,o,u}, consonants are all other letters, syllables are vowel clusters, and prime lengths are {2,3,5,7,11,13}.",
                "For ratio, balance, or overlap constraints, compute the requested counts numerically and edit until the measured values match.",
                "For boundary-chain constraints, compare exact normalized boundary words and repair mismatches.",
                "For format constraints, make the visible surface form exact: delimiters, bullets, numbering, casing, quotes, whitespace, and line breaks.",
                "If the draft is already compliant, return it unchanged.",
            ],
        ),
        (
            "Constraints",
            [
                "Do not mention verification, planning, the task contract, benchmark metadata, or hidden reasoning.",
                "Do not add explanations, labels, markdown fences, or wrappers unless the original user prompt asks for them.",
                "Obey the original user prompt over the task contract if they conflict.",
            ],
        ),
        (
            "Output Format",
            [
                "Return only the corrected final answer requested by the original prompt.",
            ],
        ),
        (
            "Quality Bar",
            [
                "The answer is complete only if the visible output itself satisfies the task; hidden analysis cannot rescue a constraint violation.",
            ],
        ),
    ]
    return _render_structured_sections(sections)


def _planned_refinement_input(
    user_input: Any,
    task_contract: Any,
    draft_output: Any,
) -> str:
    input_text = user_input if isinstance(user_input, str) else json.dumps(user_input, sort_keys=True)
    contract_text = "" if task_contract is None else str(task_contract)
    draft_text = "" if draft_output is None else str(draft_output)
    return (
        "Original user prompt:\n"
        f"{input_text}\n\n"
        "Task contract from label-free planning pass:\n"
        f"{contract_text}\n\n"
        "Draft final answer:\n"
        f"{draft_text}\n\n"
        "Return only the corrected final answer."
    )


def _self_refinement_prompt(runtime_prompt: str) -> str:
    clipped_prompt = _truncate_text(runtime_prompt.strip(), max_chars=9000)
    sections = [
        (
            "Goal",
            [
                "Repair a draft answer so it satisfies the original user task and the runtime instructions.",
            ],
        ),
        (
            "Context",
            [
                "This is a label-free verification pass. You do not have benchmark labels or scorer feedback.",
                "The runtime instructions that governed the draft are included below. Apply them as authoritative guidance.",
                clipped_prompt,
            ],
        ),
        (
            "Role",
            [
                "Act as a strict final-answer verifier and editor.",
            ],
        ),
        (
            "Input",
            [
                "You will receive the original user prompt and the draft answer.",
            ],
        ),
        (
            "Task",
            [
                "Privately extract every explicit requirement from the original user prompt.",
                "Check the draft against task intent, output format, counts, ordering, required content, prohibited content, and surface-form constraints.",
                "For vowel or consonant constraints: go word-by-word through the draft. Vowels are {a,e,i,o,u}; consonants are all other letters. Flag any word that violates the constraint and replace it.",
                "For syllable odd/even or syllable-count constraints: count vowel clusters per word (each consecutive group of vowels = one syllable). Flag any word with wrong parity or count and replace it.",
                "For prime word-length constraints: prime lengths ≤ 15 are {2,3,5,7,11,13}. Count each word's letters and flag violations.",
                "For consonant-cluster constraints: scan each word for consecutive consonant pairs. If any word has no consecutive consonants, replace it with one that does.",
                "For ratio or overlap constraints: count actual values from the draft numerically, compute the ratio, and edit until it matches the requirement.",
                "For position-chain constraints: verify the exact last word of each sentence or paragraph matches the exact first word of the next. Fix any mismatch by adjusting the boundary word.",
                "If the draft is already correct, return it unchanged.",
                "If the draft violates any requirement, minimally rewrite the failing parts until all constraints are satisfied.",
            ],
        ),
        (
            "Constraints",
            [
                "Do not use or invent benchmark labels, expected answers, hidden evaluator data, or external feedback.",
                "Preserve correct parts of the draft when possible.",
                "For math, exact-match, multiple-choice, code, QA, and instruction-following tasks, optimize the final answer for strict automated scoring.",
            ],
        ),
        (
            "Output Format",
            [
                "Return only the corrected final answer.",
                "Do not include analysis, checklists, labels such as 'Corrected answer:', markdown fences, or explanations unless the original user prompt explicitly requires them.",
            ],
        ),
        (
            "Quality Bar",
            [
                "The final answer is acceptable only if a strict evaluator could score it without reading any hidden reasoning.",
            ],
        ),
    ]
    return _render_structured_sections(sections)


def _self_refinement_input(user_input: Any, draft_output: Any) -> str:
    input_text = user_input if isinstance(user_input, str) else json.dumps(user_input, sort_keys=True)
    draft_text = "" if draft_output is None else str(draft_output)
    return (
        "Original user prompt:\n"
        f"{input_text}\n\n"
        "Draft answer:\n"
        f"{draft_text}\n\n"
        "Return only the corrected final answer."
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


def constraint_solver_prompt(spec: Optional[BenchmarkSpec] = None) -> str:
    sections = _default_prompt_sections(spec)
    _extend_section(
        sections,
        "Role",
        [
            "Act as a constraint solver first and a writer second.",
        ],
    )
    _extend_section(
        sections,
        "Task",
        [
            "Privately compile the prompt into measurable constraints before writing any final text.",
            "Choose an output skeleton with the required number of sentences, lines, bullets, choices, code blocks, tokens, or answer fields before filling content.",
            "Resolve the most evaluator-visible constraints first: exact answer shape, required labels, counts, positions, allowed characters, forbidden content, and formatting.",
            "Use low-variance wording. Prefer simple reusable structures over creative prose when creativity increases counting or formatting risk.",
            "After drafting, run a final private pass over each measurable constraint and minimally edit any failing part.",
        ],
    )
    _extend_section(
        sections,
        "Constraints",
        [
            "When two requirements compete, preserve the explicit output format and mechanically checkable requirements before style.",
            "Do not carry constraints from prior examples into the current user prompt.",
            "Do not introduce decorative wrappers, markup, brackets, bullets, explanations, or extra sections unless the current user prompt asks for them.",
        ],
    )
    task_type = spec.task_type if spec else None
    if task_type == BenchmarkTaskType.INSTRUCTION_FOLLOWING:
        _extend_section(
            sections,
            "Task",
            [
                "For exact word counts or ranges, draft with short common words and count the final visible words only.",
                "For unique-word constraints, avoid reusing any content word and check case-insensitively.",
                "For keyword-position constraints, place the required word first, last, or at the requested index before drafting surrounding text.",
                "For vowel constraints, screen each candidate word by its first or last letter: vowels are a, e, i, o, u only.",
                "For consonant constraints, count non-vowel letters (b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z) per word letter-by-letter before selecting it.",
                "For syllable odd/even constraints, count vowel clusters per word (each consecutive vowel run = one syllable), then check parity; do not rely on intuition alone.",
                "For prime word-length constraints, the prime lengths ≤ 15 are exactly 2, 3, 5, 7, 11, 13; reject words whose character count is not in this set.",
                "For last-word-to-first-word chaining across sentences or paragraphs, write the boundary words for every unit first, then fill the interior content.",
                "For overlap or balance ratio constraints, tally shared words or sentence types in your draft numerically and revise until the ratio is satisfied.",
                "For repeated-span constraints, create the exact span once, copy it the requested number of times, then apply only the allowed changes.",
                "For no-whitespace, indentation, quote, title-case, option, punctuation, or bracket constraints, verify the final visible characters exactly.",
            ],
        )
    elif task_type == BenchmarkTaskType.MATH:
        _extend_section(
            sections,
            "Task",
            [
                "Track units and operations explicitly, then put only the normalized final value in the requested answer shape.",
            ],
        )
    elif task_type == BenchmarkTaskType.CODE_GENERATION:
        _extend_section(
            sections,
            "Task",
            [
                "Treat the prompt and public tests as an executable specification; cover edge cases before returning code.",
            ],
        )
    elif task_type == BenchmarkTaskType.MULTIPLE_CHOICE:
        _extend_section(
            sections,
            "Task",
            [
                "Compare every option against the question, then output one label with no alternate labels.",
            ],
        )
    elif task_type == BenchmarkTaskType.QUESTION_ANSWERING:
        _extend_section(
            sections,
            "Task",
            [
                "Constrain the answer to the shortest supported span or value that satisfies the question.",
            ],
        )
    metric_items = _metric_strategy_items(spec.metrics if spec else [])
    if metric_items:
        _extend_section(sections, "Quality Bar", metric_items)
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
        task.append("Privately enumerate every measurable constraint before drafting: word/sentence/line counts, ratios, required words, forbidden words, casing, delimiters, and word-property rules.")
        task.append("For count or ratio constraints, decide the exact target value first, choose a simple output skeleton, then count your draft and revise until exact.")
        task.append("For vowel/consonant constraints: vowels are {a, e, i, o, u}; consonants are all other letters. Count letter-by-letter for each candidate word before committing to it.")
        task.append("For syllable odd/even constraints: count syllables by counting distinct vowel clusters in a word (each consecutive run of vowels = one syllable). Odd = 1,3,5,...; even = 2,4,6,... Verify parity per word.")
        task.append("For prime word-length constraints: prime character counts ≤ 15 are 2, 3, 5, 7, 11, 13. Non-prime: 1, 4, 6, 8, 9, 10, 12, 14, 15. Check each required word's length explicitly.")
        task.append("For last-word-to-first-word chaining across sentences or paragraphs: plan the full chain of boundary words before writing any content, then fill each unit around that skeleton.")
        task.append("For ratio or overlap constraints: after drafting, compute the ratio numerically—count shared words between sections or sentence-type distributions—and adjust the draft until the ratio matches.")
        task.append("For keyword-in-every-sentence or keyword-at-position constraints: write each sentence starting from the required keyword rather than inserting it afterward.")
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


_ALLOWED_EXECUTION_MODES = {"direct", "self_refine", "plan", "plan_refine"}


def _resolve_execution_mode(mode: str, self_refine_rounds: int) -> str:
    normalized = str(mode or "auto").strip().lower().replace("-", "_")
    if normalized == "auto":
        return "self_refine" if self_refine_rounds > 0 else "direct"
    if normalized not in _ALLOWED_EXECUTION_MODES:
        raise ValueError(
            "execution mode must be one of: "
            + ", ".join(sorted(_ALLOWED_EXECUTION_MODES | {"auto"}))
        )
    if normalized == "self_refine" and self_refine_rounds <= 0:
        return "direct"
    return normalized


def _resolve_execution_modes(
    modes: Sequence[str],
    self_refine_rounds: int,
) -> Tuple[str, ...]:
    if not modes:
        raise ValueError("execution_modes must include at least one mode")
    resolved: List[str] = []
    for mode in modes:
        normalized = _resolve_execution_mode(mode, self_refine_rounds)
        if normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError("execution_modes must include at least one executable mode")
    return tuple(resolved)


def _evaluation_candidate_key(prompt_name: str, execution_mode: str) -> str:
    return f"{prompt_name}/{execution_mode}"


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
        "words": (
            "for word-property constraints, verify each property mechanically before choosing a word: "
            "(a) vowel start/end = first or last letter in {a,e,i,o,u}; "
            "(b) consonant count = count every letter that is NOT a,e,i,o,u; "
            "(c) syllable odd/even = count vowel clusters per word (each consecutive vowel run = one syllable), then check parity; "
            "(d) prime word length = character count must be in {2,3,5,7,11,13}; "
            "(e) last-first chaining = the exact last word of one sentence/paragraph must be the exact first word of the next"
        ),
        "count": "for count constraints, decide the exact number of words, lines, list items, or occurrences before writing, then count the draft and adjust",
        "ratio": (
            "for ratio constraints, compute the ratio numerically after drafting: "
            "for overlap ratios count shared content words between sections; "
            "for sentence-type balance count each type (declarative, interrogative, exclamatory); "
            "for word-per-sentence ratios count words per sentence and verify the required proportion"
        ),
        "sentence": "for sentence constraints, construct and verify one sentence at a time—check required keywords, punctuation, alliteration, length, or ordering before moving on",
        "format": "for format constraints, match requested delimiters, casing, whitespace, bullets, quotes, and line breaks exactly",
        "repeat": "for repetition constraints, create the exact span once, copy it the required number of times, then apply only the allowed modifications",
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


async def _generate_pareto_prompt_candidates(
    *,
    base_prompt: str,
    benchmark_spec: Optional[BenchmarkSpec],
    report: Any,
    logs: Sequence[Mapping[str, Any]],
    optimizer_client: OptimizerClient,
    max_candidates: int,
    frontier_size: int,
) -> ParetoSearchResult:
    requested = max(0, max_candidates)
    frontier_limit = max(0, min(frontier_size, requested))
    if requested == 0 or frontier_limit == 0:
        return ParetoSearchResult(
            candidates=[],
            api_calls=0,
            summary={
                "enabled": False,
                "requested_candidates": requested,
                "frontier_size": frontier_limit,
                "proposed": 0,
                "selected": 0,
                "api_calls": 0,
                "reason": "pareto search disabled",
            },
        )

    response = await optimizer_client.complete(
        _build_pareto_mutation_messages(
            base_prompt=base_prompt,
            benchmark_spec=benchmark_spec,
            report=report,
            logs=logs,
            max_candidates=requested,
        ),
        response_schema=ParetoPromptMutationBatch.model_json_schema(),
        metadata={
            "phase": "pareto_prompt_mutations",
            "max_candidates": requested,
            "frontier_size": frontier_limit,
        },
    )
    api_calls = 1
    try:
        batch = strict_parse_json_model(ParetoPromptMutationBatch, response.output)
    except (TypeError, ValueError, ValidationError) as exc:
        return ParetoSearchResult(
            candidates=[],
            api_calls=api_calls,
            summary={
                "enabled": True,
                "requested_candidates": requested,
                "frontier_size": frontier_limit,
                "proposed": 0,
                "selected": 0,
                "api_calls": api_calls,
                "error": "mutation_schema_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )

    mutations = _sanitize_pareto_mutations(
        batch.mutations,
        base_prompt=base_prompt,
        max_candidates=requested,
    )
    frontier = _pareto_frontier(mutations, base_prompt=base_prompt)
    ranked_frontier = _rank_pareto_mutations(frontier, base_prompt=base_prompt)
    if len(ranked_frontier) < frontier_limit:
        frontier_ids = {id(mutation) for mutation in ranked_frontier}
        backfill = [
            mutation
            for mutation in _rank_pareto_mutations(mutations, base_prompt=base_prompt)
            if id(mutation) not in frontier_ids
        ]
        selected = (ranked_frontier + backfill)[:frontier_limit]
    else:
        selected = ranked_frontier[:frontier_limit]
    candidates = [
        _pareto_mutation_candidate(mutation, index=index)
        for index, mutation in enumerate(selected, start=1)
    ]
    return ParetoSearchResult(
        candidates=candidates,
        api_calls=api_calls,
        summary={
            "enabled": True,
            "requested_candidates": requested,
            "frontier_size": frontier_limit,
            "proposed": len(batch.mutations),
            "valid": len(mutations),
            "frontier": len(frontier),
            "selected": len(candidates),
            "api_calls": api_calls,
            "no_mutation_reason": batch.no_mutation_reason,
            "candidates": [
                _pareto_mutation_summary(mutation, candidate.name)
                for mutation, candidate in zip(selected, candidates)
            ],
        },
    )


def _build_pareto_mutation_messages(
    *,
    base_prompt: str,
    benchmark_spec: Optional[BenchmarkSpec],
    report: Any,
    logs: Sequence[Mapping[str, Any]],
    max_candidates: int,
) -> List[Dict[str, str]]:
    schema = json.dumps(
        _openai_strict_response_schema(ParetoPromptMutationBatch.model_json_schema()),
        indent=2,
        sort_keys=True,
    )
    evidence_payload = json.dumps(
        _pareto_evidence_payload(logs),
        indent=2,
        sort_keys=True,
    )
    report_payload = json.dumps(
        _pareto_report_payload(report),
        indent=2,
        sort_keys=True,
    )
    spec_payload = json.dumps(
        _pareto_spec_payload(benchmark_spec),
        indent=2,
        sort_keys=True,
    )
    system = (
        "You are GAVEL-Pareto, a general prompt optimizer. Generate a small "
        "Pareto frontier of transferable runtime prompts from training evidence. "
        "Optimize for validation performance, broad transfer, low regression risk, "
        "short prompts, and diversity of behavioral contracts. Return only JSON "
        "matching the schema."
    )
    user = f"""
GAVEL-Pareto mutation request

Objective:
- Propose exactly {max_candidates} complete runtime prompts unless the evidence is empty or unusable.
- Each prompt must be a general candidate for unseen validation/test examples, not a benchmark-specific solver.
- Each mutation.prompt must contain the full runtime prompt that should be sent to the target model. It must not be a diff, strategy note, title, or unchanged copy of the base prompt.
- Use training failures to infer reusable behavioral contracts, output-shape policies, reasoning policies, demo policies, or anti-patterns.
- Keep successful training rows as regression constraints.
- Do not copy training answers, example ids, labels, hidden evaluator kwargs, or per-example expected outputs into runtime prompts.
- Do not rely on scorer internals, benchmark metadata unavailable to the target model, or any held-out validation/test information.
- Diversity is mandatory. No two prompt fields may be identical or near-identical.
- Cover distinct strategy families when possible: compact strict format, conservative verifier, task-type decision rules, failure-pattern repair, regression-preserving minimalism, and robust fallback for ambiguous prompts.
- A candidate is useful only if a normal target model can execute it from the current user prompt alone.
- Return complete prompts, not diffs. Preserve the eight-section style when possible: Goal, Context, Role, Input, Task, Constraints, Output Format, Quality Bar.
- Include every schema field. Use [] for empty lists and null for no_mutation_reason when mutations are present.
- Keep every prompt transferable. If a rule came from one training row, phrase it conditionally: "When the current prompt requests X, do Y."

Base runtime prompt:
{_truncate_text(base_prompt, max_chars=12000)}

Benchmark/task summary:
{spec_payload}

Compiled GAVEL report summary:
{report_payload}

Training evidence summary:
{json.dumps(_summarize_evidence_patterns(logs), indent=2, sort_keys=True)}

Compact training evidence:
{evidence_payload}

Required JSON schema:
{schema}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _pareto_spec_payload(spec: Optional[BenchmarkSpec]) -> Dict[str, Any]:
    if spec is None:
        return {}
    return {
        "benchmark_id": spec.benchmark_id,
        "display_name": spec.display_name,
        "task_type": spec.task_type.value,
        "metrics": [
            metric.value if isinstance(metric, MetricKind) else str(metric)
            for metric in spec.metrics
        ],
        "description": spec.description,
        "evaluator": spec.evaluator,
    }


def _pareto_report_payload(report: Any) -> Dict[str, Any]:
    decision = getattr(report, "decision", None)
    patch = getattr(report, "patch", None)
    defects = []
    for defect in list(getattr(report, "defect_posterior", []) or [])[:6]:
        dumped = defect.model_dump(mode="json") if hasattr(defect, "model_dump") else dict(defect)
        defects.append(_compact_value(dumped, max_chars=500))
    payload: Dict[str, Any] = {
        "accepted": bool(getattr(report, "accepted", False)),
        "decision": getattr(decision, "value", str(decision)),
        "notes": [_truncate_text(str(note), max_chars=400) for note in list(getattr(report, "notes", []) or [])[:8]],
        "defect_posterior": defects,
    }
    if patch is not None:
        payload["patch"] = {
            "summary": _truncate_text(str(getattr(patch, "summary", "")), max_chars=500),
            "expected_fixed_behaviors": [
                _truncate_text(str(item), max_chars=240)
                for item in list(getattr(patch, "expected_fixed_behaviors", []) or [])[:8]
            ],
            "expected_unchanged_behaviors": [
                _truncate_text(str(item), max_chars=240)
                for item in list(getattr(patch, "expected_unchanged_behaviors", []) or [])[:8]
            ],
            "regression_risks": [
                _truncate_text(str(item), max_chars=240)
                for item in list(getattr(patch, "regression_risks", []) or [])[:8]
            ],
        }
    return payload


def _pareto_evidence_payload(logs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    failures = [row for row in logs if row.get("passed") is False]
    successes = [row for row in logs if row.get("passed") is True]
    selected = failures[:12] + successes[:4]
    payload: List[Dict[str, Any]] = []
    for row in selected:
        metadata = dict(row.get("metadata") or {})
        payload.append(
            {
                "evidence_id": row.get("evidence_id"),
                "passed": row.get("passed"),
                "area": row.get("area"),
                "severity": row.get("severity"),
                "prompt_fixability": row.get("prompt_fixability"),
                "input": _truncate_text(str(row.get("input") or ""), max_chars=1000),
                "expected": _truncate_text(str(row.get("expected") or ""), max_chars=700),
                "observed": _truncate_text(str(row.get("observed") or ""), max_chars=900),
                "metadata": _compact_value(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "metric",
                            "diagnostics",
                            "choices",
                            "public_tests",
                            "failed_instruction_ids",
                        }
                    },
                    max_chars=500,
                ),
            }
        )
    return payload


def _sanitize_pareto_mutations(
    mutations: Sequence[ParetoPromptMutation],
    *,
    base_prompt: str,
    max_candidates: int,
) -> List[ParetoPromptMutation]:
    seen = {base_prompt.strip()}
    sanitized: List[ParetoPromptMutation] = []
    for mutation in mutations:
        prompt = mutation.prompt.strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        sanitized.append(
            mutation.model_copy(
                update={
                    "mutation_id": _slug(mutation.mutation_id) or f"mutation_{len(sanitized) + 1}",
                    "title": _truncate_text(mutation.title.strip() or "Pareto prompt mutation", max_chars=100),
                    "strategy": _truncate_text(mutation.strategy.strip() or "general", max_chars=80),
                    "prompt": _truncate_text(prompt, max_chars=60000),
                }
            )
        )
        if len(sanitized) >= max_candidates:
            break
    return sanitized


def _pareto_frontier(
    mutations: Sequence[ParetoPromptMutation],
    *,
    base_prompt: str,
) -> List[ParetoPromptMutation]:
    frontier: List[ParetoPromptMutation] = []
    for candidate in mutations:
        if any(_pareto_dominates(other, candidate, base_prompt=base_prompt) for other in mutations if other is not candidate):
            continue
        frontier.append(candidate)
    return frontier


def _pareto_dominates(
    left: ParetoPromptMutation,
    right: ParetoPromptMutation,
    *,
    base_prompt: str,
) -> bool:
    left_values = _pareto_objectives(left, base_prompt=base_prompt)
    right_values = _pareto_objectives(right, base_prompt=base_prompt)
    better_or_equal = all(left_values[key] >= right_values[key] for key in left_values)
    strictly_better = any(left_values[key] > right_values[key] for key in left_values)
    return better_or_equal and strictly_better


def _pareto_objectives(
    mutation: ParetoPromptMutation,
    *,
    base_prompt: str,
) -> Dict[str, float]:
    prompt_tokens = max(1, estimate_token_count(mutation.prompt))
    base_tokens = max(1, estimate_token_count(base_prompt))
    token_ratio = prompt_tokens / base_tokens
    return {
        "expected_gain": float(mutation.expected_gain),
        "confidence": float(mutation.confidence),
        "support": min(1.0, len(mutation.evidence_ids) / 8.0),
        "low_regression": 1.0 - float(mutation.regression_risk),
        "compactness": 1.0 / max(1.0, token_ratio),
    }


def _rank_pareto_mutations(
    mutations: Sequence[ParetoPromptMutation],
    *,
    base_prompt: str,
) -> List[ParetoPromptMutation]:
    by_strategy: Dict[str, List[ParetoPromptMutation]] = {}
    for mutation in mutations:
        by_strategy.setdefault(_slug(mutation.strategy) or "general", []).append(mutation)
    for strategy, items in by_strategy.items():
        by_strategy[strategy] = sorted(items, key=lambda item: _pareto_rank_key(item, base_prompt=base_prompt), reverse=True)
    ranked: List[ParetoPromptMutation] = []
    while by_strategy:
        for strategy in list(by_strategy):
            items = by_strategy[strategy]
            if not items:
                del by_strategy[strategy]
                continue
            ranked.append(items.pop(0))
    return sorted(ranked, key=lambda item: _pareto_rank_key(item, base_prompt=base_prompt), reverse=True)


def _pareto_rank_key(
    mutation: ParetoPromptMutation,
    *,
    base_prompt: str,
) -> tuple[float, float, float, int]:
    objectives = _pareto_objectives(mutation, base_prompt=base_prompt)
    score = (
        0.40 * objectives["expected_gain"]
        + 0.20 * objectives["confidence"]
        + 0.15 * objectives["support"]
        + 0.15 * objectives["low_regression"]
        + 0.10 * objectives["compactness"]
    )
    return (
        score,
        objectives["expected_gain"],
        objectives["low_regression"],
        -estimate_token_count(mutation.prompt),
    )


def _pareto_mutation_candidate(
    mutation: ParetoPromptMutation,
    *,
    index: int,
) -> PromptCandidate:
    slug = _slug(mutation.title) or _slug(mutation.mutation_id) or "mutation"
    return PromptCandidate(
        name=f"pareto_{index}_{slug[:28]}",
        prompt=mutation.prompt,
        source=f"pareto:{_slug(mutation.strategy) or 'general'}",
        report_decision="validation_selected_pareto",
        report_accepted=False,
    )


def _pareto_mutation_summary(
    mutation: ParetoPromptMutation,
    candidate_name: str,
) -> Dict[str, Any]:
    return {
        "candidate": candidate_name,
        "mutation_id": mutation.mutation_id,
        "title": mutation.title,
        "strategy": mutation.strategy,
        "expected_gain": mutation.expected_gain,
        "confidence": mutation.confidence,
        "regression_risk": mutation.regression_risk,
        "evidence_ids": list(mutation.evidence_ids[:12]),
        "prompt_tokens": estimate_token_count(mutation.prompt),
        "expected_fixed_behaviors": [
            _truncate_text(str(item), max_chars=180)
            for item in mutation.expected_fixed_behaviors[:6]
        ],
        "risk_notes": [
            _truncate_text(str(item), max_chars=180)
            for item in mutation.risk_notes[:6]
        ],
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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
    confidence_z: float,
    rejected_candidate_margin: float,
    prompt_complexity_margin: float,
    self_refine_rounds: int,
    execution_modes: Sequence[str],
) -> PromptSelection:
    unique_candidates = _dedupe_candidates(candidates, base_prompt=base_prompt)
    resolved_execution_modes = _resolve_execution_modes(execution_modes, self_refine_rounds)
    fallback_execution_mode = _resolve_execution_mode("auto", self_refine_rounds)
    if not enabled:
        candidate = _preferred_candidate(unique_candidates)
        selected_variant = candidate.name if candidate is not None and candidate.report_accepted else "base"
        return PromptSelection(
            prompt=candidate.prompt if candidate is not None and selected_variant != "base" else base_prompt,
            selected_variant=selected_variant,
            execution_mode=fallback_execution_mode,
            decision=candidate.report_decision if candidate is not None else "base",
            reason="validation gate disabled",
            margin=margin,
            enabled=False,
            candidate_source=candidate.source if candidate is not None else "base",
            effective_margin=margin,
        )
    if not unique_candidates and len(resolved_execution_modes) == 1:
        return PromptSelection(
            prompt=base_prompt,
            selected_variant="base",
            execution_mode=fallback_execution_mode,
            decision="base",
            reason="no validation candidates are available",
            margin=margin,
            candidate_source="base",
            effective_margin=margin,
        )
    if not val_examples:
        candidate = _preferred_candidate(unique_candidates)
        if candidate is None:
            return PromptSelection(
                prompt=base_prompt,
                selected_variant="base",
                execution_mode=fallback_execution_mode,
                decision="base",
                reason="no validation examples available",
                margin=margin,
                candidate_source="base",
                effective_margin=margin,
            )
        return PromptSelection(
            prompt=candidate.prompt,
            selected_variant=candidate.name,
            execution_mode=fallback_execution_mode,
            decision=candidate.report_decision,
            reason="no validation examples available",
            margin=margin,
            candidate_source=candidate.source,
            effective_margin=margin,
        )

    reference_mode = "direct" if "direct" in resolved_execution_modes else resolved_execution_modes[0]
    base_predictions, base_scores, base_stats = await evaluate_prompt(
        prompt=base_prompt,
        examples=val_examples,
        target_client=target_client,
        allow_code_execution=allow_code_execution,
        show_progress=show_progress,
        workers=workers,
        description="GAVEL validation gate/base",
        phase=_validation_phase("base", reference_mode),
        self_refine_rounds=self_refine_rounds,
        execution_mode=reference_mode,
    )
    del base_predictions

    base_summary = _split_result_summary(
        base_scores,
        examples=len(val_examples),
        api_calls=base_stats.api_calls,
        stats=base_stats,
    )
    base_summary["prompt_name"] = "base"
    base_summary["execution_mode"] = reference_mode
    candidate_summaries: Dict[str, Dict[str, Any]] = {}
    evaluated_candidates: List[tuple[EvaluationCandidate, Dict[str, Any], List[ScoreResult]]] = []
    api_calls = base_stats.api_calls
    prompt_options = [
        PromptCandidate(
            name="base",
            prompt=base_prompt,
            source="base",
            report_decision="base",
            report_accepted=True,
        ),
        *unique_candidates,
    ]
    for prompt_candidate in prompt_options:
        for execution_mode in resolved_execution_modes:
            eval_candidate = EvaluationCandidate(
                prompt_name=prompt_candidate.name,
                prompt=prompt_candidate.prompt,
                source=prompt_candidate.source,
                report_decision=prompt_candidate.report_decision,
                report_accepted=prompt_candidate.report_accepted,
                execution_mode=execution_mode,
            )
            if eval_candidate.prompt_name == "base" and execution_mode == reference_mode:
                continue
            candidate_predictions, candidate_scores, candidate_stats = await evaluate_prompt(
                prompt=eval_candidate.prompt,
                examples=val_examples,
                target_client=target_client,
                allow_code_execution=allow_code_execution,
                show_progress=show_progress,
                workers=workers,
                description=f"GAVEL validation gate/{eval_candidate.name}",
                phase=_validation_phase(eval_candidate.prompt_name, execution_mode),
                self_refine_rounds=self_refine_rounds,
                execution_mode=execution_mode,
            )
            del candidate_predictions
            candidate_summary = _split_result_summary(
                candidate_scores,
                examples=len(val_examples),
                api_calls=candidate_stats.api_calls,
                stats=candidate_stats,
            )
            paired = _paired_score_stats(base_scores, candidate_scores)
            required_margin = _effective_validation_margin(
                base_prompt=base_prompt,
                candidate=PromptCandidate(
                    name=eval_candidate.prompt_name,
                    prompt=eval_candidate.prompt,
                    source=eval_candidate.source,
                    report_decision=eval_candidate.report_decision,
                    report_accepted=eval_candidate.report_accepted,
                ),
                base_summary=base_summary,
                candidate_summary=candidate_summary,
                user_margin=margin,
                confidence_z=confidence_z,
                rejected_candidate_margin=rejected_candidate_margin,
                prompt_complexity_margin=prompt_complexity_margin,
            )
            candidate_summary["prompt_name"] = eval_candidate.prompt_name
            candidate_summary["execution_mode"] = execution_mode
            candidate_summary["candidate_source"] = eval_candidate.source
            candidate_summary["paired"] = paired
            candidate_summary["required_margin"] = required_margin
            candidate_summaries[eval_candidate.name] = candidate_summary
            if eval_candidate.prompt_name != "base" and execution_mode == reference_mode:
                candidate_summaries.setdefault(eval_candidate.prompt_name, candidate_summary)
            evaluated_candidates.append((eval_candidate, candidate_summary, candidate_scores))
            api_calls += candidate_stats.api_calls

    base_metric = _selection_metric(base_summary)
    scored_candidates = [
        (candidate, summary, metric, float(summary.get("required_margin") or margin))
        for candidate, summary, _scores in evaluated_candidates
        for metric in [_selection_metric(summary)]
        if metric is not None
    ]
    if base_metric is None or not scored_candidates:
        fallback = _preferred_candidate(unique_candidates)
        selected_variant = (
            fallback.name
            if fallback is not None and base_metric is None and fallback.report_accepted
            else "base"
        )
        return PromptSelection(
            prompt=fallback.prompt if selected_variant != "base" else base_prompt,
            selected_variant=selected_variant,
            execution_mode=fallback_execution_mode,
            decision=(
                fallback.report_decision
                if selected_variant != "base"
                else "validation_unscored_rollback"
            ),
            reason="validation scorer produced no comparable metric",
            api_calls=api_calls,
            base_summary=base_summary,
            candidate_summary=(
                candidate_summaries.get(_evaluation_candidate_key(fallback.name, fallback_execution_mode))
                or candidate_summaries.get(fallback.name)
                if fallback is not None
                else None
            ),
            candidate_summaries=candidate_summaries,
            margin=margin,
            candidate_source=fallback.source if fallback is not None else "base",
            effective_margin=margin,
        )

    best_candidate, best_summary, best_metric, best_required_margin = max(
        scored_candidates,
        key=lambda item: (
            item[2] - base_metric - item[3],
            item[2],
            1 if item[0].report_accepted else 0,
            _evaluation_candidate_priority(item[0]),
        ),
    )
    paired = best_summary.get("paired") or {}
    paired_ok = int(paired.get("wins") or 0) >= int(paired.get("losses") or 0)
    if best_metric + 1e-12 >= base_metric + best_required_margin and paired_ok:
        if best_candidate.prompt_name == "base":
            decision = "validation_selected_execution_mode"
        elif best_candidate.source.startswith("pareto:"):
            decision = best_candidate.report_decision
        else:
            decision = (
                best_candidate.report_decision
                if best_candidate.report_accepted
                else "validation_override"
            )
        return PromptSelection(
            prompt=best_candidate.prompt,
            selected_variant=best_candidate.prompt_name,
            execution_mode=best_candidate.execution_mode,
            decision=decision,
            reason=(
                f"{best_candidate.name} validation metric {best_metric:.4f} met "
                f"base {base_metric:.4f} with required margin {best_required_margin:.4f}"
            ),
            api_calls=api_calls,
            base_summary=base_summary,
            candidate_summary=best_summary,
            candidate_summaries=candidate_summaries,
            margin=margin,
            candidate_source=best_candidate.source,
            effective_margin=best_required_margin,
        )
    return PromptSelection(
        prompt=base_prompt,
        selected_variant="base",
        execution_mode=reference_mode,
        decision="validation_rollback",
        reason=(
            f"best candidate validation metric {best_metric:.4f} fell below "
            f"base {base_metric:.4f} with required margin {best_required_margin:.4f}"
        ),
        api_calls=api_calls,
        base_summary=base_summary,
        candidate_summary=best_summary,
        candidate_summaries=candidate_summaries,
        margin=margin,
        candidate_source=best_candidate.source,
        effective_margin=best_required_margin,
    )


def _build_prompt_candidates(
    *,
    base_prompt: str,
    report: Any,
    benchmark_spec: Optional[BenchmarkSpec],
    logs: Sequence[Mapping[str, Any]],
    pareto_candidates: Sequence[PromptCandidate],
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
        ),
        PromptCandidate(
            name="constraint_solver",
            prompt=constraint_solver_prompt(benchmark_spec),
            source="deterministic_constraint_solver",
            report_decision="validation_selected_constraint_solver",
        )
    ]
    candidates.extend(pareto_candidates)
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
    if candidate.name.startswith("pareto_") or candidate.source.startswith("pareto:"):
        return 4
    priorities = {
        "optimized": 3,
        "constraint_solver": 3,
        "evidence_strategy": 2,
        "task_strategy": 1,
    }
    return priorities.get(candidate.name, 0)


def _evaluation_candidate_priority(candidate: EvaluationCandidate) -> int:
    mode_priorities = {
        "plan_refine": 4,
        "plan": 3,
        "self_refine": 2,
        "direct": 1,
    }
    return _candidate_priority(
        PromptCandidate(
            name=candidate.prompt_name,
            prompt=candidate.prompt,
            source=candidate.source,
            report_decision=candidate.report_decision,
            report_accepted=candidate.report_accepted,
        )
    ) * 10 + mode_priorities.get(candidate.execution_mode, 0)


def _validation_phase(prompt_name: str, execution_mode: str) -> str:
    base = f"validation_gate_{prompt_name}"
    return base if execution_mode == "direct" else f"{base}_{execution_mode}"


def _paired_score_stats(
    base_scores: Sequence[ScoreResult],
    candidate_scores: Sequence[ScoreResult],
) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0
    comparable = 0
    for base, candidate in zip(base_scores, candidate_scores):
        if base.score is None or candidate.score is None:
            continue
        comparable += 1
        base_value = float(base.score)
        candidate_value = float(candidate.score)
        if candidate_value > base_value:
            wins += 1
        elif candidate_value < base_value:
            losses += 1
        else:
            ties += 1
    net = wins - losses
    return {
        "comparable": comparable,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_wins": net,
        "net_win_rate": None if comparable == 0 else net / comparable,
    }


def _effective_validation_margin(
    *,
    base_prompt: str,
    candidate: PromptCandidate,
    base_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    user_margin: float,
    confidence_z: float,
    rejected_candidate_margin: float,
    prompt_complexity_margin: float,
) -> float:
    base_metric = _selection_metric(base_summary)
    candidate_metric = _selection_metric(candidate_summary)
    statistical = 0.0
    scored = min(
        int(base_summary.get("scored") or 0),
        int(candidate_summary.get("scored") or 0),
    )
    if base_metric is not None and candidate_metric is not None and scored > 0:
        base_var = max(0.0, min(1.0, base_metric)) * max(0.0, 1.0 - min(1.0, base_metric))
        candidate_var = max(0.0, min(1.0, candidate_metric)) * max(0.0, 1.0 - min(1.0, candidate_metric))
        statistical = confidence_z * math.sqrt((base_var + candidate_var) / scored)
    source_penalty = rejected_candidate_margin if candidate.source == "rejected" else 0.0
    complexity_penalty = _prompt_complexity_penalty(
        base_prompt,
        candidate.prompt,
        prompt_complexity_margin=prompt_complexity_margin,
    )
    return min(1.0, max(user_margin, statistical) + source_penalty + complexity_penalty)


def _prompt_complexity_penalty(
    base_prompt: str,
    candidate_prompt: str,
    *,
    prompt_complexity_margin: float,
) -> float:
    if prompt_complexity_margin <= 0:
        return 0.0
    base_tokens = estimate_token_count(base_prompt)
    candidate_tokens = estimate_token_count(candidate_prompt)
    delta_tokens = max(0, candidate_tokens - base_tokens)
    if delta_tokens <= 0:
        return 0.0
    return min(0.12, prompt_complexity_margin * (delta_tokens / 1000.0))


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
    self_refine_rounds: int,
    execution_mode: str,
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
            self_refine_rounds=self_refine_rounds,
            execution_mode=execution_mode,
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
                api_calls=stats.api_calls,
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
