"""DSPy, MIPROv2, and GEPA benchmark baselines.

The integration follows the public API from the official DSPy GitHub repo:
`dspy.LM`, `dspy.configure`, `dspy.Predict` / `dspy.ChainOfThought`, and
`dspy.MIPROv2.compile(...)` / `dspy.GEPA.compile(...)`.
"""

from __future__ import annotations

import importlib
import json
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from pact_el.benchmarks.scoring import (
    ScoreAccumulator,
    load_normalized_examples,
    score_prediction,
    summarize_scores,
)
from pact_el.benchmarks.schemas import BenchmarkExample, ScoreResult


DSPY_GITHUB_URL = "https://github.com/stanfordnlp/dspy"
DSPY_MIPROV2_GITHUB_URL = (
    "https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md"
)
DSPY_GEPA_GITHUB_URL = (
    "https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md"
)


class MissingDSPyError(RuntimeError):
    """Raised when the optional DSPy baseline dependency is unavailable."""


@dataclass
class DSPyMIPROConfig:
    """Configuration for a DSPy direct, MIPROv2, or GEPA optimized run."""

    model: str = "openai/gpt-4o-mini"
    optimizer: str = "mipro"
    program: str = "cot"
    output_dir: Path = Path("output/baselines/dspy_mipro")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model_type: str = "chat"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    cache: bool = True
    lm_kwargs: Dict[str, Any] = field(default_factory=dict)
    auto: Optional[str] = "light"
    num_candidates: Optional[int] = None
    num_trials: Optional[int] = None
    num_threads: Optional[int] = None
    max_metric_calls: Optional[int] = None
    max_full_evals: Optional[int] = None
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    metric_threshold: Optional[float] = None
    minibatch: bool = True
    minibatch_size: int = 35
    minibatch_full_eval_steps: int = 5
    reflection_model: Optional[str] = None
    reflection_temperature: Optional[float] = None
    reflection_max_tokens: Optional[int] = None
    reflection_minibatch_size: int = 3
    seed: int = 9
    allow_code_execution: bool = False
    save_program: bool = True
    prediction_field: str = "answer"
    show_progress: bool = True
    workers: int = 1

    def validate(self) -> None:
        if self.optimizer not in {"dspy", "mipro", "gepa"}:
            raise ValueError("optimizer must be 'dspy', 'mipro', or 'gepa'")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.program not in {"predict", "cot", "chain_of_thought"}:
            raise ValueError("program must be 'predict', 'cot', or 'chain_of_thought'")
        if self.auto not in {None, "light", "medium", "heavy"}:
            raise ValueError("auto must be None, 'light', 'medium', or 'heavy'")
        if self.optimizer != "gepa" and (
            self.max_metric_calls is not None or self.max_full_evals is not None
        ):
            raise ValueError("--max-metric-calls and --max-full-evals are only used by GEPA")
        if self.optimizer == "mipro" and self.auto is not None and (
            self.num_candidates is not None or self.num_trials is not None
        ):
            raise ValueError(
                "DSPy MIPROv2 does not allow num_candidates/num_trials when auto is set"
            )
        if self.auto is None and self.optimizer == "mipro":
            if self.num_candidates is None or self.num_trials is None:
                raise ValueError(
                    "When auto is None for MIPROv2, both num_candidates and num_trials are required"
                )
        if self.optimizer == "gepa":
            budget_count = sum(
                value is not None
                for value in (self.auto, self.max_metric_calls, self.max_full_evals)
            )
            if budget_count != 1:
                raise ValueError(
                    "GEPA requires exactly one budget: --auto, --max-metric-calls, "
                    "or --max-full-evals"
                )
            if self.num_candidates is not None or self.num_trials is not None:
                raise ValueError(
                    "GEPA does not use --num-candidates or --num-trials; use --auto, "
                    "--max-metric-calls, or --max-full-evals"
                )
            if self.reflection_minibatch_size < 1:
                raise ValueError("reflection_minibatch_size must be at least 1")


@dataclass
class DSPyRunResult:
    """File paths and aggregate metrics from a baseline run."""

    summary: Dict[str, Any]
    predictions_path: Path
    scores_path: Path
    program_path: Optional[Path] = None


@dataclass
class EvaluationStats:
    """Concurrency stats captured during DSPy final evaluation."""

    requested_workers: int
    max_in_flight: int = 0


def load_examples(path: Path, limit: Optional[int] = None) -> List[BenchmarkExample]:
    examples = load_normalized_examples(path)
    return examples[:limit] if limit is not None else examples


def split_train_val(
    train_examples: Sequence[BenchmarkExample],
    val_examples: Optional[Sequence[BenchmarkExample]] = None,
    val_size: int = 50,
    seed: int = 9,
) -> tuple[List[BenchmarkExample], List[BenchmarkExample]]:
    if val_examples is not None:
        return list(train_examples), list(val_examples)
    if len(train_examples) < 2:
        raise ValueError("optimizer needs at least two examples to derive train/validation sets")
    shuffled = list(train_examples)
    random.Random(seed).shuffle(shuffled)
    effective_val_size = min(max(1, val_size), len(shuffled) - 1)
    return shuffled[effective_val_size:], shuffled[:effective_val_size]


def build_dspy_metric(
    allow_code_execution: bool = False,
    prediction_field: str = "answer",
) -> Callable[[Any, Any, Any], float]:
    """Build a DSPy metric that delegates scoring to the normalized benchmark scorer."""

    def metric(example: Any, pred: Any, trace: Any = None) -> float:
        del trace
        benchmark_example = _benchmark_example_from_dspy(example)
        prediction = _prediction_text(pred, field=prediction_field)
        result = score_prediction(
            benchmark_example,
            prediction,
            allow_code_execution=allow_code_execution,
        )
        return float(result.score or 0.0)

    return metric


class _ScoreWithFeedback(dict):
    """Small DSPy-compatible score object for GEPA feedback metrics."""

    @property
    def score(self) -> float:
        return float(self["score"])

    @property
    def feedback(self) -> str:
        return str(self["feedback"])

    def __float__(self) -> float:
        return self.score

    def __add__(self, other: Any) -> float:
        return self.score + _numeric_score(other)

    def __radd__(self, other: Any) -> float:
        return _numeric_score(other) + self.score

    def __truediv__(self, other: Any) -> float:
        return self.score / _numeric_score(other)

    def __lt__(self, other: Any) -> bool:
        return self.score < _numeric_score(other)

    def __le__(self, other: Any) -> bool:
        return self.score <= _numeric_score(other)

    def __gt__(self, other: Any) -> bool:
        return self.score > _numeric_score(other)

    def __ge__(self, other: Any) -> bool:
        return self.score >= _numeric_score(other)


def _numeric_score(value: Any) -> float:
    if hasattr(value, "score"):
        return float(value.score)
    return float(value)


def build_gepa_metric(
    allow_code_execution: bool = False,
    prediction_field: str = "answer",
) -> Callable[[Any, Any, Any, Any, Any], _ScoreWithFeedback]:
    """Build a GEPA metric with scalar score plus textual feedback."""

    def metric(
        example: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> _ScoreWithFeedback:
        del trace, pred_name, pred_trace
        benchmark_example = _benchmark_example_from_dspy(example)
        prediction = _prediction_text(pred, field=prediction_field)
        result = score_prediction(
            benchmark_example,
            prediction,
            allow_code_execution=allow_code_execution,
        )
        score = float(result.score or 0.0)
        return _ScoreWithFeedback(
            score=score,
            feedback=_format_gepa_feedback(benchmark_example, result),
        )

    return metric


def run_dspy_baseline(
    eval_examples: Sequence[BenchmarkExample],
    config: DSPyMIPROConfig,
    train_examples: Optional[Sequence[BenchmarkExample]] = None,
    val_examples: Optional[Sequence[BenchmarkExample]] = None,
    selection_summary: Optional[Mapping[str, Any]] = None,
) -> DSPyRunResult:
    """Run a direct DSPy program or optimize it with MIPROv2/GEPA, then score outputs."""

    config.validate()
    dspy = import_dspy()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lm = configure_lm(dspy, config)
    scalar_metric = build_dspy_metric(
        allow_code_execution=config.allow_code_execution,
        prediction_field=config.prediction_field,
    )
    program = build_program(dspy, config.program)

    train_split: List[BenchmarkExample] = []
    val_split: List[BenchmarkExample] = []
    optimization_api_calls: Optional[int] = 0 if config.optimizer == "dspy" else None
    dspy_train: List[Any] = []
    dspy_val: List[Any] = []
    if config.optimizer in {"mipro", "gepa"}:
        if not train_examples:
            raise ValueError(f"{_method_label(config)} requires --train-dataset examples")
        train_split, val_split = split_train_val(
            train_examples,
            val_examples=val_examples,
            seed=config.seed,
        )
        dspy_train = to_dspy_examples(dspy, train_split)
        dspy_val = to_dspy_examples(dspy, val_split)
        before_compile_calls = _lm_history_count(lm)
        reflection_lm = None
        before_reflection_calls: Optional[int] = None
        if config.optimizer == "mipro":
            program = compile_mipro(
                dspy,
                program,
                dspy_train,
                dspy_val,
                scalar_metric,
                config,
            )
        else:
            gepa_metric = build_gepa_metric(
                allow_code_execution=config.allow_code_execution,
                prediction_field=config.prediction_field,
            )
            reflection_lm = configure_reflection_lm(dspy, config)
            before_reflection_calls = _lm_history_count(reflection_lm)
            program = compile_gepa(
                dspy,
                program,
                dspy_train,
                dspy_val,
                gepa_metric,
                reflection_lm,
                config,
            )
        after_compile_calls = _lm_history_count(lm)
        after_reflection_calls = _lm_history_count(reflection_lm)
        optimization_api_calls = _sum_optional_counts(
            _count_delta(before_compile_calls, after_compile_calls),
            _count_delta(before_reflection_calls, after_reflection_calls),
        )

    run_name = _run_name(config)
    split_evaluations = _evaluate_report_splits(
        program=program,
        run_name=run_name,
        output_dir=config.output_dir,
        train_examples=train_split,
        val_examples=val_split,
        test_examples=eval_examples,
        allow_code_execution=config.allow_code_execution,
        prediction_field=config.prediction_field,
        show_progress=config.show_progress,
        workers=config.workers,
        lm=lm,
    )
    test_eval = split_evaluations["test"]
    predictions_path = Path(test_eval["predictions_path"])
    scores_path = Path(test_eval["scores_path"])
    scores = test_eval["scores"]
    split_results = {
        split_name: split_eval["summary"]
        for split_name, split_eval in split_evaluations.items()
    }
    split_results["optimization"] = {
        "score": None,
        "stddev": None,
        "api_calls": optimization_api_calls,
        "examples": 0,
        "scored": 0,
    }
    max_in_flight = max(
        int(row.get("max_in_flight") or 0)
        for row in split_results.values()
    )

    program_path: Optional[Path] = None
    program_save_error: Optional[str] = None
    if config.save_program and hasattr(program, "save"):
        program_path = config.output_dir / f"{run_name}.program.json"
        try:
            program.save(str(program_path))
        except Exception as exc:  # pragma: no cover - depends on optional DSPy serializers
            program_path = None
            program_save_error = str(exc)

    summary = {
        **summarize_scores(scores),
        "method": _method_label(config),
        "optimizer": config.optimizer,
        "program": config.program,
        "model": config.model,
        "reflection_model": _effective_reflection_model(config) if config.optimizer == "gepa" else None,
        "eval_examples": len(eval_examples),
        "test_examples": len(eval_examples),
        "train_examples": len(dspy_train),
        "val_examples": len(dspy_val),
        "workers": config.workers,
        "max_in_flight": max_in_flight,
        "cache": config.cache,
        "num_threads": _effective_num_threads(config),
        "max_metric_calls": config.max_metric_calls,
        "max_full_evals": config.max_full_evals,
        "predictions_path": str(predictions_path),
        "scores_path": str(scores_path),
        "program_path": str(program_path) if program_path else None,
        "program_save_error": program_save_error,
        "split_results": split_results,
        "split_paths": {
            split_name: {
                "predictions_path": split_eval["predictions_path"],
                "scores_path": split_eval["scores_path"],
            }
            for split_name, split_eval in split_evaluations.items()
        },
        "official_sources": {
            "dspy": DSPY_GITHUB_URL,
            "mipro_v2": DSPY_MIPROV2_GITHUB_URL,
            "gepa": DSPY_GEPA_GITHUB_URL,
        },
    }
    if selection_summary is not None:
        summary["selection"] = dict(selection_summary)
    summary_path = config.output_dir / f"{run_name}.summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return DSPyRunResult(
        summary=summary,
        predictions_path=predictions_path,
        scores_path=scores_path,
        program_path=program_path,
    )


def import_dspy() -> Any:
    try:
        return importlib.import_module("dspy")
    except ModuleNotFoundError as exc:
        raise MissingDSPyError(
            "DSPy is optional. Install it on Python 3.10+ with "
            "`python -m pip install -e '.[baselines]'`. The baselines extra "
            "includes DSPy's optuna support for MIPROv2 and DSPy's GEPA dependency."
        ) from exc


def configure_lm(dspy: Any, config: DSPyMIPROConfig) -> Any:
    kwargs = dict(config.lm_kwargs)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    lm = dspy.LM(
        config.model,
        model_type=config.model_type,
        cache=config.cache,
        **kwargs,
    )
    dspy.configure(lm=lm)
    return lm


def configure_reflection_lm(dspy: Any, config: DSPyMIPROConfig) -> Any:
    kwargs = dict(config.lm_kwargs)
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.reflection_temperature is not None:
        kwargs["temperature"] = config.reflection_temperature
    elif config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.reflection_max_tokens is not None:
        kwargs["max_tokens"] = config.reflection_max_tokens
    elif config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    return dspy.LM(
        _effective_reflection_model(config),
        model_type=config.model_type,
        cache=config.cache,
        **kwargs,
    )


def build_program(dspy: Any, program: str) -> Any:
    signature = "question -> answer"
    if program == "predict":
        return dspy.Predict(signature)
    if program in {"cot", "chain_of_thought"}:
        return dspy.ChainOfThought(signature)
    raise ValueError(f"unsupported DSPy program: {program}")


def compile_mipro(
    dspy: Any,
    program: Any,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    metric: Callable[[Any, Any, Any], float],
    config: DSPyMIPROConfig,
) -> Any:
    mipro_kwargs: Dict[str, Any] = {
        "metric": metric,
        "auto": config.auto,
        "num_threads": _effective_num_threads(config),
        "max_bootstrapped_demos": config.max_bootstrapped_demos,
        "max_labeled_demos": config.max_labeled_demos,
        "seed": config.seed,
        "metric_threshold": config.metric_threshold,
    }
    if config.num_candidates is not None:
        mipro_kwargs["num_candidates"] = config.num_candidates
    teleprompter = dspy.MIPROv2(**mipro_kwargs)

    compile_kwargs: Dict[str, Any] = {
        "trainset": list(trainset),
        "valset": list(valset),
        "seed": config.seed,
        "minibatch": config.minibatch,
        "minibatch_size": config.minibatch_size,
        "minibatch_full_eval_steps": config.minibatch_full_eval_steps,
    }
    if config.num_trials is not None:
        compile_kwargs["num_trials"] = config.num_trials
    return teleprompter.compile(program, **compile_kwargs)


def compile_gepa(
    dspy: Any,
    program: Any,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    metric: Callable[[Any, Any, Any, Any, Any], Any],
    reflection_lm: Any,
    config: DSPyMIPROConfig,
) -> Any:
    gepa_kwargs: Dict[str, Any] = {
        "metric": metric,
        "reflection_lm": reflection_lm,
        "num_threads": _effective_num_threads(config),
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "track_stats": True,
        "seed": config.seed,
        "log_dir": str(config.output_dir / "gepa_logs"),
    }
    if config.auto is not None:
        gepa_kwargs["auto"] = config.auto
    if config.max_metric_calls is not None:
        gepa_kwargs["max_metric_calls"] = config.max_metric_calls
    if config.max_full_evals is not None:
        gepa_kwargs["max_full_evals"] = config.max_full_evals
    teleprompter = dspy.GEPA(**gepa_kwargs)
    return teleprompter.compile(
        program,
        trainset=list(trainset),
        valset=list(valset),
    )


def to_dspy_examples(dspy: Any, examples: Sequence[BenchmarkExample]) -> List[Any]:
    return [_to_dspy_example(dspy, example) for example in examples]


def _evaluate_report_splits(
    *,
    program: Any,
    run_name: str,
    output_dir: Path,
    train_examples: Sequence[BenchmarkExample],
    val_examples: Sequence[BenchmarkExample],
    test_examples: Sequence[BenchmarkExample],
    allow_code_execution: bool,
    prediction_field: str,
    show_progress: bool,
    workers: int,
    lm: Any,
) -> Dict[str, Dict[str, Any]]:
    split_examples = {
        "train": list(train_examples),
        "val": list(val_examples),
        "test": list(test_examples),
    }
    evaluations: Dict[str, Dict[str, Any]] = {}
    for split_name, examples in split_examples.items():
        before_calls = _lm_history_count(lm)
        predictions, scores, stats = _evaluate_program_with_stats(
            program,
            examples,
            allow_code_execution=allow_code_execution,
            prediction_field=prediction_field,
            show_progress=show_progress and bool(examples),
            workers=workers,
            description=f"DSPy {split_name}",
        )
        after_calls = _lm_history_count(lm)
        api_calls = _count_delta(before_calls, after_calls)
        api_calls = max(api_calls or 0, len(examples))
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
            "summary": _split_result_summary(scores, examples=len(examples), api_calls=api_calls),
        }
        evaluations[split_name]["summary"]["workers"] = stats.requested_workers
        evaluations[split_name]["summary"]["max_in_flight"] = stats.max_in_flight
    return evaluations


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
    }


def _sample_stddev(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.stdev(values)


def evaluate_program(
    program: Any,
    examples: Sequence[BenchmarkExample],
    allow_code_execution: bool = False,
    prediction_field: str = "answer",
    show_progress: bool = False,
    workers: int = 1,
    description: str = "DSPy eval",
) -> tuple[List[Dict[str, Any]], List[ScoreResult]]:
    predictions, scores, _stats = _evaluate_program_with_stats(
        program,
        examples,
        allow_code_execution=allow_code_execution,
        prediction_field=prediction_field,
        show_progress=show_progress,
        workers=workers,
        description=description,
    )
    return predictions, scores


def _evaluate_program_with_stats(
    program: Any,
    examples: Sequence[BenchmarkExample],
    allow_code_execution: bool = False,
    prediction_field: str = "answer",
    show_progress: bool = False,
    workers: int = 1,
    description: str = "DSPy eval",
) -> tuple[List[Dict[str, Any]], List[ScoreResult], EvaluationStats]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    predictions: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    scores: List[Optional[ScoreResult]] = [None] * len(examples)
    progress = tqdm(
        total=len(examples),
        desc=description,
        unit="ex",
        colour="magenta",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    tracker = ScoreAccumulator()
    stats = EvaluationStats(requested_workers=workers)
    active = 0
    active_lock = threading.Lock()

    def evaluate_tracked(index: int, example: BenchmarkExample) -> Tuple[int, Dict[str, Any], ScoreResult]:
        nonlocal active
        with active_lock:
            active += 1
            stats.max_in_flight = max(stats.max_in_flight, active)
        try:
            return _evaluate_one_program(
                index,
                program,
                example,
                allow_code_execution=allow_code_execution,
                prediction_field=prediction_field,
            )
        finally:
            with active_lock:
                active -= 1

    def active_count() -> int:
        with active_lock:
            return active

    with progress:
        progress.set_postfix(
            workers=workers,
            in_flight=0,
            max_in_flight=0,
            refresh=False,
        )
        if workers == 1:
            for index, example in enumerate(examples):
                row, score = evaluate_tracked(index, example)
                _record_evaluation_result(
                    index,
                    row,
                    score,
                    predictions,
                    scores,
                    tracker,
                    progress,
                    workers=workers,
                    in_flight=active_count(),
                    max_in_flight=stats.max_in_flight,
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        evaluate_tracked,
                        index,
                        example,
                    )
                    for index, example in enumerate(examples)
                ]
                for future in as_completed(futures):
                    index, row, score = future.result()
                    _record_evaluation_result(
                        index,
                        row,
                        score,
                        predictions,
                        scores,
                        tracker,
                        progress,
                        workers=workers,
                        in_flight=active_count(),
                        max_in_flight=stats.max_in_flight,
                    )
    return (
        [prediction for prediction in predictions if prediction is not None],
        [score for score in scores if score is not None],
        stats,
    )


def _evaluate_one_program(
    index: int,
    program: Any,
    example: BenchmarkExample,
    allow_code_execution: bool,
    prediction_field: str,
) -> Tuple[int, Dict[str, Any], ScoreResult]:
    raw_prediction = program(question=example.prompt)
    prediction = _prediction_text(raw_prediction, field=prediction_field)
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
            "raw_prediction": _json_safe_prediction(raw_prediction),
        },
        score,
    )


def _record_evaluation_result(
    index: int,
    prediction: Dict[str, Any],
    score: ScoreResult,
    predictions: List[Optional[Dict[str, Any]]],
    scores: List[Optional[ScoreResult]],
    tracker: ScoreAccumulator,
    progress: Any,
    workers: Optional[int] = None,
    in_flight: Optional[int] = None,
    max_in_flight: Optional[int] = None,
) -> None:
    predictions[index] = prediction
    scores[index] = score
    if tracker.add(score):
        postfix = tracker.progress_postfix()
        if workers is not None:
            postfix["workers"] = workers
        if in_flight is not None:
            postfix["in_flight"] = in_flight
        if max_in_flight is not None:
            postfix["max_in_flight"] = max_in_flight
        progress.set_postfix(**postfix, refresh=False)
    progress.update(1)


def _to_dspy_example(dspy: Any, example: BenchmarkExample) -> Any:
    payload = {
        "question": example.prompt,
        "answer": example.expected_answer,
        "benchmark_example": example.model_dump(mode="json"),
        "example_id": example.example_id,
        "benchmark_id": example.benchmark_id,
    }
    if example.choices:
        payload["choices"] = list(example.choices)
    return dspy.Example(**payload).with_inputs("question")


def _benchmark_example_from_dspy(example: Any) -> BenchmarkExample:
    raw = _get_field(example, "benchmark_example")
    if raw is None:
        raise ValueError("DSPy example is missing benchmark_example metadata")
    if isinstance(raw, BenchmarkExample):
        return raw
    return BenchmarkExample.model_validate(raw)


def _prediction_text(prediction: Any, field: str = "answer") -> str:
    value = _get_field(prediction, field)
    if value is None and isinstance(prediction, Mapping):
        value = prediction.get("prediction") or prediction.get("output")
    if value is None:
        value = prediction
    return str(value)


def _get_field(obj: Any, field: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(field)
    if hasattr(obj, field):
        return getattr(obj, field)
    try:
        return obj[field]
    except (KeyError, TypeError):
        return None


def _json_safe_prediction(prediction: Any) -> Any:
    if isinstance(prediction, Mapping):
        return dict(prediction)
    if hasattr(prediction, "toDict"):
        return prediction.toDict()
    if hasattr(prediction, "__dict__"):
        return {
            key: value
            for key, value in vars(prediction).items()
            if not key.startswith("_")
        }
    return str(prediction)


def _lm_history_count(lm: Any) -> Optional[int]:
    if lm is None:
        return None
    for attr in ("history", "_history", "request_history"):
        history = getattr(lm, attr, None)
        length = _safe_len(history)
        if length is not None:
            return length
    for method_name in ("get_history", "inspect_history"):
        method = getattr(lm, method_name, None)
        if not callable(method):
            continue
        try:
            history = method()
        except TypeError:
            continue
        length = _safe_len(history)
        if length is not None:
            return length
    return None


def _safe_len(value: Any) -> Optional[int]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _count_delta(before: Optional[int], after: Optional[int]) -> Optional[int]:
    if before is None or after is None:
        return None
    return max(0, after - before)


def _sum_optional_counts(*values: Optional[int]) -> Optional[int]:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(known)


def _effective_num_threads(config: DSPyMIPROConfig) -> Optional[int]:
    if config.optimizer not in {"mipro", "gepa"}:
        return config.num_threads
    return config.num_threads if config.num_threads is not None else config.workers


def _effective_reflection_model(config: DSPyMIPROConfig) -> str:
    return config.reflection_model or config.model


def _method_label(config: DSPyMIPROConfig) -> str:
    if config.optimizer == "mipro":
        return "miprov2"
    return config.optimizer


def _run_name(config: DSPyMIPROConfig) -> str:
    model = config.model.replace("/", "_").replace(":", "_")
    return f"{config.optimizer}-{config.program}-{model}"


def _format_gepa_feedback(example: BenchmarkExample, result: ScoreResult) -> str:
    status = "unscored"
    if result.passed is True:
        status = "passed"
    elif result.passed is False:
        status = "failed"
    details = _compact_json(result.details)
    return (
        f"Benchmark: {example.benchmark_id}\n"
        f"Metric: {result.metric.value}\n"
        f"Status: {status}\n"
        f"Score: {0.0 if result.score is None else float(result.score):.3f}\n"
        f"Expected: {_truncate_text(result.expected)}\n"
        f"Prediction: {_truncate_text(result.prediction)}\n"
        f"Scoring details: {_truncate_text(details, limit=1200)}"
    )


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _truncate_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
