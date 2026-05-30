"""DSPy and MIPROv2 benchmark baselines.

The integration follows the public API from the official DSPy GitHub repo:
`dspy.LM`, `dspy.configure`, `dspy.Predict` / `dspy.ChainOfThought`, and
`dspy.MIPROv2.compile(...)`.
"""

from __future__ import annotations

import importlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from pact_el.benchmarks.scoring import (
    load_normalized_examples,
    score_prediction,
    summarize_scores,
)
from pact_el.benchmarks.schemas import BenchmarkExample, ScoreResult


DSPY_GITHUB_URL = "https://github.com/stanfordnlp/dspy"
DSPY_MIPROV2_GITHUB_URL = (
    "https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md"
)


class MissingDSPyError(RuntimeError):
    """Raised when the optional DSPy baseline dependency is unavailable."""


@dataclass
class DSPyMIPROConfig:
    """Configuration for a DSPy direct or MIPROv2 optimized run."""

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
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    metric_threshold: Optional[float] = None
    minibatch: bool = True
    minibatch_size: int = 35
    minibatch_full_eval_steps: int = 5
    seed: int = 9
    allow_code_execution: bool = False
    save_program: bool = True
    prediction_field: str = "answer"
    show_progress: bool = True

    def validate(self) -> None:
        if self.optimizer not in {"dspy", "mipro"}:
            raise ValueError("optimizer must be either 'dspy' or 'mipro'")
        if self.program not in {"predict", "cot", "chain_of_thought"}:
            raise ValueError("program must be 'predict', 'cot', or 'chain_of_thought'")
        if self.auto not in {None, "light", "medium", "heavy"}:
            raise ValueError("auto must be None, 'light', 'medium', or 'heavy'")
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


@dataclass
class DSPyRunResult:
    """File paths and aggregate metrics from a baseline run."""

    summary: Dict[str, Any]
    predictions_path: Path
    scores_path: Path
    program_path: Optional[Path] = None


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
        raise ValueError("MIPROv2 needs at least two examples to derive train/validation sets")
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


def run_dspy_baseline(
    eval_examples: Sequence[BenchmarkExample],
    config: DSPyMIPROConfig,
    train_examples: Optional[Sequence[BenchmarkExample]] = None,
    val_examples: Optional[Sequence[BenchmarkExample]] = None,
    selection_summary: Optional[Mapping[str, Any]] = None,
) -> DSPyRunResult:
    """Run a direct DSPy program or optimize it with MIPROv2, then score outputs."""

    config.validate()
    dspy = import_dspy()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    configure_lm(dspy, config)
    metric = build_dspy_metric(
        allow_code_execution=config.allow_code_execution,
        prediction_field=config.prediction_field,
    )
    program = build_program(dspy, config.program)

    dspy_train: List[Any] = []
    dspy_val: List[Any] = []
    if config.optimizer == "mipro":
        if not train_examples:
            raise ValueError("MIPROv2 requires --train-dataset examples")
        train_split, val_split = split_train_val(
            train_examples,
            val_examples=val_examples,
            seed=config.seed,
        )
        dspy_train = to_dspy_examples(dspy, train_split)
        dspy_val = to_dspy_examples(dspy, val_split)
        program = compile_mipro(dspy, program, dspy_train, dspy_val, metric, config)

    predictions, scores = evaluate_program(
        program,
        eval_examples,
        allow_code_execution=config.allow_code_execution,
        prediction_field=config.prediction_field,
        show_progress=config.show_progress,
    )

    run_name = _run_name(config)
    predictions_path = config.output_dir / f"{run_name}.predictions.jsonl"
    scores_path = config.output_dir / f"{run_name}.scores.jsonl"
    predictions_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in predictions)
        + ("\n" if predictions else "")
    )
    scores_path.write_text(
        "\n".join(score.model_dump_json() for score in scores) + ("\n" if scores else "")
    )

    program_path: Optional[Path] = None
    if config.save_program and hasattr(program, "save"):
        program_path = config.output_dir / f"{run_name}.program.json"
        program.save(str(program_path))

    summary = {
        **summarize_scores(scores),
        "optimizer": config.optimizer,
        "program": config.program,
        "model": config.model,
        "eval_examples": len(eval_examples),
        "test_examples": len(eval_examples),
        "train_examples": len(dspy_train),
        "val_examples": len(dspy_val),
        "predictions_path": str(predictions_path),
        "scores_path": str(scores_path),
        "program_path": str(program_path) if program_path else None,
        "official_sources": {
            "dspy": DSPY_GITHUB_URL,
            "mipro_v2": DSPY_MIPROV2_GITHUB_URL,
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
            "`python -m pip install -e '.[baselines]'`. MIPROv2 also needs "
            "DSPy's optuna extra, included by this package extra."
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
        "num_threads": config.num_threads,
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


def to_dspy_examples(dspy: Any, examples: Sequence[BenchmarkExample]) -> List[Any]:
    return [_to_dspy_example(dspy, example) for example in examples]


def evaluate_program(
    program: Any,
    examples: Sequence[BenchmarkExample],
    allow_code_execution: bool = False,
    prediction_field: str = "answer",
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], List[ScoreResult]]:
    predictions: List[Dict[str, Any]] = []
    scores: List[ScoreResult] = []
    progress = tqdm(
        total=len(examples),
        desc="DSPy eval",
        unit="ex",
        colour="magenta",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    scored = 0
    passed = 0
    with progress:
        for example in examples:
            raw_prediction = program(question=example.prompt)
            prediction = _prediction_text(raw_prediction, field=prediction_field)
            score = score_prediction(
                example,
                prediction,
                allow_code_execution=allow_code_execution,
            )
            predictions.append(
                {
                    "example_id": example.example_id,
                    "benchmark_id": example.benchmark_id,
                    "prediction": prediction,
                    "raw_prediction": _json_safe_prediction(raw_prediction),
                }
            )
            scores.append(score)
            if score.score is not None:
                scored += 1
                if score.passed is True:
                    passed += 1
                progress.set_postfix(
                    scored=scored,
                    passed=passed,
                    accuracy=f"{passed / scored:.1%}",
                    refresh=False,
                )
            progress.update(1)
    return predictions, scores


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


def _run_name(config: DSPyMIPROConfig) -> str:
    model = config.model.replace("/", "_").replace(":", "_")
    return f"{config.optimizer}-{config.program}-{model}"
