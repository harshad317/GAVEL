"""Local first-pass scoring for normalized benchmark examples."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import copy
import importlib
import importlib.metadata as importlib_metadata
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind, ScoreResult

IFBENCH_EVALUATOR_MISSING_REASON = "official IFBench evaluator is not installed"
IFBENCH_INSTALL_COMMAND = "python -m pip install -e '.[ifbench]'"
IFBENCH_EVALUATOR_MODULES = ("evaluation_lib", "ifbench.evaluation_lib")


class MissingIFBenchEvaluatorError(ModuleNotFoundError):
    """Raised when none of the known IFBench evaluator import paths are available."""

    def __init__(self, attempted_modules: Sequence[Mapping[str, str]]) -> None:
        self.attempted_modules = [dict(attempt) for attempt in attempted_modules]
        missing_module = (
            self.attempted_modules[0].get("missing_module")
            if self.attempted_modules
            else "evaluation_lib"
        )
        super().__init__(
            "No known IFBench evaluator module could be imported",
            name=missing_module,
        )


@dataclass
class ScoreAccumulator:
    """Shared aggregate tracker for all benchmark evaluation methods."""

    total: int = 0
    scored: int = 0
    passed: int = 0
    score_total: float = 0.0

    def add(self, result: ScoreResult) -> bool:
        self.total += 1
        if result.score is None:
            return False
        self.scored += 1
        self.score_total += float(result.score)
        if result.passed is True:
            self.passed += 1
        return True

    @property
    def unscored(self) -> int:
        return self.total - self.scored

    @property
    def accuracy(self) -> Optional[float]:
        return self.passed / self.scored if self.scored else None

    @property
    def mean_score(self) -> Optional[float]:
        return self.score_total / self.scored if self.scored else None

    def progress_postfix(self) -> Dict[str, Any]:
        return {
            "acc": format_accuracy(self.accuracy, precision=1),
            "mean": format_mean_score(self.mean_score),
            "passed": self.passed,
            "scored": self.scored,
            "unscored": self.unscored,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "scored": self.scored,
            "unscored": self.unscored,
            "passed": self.passed,
            "accuracy": self.accuracy,
            "mean_score": self.mean_score,
        }


def format_accuracy(value: Optional[float], precision: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{precision}%}"


def format_mean_score(value: Optional[float], precision: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def _validate_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return workers


def score_prediction(
    example: BenchmarkExample,
    prediction: Any,
    allow_code_execution: bool = False,
    timeout_seconds: float = 5.0,
) -> ScoreResult:
    if example.metric == MetricKind.NUMERIC_EXACT:
        return _score_numeric_exact(example, prediction)
    if example.metric == MetricKind.MULTIPLE_CHOICE_ACCURACY:
        return _score_multiple_choice(example, prediction)
    if example.metric == MetricKind.EXACT_MATCH:
        return _score_exact_match(example, prediction)
    if example.metric == MetricKind.TOKEN_F1:
        return _score_token_f1(example, prediction)
    if example.metric == MetricKind.PASS_AT_1:
        return _score_pass_at_1(
            example,
            prediction,
            allow_code_execution=allow_code_execution,
            timeout_seconds=timeout_seconds,
        )
    if (
        example.metric == MetricKind.OFFICIAL_EVALUATOR
        and example.benchmark_id == "ifbench"
    ):
        return _score_ifbench_official(example, prediction)
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=None,
        passed=None,
        prediction=prediction,
        expected=example.expected_answer,
        details={
            "reason": "this benchmark requires its official evaluator",
            "official_metric": example.metric.value,
        },
    )


def score_predictions(
    examples: Sequence[BenchmarkExample],
    predictions: Mapping[str, Any],
    allow_code_execution: bool = False,
    show_progress: bool = False,
    description: str = "Scoring",
    workers: int = 1,
) -> List[ScoreResult]:
    workers = _validate_workers(workers)
    results: List[Optional[ScoreResult]] = [None] * len(examples)
    progress = tqdm(
        total=len(examples),
        desc=description,
        unit="ex",
        colour="green",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    tracker = ScoreAccumulator()
    with progress:
        if workers == 1:
            for index, example in enumerate(examples):
                result = score_prediction(
                    example,
                    predictions.get(example.example_id),
                    allow_code_execution=allow_code_execution,
                )
                _record_score_result(index, result, results, tracker, progress)
        else:
            tasks = [
                (
                    index,
                    example,
                    predictions.get(example.example_id),
                    allow_code_execution,
                )
                for index, example in enumerate(examples)
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_score_prediction_task, task) for task in tasks]
                for future in as_completed(futures):
                    index, result = future.result()
                    _record_score_result(index, result, results, tracker, progress)
    return [result for result in results if result is not None]


def summarize_scores(results: Sequence[ScoreResult]) -> Dict[str, Any]:
    tracker = ScoreAccumulator()
    for result in results:
        tracker.add(result)
    return tracker.summary()


def summarize_unscored_reasons(results: Sequence[ScoreResult]) -> List[Dict[str, Any]]:
    reasons: Counter[str] = Counter()
    details_by_reason: Dict[str, Dict[str, Any]] = {}
    for result in results:
        if result.score is not None:
            continue
        reason = str(result.details.get("reason") or "score unavailable")
        reasons[reason] += 1
        details = details_by_reason.setdefault(reason, {})
        for key in ("install", "missing_module", "evaluator"):
            if key in result.details and key not in details:
                details[key] = result.details[key]
    return [
        {"reason": reason, "count": count, **details_by_reason.get(reason, {})}
        for reason, count in reasons.most_common()
    ]


def _record_score_result(
    index: int,
    result: ScoreResult,
    results: List[Optional[ScoreResult]],
    tracker: ScoreAccumulator,
    progress: Any,
) -> None:
    results[index] = result
    tracker.add(result)
    progress.set_postfix(**tracker.progress_postfix(), refresh=False)
    progress.update(1)


def _score_prediction_task(
    task: Tuple[int, BenchmarkExample, Any, bool],
) -> Tuple[int, ScoreResult]:
    index, example, prediction, allow_code_execution = task
    return index, score_prediction(
        example,
        prediction,
        allow_code_execution=allow_code_execution,
    )


def load_normalized_examples(path: Path) -> List[BenchmarkExample]:
    examples: List[BenchmarkExample] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                examples.append(BenchmarkExample.model_validate_json(stripped))
    return examples


def load_predictions(path: Path) -> Dict[str, Any]:
    payload = path.read_text().strip()
    if not payload:
        return {}
    if path.suffix == ".json":
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return _predictions_from_rows(data)
    return _predictions_from_rows(json.loads(line) for line in payload.splitlines() if line.strip())


def _predictions_from_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}
    for row in rows:
        example_id = row.get("example_id") or row.get("id") or row.get("key")
        if example_id is None:
            continue
        prediction = row.get("prediction", row.get("answer", row.get("output")))
        predictions[str(example_id)] = prediction
    return predictions


def _score_numeric_exact(example: BenchmarkExample, prediction: Any) -> ScoreResult:
    expected = _normalize_number(str(example.expected_answer))
    observed = _normalize_number(_extract_answer_text(str(prediction)))
    passed = expected is not None and observed is not None and expected == observed
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=1.0 if passed else 0.0,
        passed=passed,
        prediction=prediction,
        expected=example.expected_answer,
        details={"observed_number": observed, "expected_number": expected},
    )


def _score_multiple_choice(example: BenchmarkExample, prediction: Any) -> ScoreResult:
    observed = extract_choice(str(prediction), example.choices)
    expected = str(example.expected_answer).strip().upper()
    passed = observed == expected
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=1.0 if passed else 0.0,
        passed=passed,
        prediction=prediction,
        expected=example.expected_answer,
        details={"observed_choice": observed, "expected_choice": expected},
    )


def _score_exact_match(example: BenchmarkExample, prediction: Any) -> ScoreResult:
    aliases = _answer_aliases(example)
    observed = _normalize_text(_extract_answer_text(str(prediction)))
    passed = observed in {_normalize_text(alias) for alias in aliases}
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=1.0 if passed else 0.0,
        passed=passed,
        prediction=prediction,
        expected=example.expected_answer,
        details={"aliases": aliases},
    )


def _score_token_f1(example: BenchmarkExample, prediction: Any) -> ScoreResult:
    aliases = _answer_aliases(example)
    observed = _extract_answer_text(str(prediction))
    scores = [_token_f1(observed, alias) for alias in aliases] or [0.0]
    score = max(scores)
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=score,
        passed=math.isclose(score, 1.0),
        prediction=prediction,
        expected=example.expected_answer,
        details={"aliases": aliases, "best_f1": score},
    )


def _score_pass_at_1(
    example: BenchmarkExample,
    prediction: Any,
    allow_code_execution: bool,
    timeout_seconds: float,
) -> ScoreResult:
    if not allow_code_execution:
        return ScoreResult(
            example_id=example.example_id,
            metric=example.metric,
            score=None,
            passed=None,
            prediction=prediction,
            expected=example.expected_answer,
            details={
                "reason": "MBPP scoring requires executing generated code; rerun with allow_code_execution"
            },
        )
    tests = list(example.metadata.get("test_imports", [])) + list(
        example.metadata.get("test_list", [])
    )
    code = _strip_markdown_fences(str(prediction))
    with tempfile.TemporaryDirectory(prefix="pact_el_mbpp_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code + "\n\n" + "\n".join(tests) + "\n")
        try:
            completed = subprocess.run(
                ["python3", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ScoreResult(
                example_id=example.example_id,
                metric=example.metric,
                score=0.0,
                passed=False,
                prediction=prediction,
                expected=example.expected_answer,
                details={"reason": "timeout", "timeout_seconds": timeout_seconds, "stderr": str(exc)},
            )
    passed = completed.returncode == 0
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=1.0 if passed else 0.0,
        passed=passed,
        prediction=prediction,
        expected=example.expected_answer,
        details={
            "returncode": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        },
    )


def _score_ifbench_official(example: BenchmarkExample, prediction: Any) -> ScoreResult:
    row = example.metadata.get("official_input_row") or {}
    if not row:
        return ScoreResult(
            example_id=example.example_id,
            metric=example.metric,
            score=None,
            passed=None,
            prediction=prediction,
            expected=example.expected_answer,
            details={"reason": "IFBench example is missing official_input_row metadata"},
        )

    try:
        evaluation_lib = _import_ifbench_evaluation_lib()
    except ModuleNotFoundError as exc:
        return ScoreResult(
            example_id=example.example_id,
            metric=example.metric,
            score=None,
            passed=None,
            prediction=prediction,
            expected=example.expected_answer,
            details=_missing_ifbench_evaluator_details(exc),
        )

    inp = evaluation_lib.InputExample(
        key=row.get("key"),
        instruction_id_list=list(row.get("instruction_id_list") or []),
        prompt=str(row.get("prompt", example.prompt)),
        kwargs=copy.deepcopy(list(row.get("kwargs") or [])),
    )
    output = evaluation_lib.test_instruction_following_loose(
        inp,
        {inp.prompt: "" if prediction is None else str(prediction)},
    )
    passed = bool(output.follow_all_instructions)
    return ScoreResult(
        example_id=example.example_id,
        metric=example.metric,
        score=1.0 if passed else 0.0,
        passed=passed,
        prediction=prediction,
        expected=example.expected_answer,
        details={
            "evaluator": "allenai/IFBench test_instruction_following_loose",
            "instruction_id_list": list(output.instruction_id_list),
            "follow_instruction_list": list(output.follow_instruction_list),
        },
    )


def _import_ifbench_evaluation_lib() -> Any:
    attempted_modules: List[Dict[str, str]] = []
    for module_name in IFBENCH_EVALUATOR_MODULES:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if not _missing_requested_module(exc, module_name):
                raise
            attempted_modules.append(
                {"module": module_name, "missing_module": str(exc.name or module_name)}
            )

    distribution_module = _import_ifbench_evaluator_from_distribution()
    if distribution_module is not None:
        return distribution_module

    raise MissingIFBenchEvaluatorError(attempted_modules)


def ifbench_evaluator_missing_details() -> Optional[Dict[str, Any]]:
    try:
        _import_ifbench_evaluation_lib()
    except ModuleNotFoundError as exc:
        return _missing_ifbench_evaluator_details(exc)
    return None


def _missing_ifbench_evaluator_details(exc: ModuleNotFoundError) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "reason": IFBENCH_EVALUATOR_MISSING_REASON,
        "install": IFBENCH_INSTALL_COMMAND,
        "missing_module": exc.name,
    }
    attempted_modules = getattr(exc, "attempted_modules", None)
    if attempted_modules:
        details["attempted_modules"] = attempted_modules
    return details


def _missing_requested_module(exc: ModuleNotFoundError, module_name: str) -> bool:
    missing_name = exc.name
    return bool(
        missing_name
        and (missing_name == module_name or module_name.startswith(f"{missing_name}."))
    )


def _import_ifbench_evaluator_from_distribution() -> Optional[Any]:
    try:
        distribution = importlib_metadata.distribution("ifbench")
    except importlib_metadata.PackageNotFoundError:
        return None

    for file in distribution.files or ():
        if Path(str(file)).name != "evaluation_lib.py":
            continue
        evaluator_path = Path(distribution.locate_file(file))
        evaluator_dir = str(evaluator_path.parent)
        if evaluator_dir not in sys.path:
            sys.path.insert(0, evaluator_dir)
        return importlib.import_module("evaluation_lib")
    return None


def extract_choice(prediction: str, choices: Sequence[str]) -> Optional[str]:
    text = prediction.strip()
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?",
        r"^\(?([A-J])\)?(?:[\.:\s]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    normalized = _normalize_text(text)
    for index, choice in enumerate(choices):
        if normalized == _normalize_text(choice):
            return chr(ord("A") + index)
    return None


def _answer_aliases(example: BenchmarkExample) -> List[str]:
    aliases = example.metadata.get("answer_aliases") or example.metadata.get("correct_answers")
    if aliases:
        return [str(alias) for alias in aliases]
    return [str(example.expected_answer)]


def _extract_answer_text(text: str) -> str:
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()
    match = re.search(r"answer\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _normalize_number(text: str) -> Optional[str]:
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not numbers:
        return None
    value = numbers[-1].replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _normalize_text(prediction).split()
    ground_truth_tokens = _normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if not prediction_tokens or not ground_truth_tokens:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text
