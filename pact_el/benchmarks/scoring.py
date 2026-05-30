"""Local first-pass scoring for normalized benchmark examples."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tqdm.auto import tqdm

from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind, ScoreResult


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
) -> List[ScoreResult]:
    results: List[ScoreResult] = []
    progress = tqdm(
        total=len(examples),
        desc=description,
        unit="ex",
        colour="green",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    scored = 0
    passed = 0
    score_total = 0.0
    with progress:
        for example in examples:
            prediction = predictions.get(example.example_id)
            result = score_prediction(
                example,
                prediction,
                allow_code_execution=allow_code_execution,
            )
            results.append(result)
            if result.score is not None:
                scored += 1
                score_total += float(result.score)
                if result.passed is True:
                    passed += 1
                progress.set_postfix(
                    scored=scored,
                    passed=passed,
                    mean=f"{score_total / scored:.3f}",
                    refresh=False,
                )
            progress.update(1)
    return results


def summarize_scores(results: Sequence[ScoreResult]) -> Dict[str, Any]:
    scored = [result for result in results if result.score is not None]
    passed = [result for result in scored if result.passed is True]
    return {
        "total": len(results),
        "scored": len(scored),
        "unscored": len(results) - len(scored),
        "passed": len(passed),
        "mean_score": (
            sum(float(result.score) for result in scored) / len(scored)
            if scored
            else None
        ),
    }


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
