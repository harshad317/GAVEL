from __future__ import annotations

import csv
import json

from pact_el.benchmarks.adapters import (
    drop_answer_to_strings,
    extract_gsm8k_answer,
    normalize_examples,
)
from pact_el.benchmarks.registry import get_benchmark_spec, list_benchmarks
from pact_el.benchmarks.scoring import (
    ScoreAccumulator,
    format_accuracy,
    score_prediction,
    score_predictions,
    summarize_scores,
)
from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind


def test_registry_contains_requested_official_benchmarks():
    ids = {spec.benchmark_id for spec in list_benchmarks()}

    assert {
        "gsm8k",
        "ifbench",
        "hotpotqa",
        "drop",
        "mbpp",
        "truthfulqa",
        "livebench_math",
        "mmlu",
        "mmlu_pro",
    } <= ids
    assert get_benchmark_spec("mmlu-pro").benchmark_id == "mmlu_pro"


def test_gsm8k_adapter_extracts_final_numeric_answer(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "A has 2 apples and buys 3. How many?",
                "answer": "A has 2 + 3 = 5 apples.\n#### 5",
            }
        )
        + "\n"
    )

    [example] = normalize_examples(get_benchmark_spec("gsm8k"), "test", [path])

    assert example.expected_answer == "5"
    assert extract_gsm8k_answer("work\n#### 1,200") == "1,200"


def test_drop_answer_to_strings_handles_spans_numbers_and_dates():
    assert drop_answer_to_strings({"spans": ["A", "B"], "number": "", "date": {}}) == [
        "A",
        "B",
        "A, B",
    ]
    assert drop_answer_to_strings({"spans": [], "number": "12", "date": {}}) == ["12"]
    assert drop_answer_to_strings(
        {"spans": [], "number": "", "date": {"day": "4", "month": "July", "year": "1776"}}
    ) == ["4 July 1776"]


def test_mmlu_adapter_normalizes_csv_subject(tmp_path):
    data_dir = tmp_path / "data" / "test"
    data_dir.mkdir(parents=True)
    path = data_dir / "abstract_algebra_test.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["What is 2+2?", "3", "4", "5", "6", "B"])

    [example] = normalize_examples(get_benchmark_spec("mmlu"), "test", [path])

    assert example.example_id == "mmlu:test:abstract_algebra:0"
    assert example.expected_answer == "B"
    assert example.choices == ["3", "4", "5", "6"]


def test_scoring_numeric_multiple_choice_and_f1():
    numeric = BenchmarkExample(
        benchmark_id="gsm8k",
        example_id="n",
        split="test",
        prompt="",
        expected_answer="1200",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )
    mc = BenchmarkExample(
        benchmark_id="mmlu",
        example_id="m",
        split="test",
        prompt="",
        expected_answer="C",
        choices=["alpha", "beta", "gamma", "delta"],
        metric=MetricKind.MULTIPLE_CHOICE_ACCURACY,
        source_url="official",
    )
    f1 = BenchmarkExample(
        benchmark_id="drop",
        example_id="f",
        split="dev",
        prompt="",
        expected_answer="July 4 1776",
        metric=MetricKind.TOKEN_F1,
        source_url="official",
        metadata={"answer_aliases": ["July 4 1776"]},
    )

    assert score_prediction(numeric, "Answer: 1,200").passed
    assert score_prediction(mc, "The answer is C.").passed
    assert score_prediction(f1, "4 July 1776").score == 1.0


def test_score_summary_reports_accuracy_separately_from_mean_score():
    numeric = BenchmarkExample(
        benchmark_id="gsm8k",
        example_id="n",
        split="test",
        prompt="",
        expected_answer="1200",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )

    results = [
        score_prediction(numeric, "Answer: 1,200"),
        score_prediction(numeric, "Answer: 999"),
    ]
    summary = summarize_scores(results)

    assert summary["accuracy"] == 0.5
    assert summary["mean_score"] == 0.5


def test_score_accumulator_gives_every_method_the_same_accuracy_postfix():
    numeric = BenchmarkExample(
        benchmark_id="gsm8k",
        example_id="n",
        split="test",
        prompt="",
        expected_answer="1200",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )
    tracker = ScoreAccumulator()

    tracker.add(score_prediction(numeric, "Answer: 1,200"))
    tracker.add(score_prediction(numeric, "Answer: 999"))

    assert tracker.summary()["accuracy"] == 0.5
    assert tracker.progress_postfix() == {
        "accuracy": "50.0%",
        "passed": 1,
        "scored": 2,
    }
    assert format_accuracy(tracker.accuracy, precision=2) == "50.00%"


def test_score_predictions_can_use_process_workers():
    examples = [
        BenchmarkExample(
            benchmark_id="gsm8k",
            example_id="n0",
            split="test",
            prompt="",
            expected_answer="1200",
            metric=MetricKind.NUMERIC_EXACT,
            source_url="official",
        ),
        BenchmarkExample(
            benchmark_id="gsm8k",
            example_id="n1",
            split="test",
            prompt="",
            expected_answer="7",
            metric=MetricKind.NUMERIC_EXACT,
            source_url="official",
        ),
    ]

    results = score_predictions(
        examples,
        {"n0": "Answer: 1,200", "n1": "Answer: 8"},
        workers=2,
    )

    assert [result.example_id for result in results] == ["n0", "n1"]
    assert summarize_scores(results)["accuracy"] == 0.5


def test_mbpp_scoring_requires_explicit_code_execution():
    example = BenchmarkExample(
        benchmark_id="mbpp",
        example_id="code",
        split="sanitized",
        prompt="",
        expected_answer="",
        metric=MetricKind.PASS_AT_1,
        source_url="official",
        metadata={"test_list": ["assert add(1, 2) == 3"]},
    )

    result = score_prediction(example, "def add(a, b):\n    return a + b")

    assert result.score is None
    assert result.passed is None
