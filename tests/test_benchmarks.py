from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from pact_el.benchmarks import scoring as scoring_module
from pact_el.benchmarks.adapters import (
    drop_answer_to_strings,
    extract_gsm8k_answer,
    normalize_examples,
)
from pact_el.benchmarks.registry import get_benchmark_spec, list_benchmarks
from pact_el.benchmarks.scoring import (
    ScoreAccumulator,
    format_accuracy,
    format_mean_score,
    score_prediction,
    score_predictions,
    summarize_scores,
    summarize_unscored_reasons,
)
from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind, ScoreResult


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
        "acc": "50.0%",
        "mean": "0.500",
        "passed": 1,
        "scored": 2,
        "unscored": 0,
    }
    assert format_accuracy(tracker.accuracy, precision=2) == "50.00%"
    assert format_mean_score(tracker.mean_score) == "0.500"


def test_score_accumulator_postfix_reports_unscored_progress():
    tracker = ScoreAccumulator()

    tracker.add(
        ScoreResult(
            example_id="ifbench:test:0",
            metric=MetricKind.OFFICIAL_EVALUATOR,
            score=None,
            passed=None,
            prediction="",
            expected=None,
            details={"reason": "official evaluator is not installed"},
        )
    )

    assert tracker.progress_postfix() == {
        "acc": "n/a",
        "mean": "n/a",
        "passed": 0,
        "scored": 0,
        "unscored": 1,
    }


def test_unscored_reason_summary_keeps_actionable_fix():
    results = [
        ScoreResult(
            example_id="ifbench:test:0",
            metric=MetricKind.OFFICIAL_EVALUATOR,
            score=None,
            passed=None,
            prediction="",
            expected=None,
            details={
                "reason": "official IFBench evaluator is not installed",
                "install": "python -m pip install -e '.[ifbench]'",
                "missing_module": "ifbench",
            },
        ),
        ScoreResult(
            example_id="ifbench:test:1",
            metric=MetricKind.OFFICIAL_EVALUATOR,
            score=None,
            passed=None,
            prediction="",
            expected=None,
            details={
                "reason": "official IFBench evaluator is not installed",
                "install": "python -m pip install -e '.[ifbench]'",
                "missing_module": "ifbench",
            },
        ),
    ]

    assert summarize_unscored_reasons(results) == [
        {
            "reason": "official IFBench evaluator is not installed",
            "count": 2,
            "install": "python -m pip install -e '.[ifbench]'",
            "missing_module": "ifbench",
        }
    ]


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


def test_ifbench_official_evaluator_metric_uses_installed_verifier(monkeypatch):
    class FakeEvaluationLib:
        class InputExample:
            def __init__(self, key, instruction_id_list, prompt, kwargs):
                self.key = key
                self.instruction_id_list = instruction_id_list
                self.prompt = prompt
                self.kwargs = kwargs

        @staticmethod
        def test_instruction_following_loose(inp, prompt_to_response):
            return SimpleNamespace(
                instruction_id_list=inp.instruction_id_list,
                prompt=inp.prompt,
                response=prompt_to_response[inp.prompt],
                follow_all_instructions=True,
                follow_instruction_list=[True],
            )

    monkeypatch.setitem(sys.modules, "evaluation_lib", FakeEvaluationLib)
    example = BenchmarkExample(
        benchmark_id="ifbench",
        example_id="ifbench:test:0",
        split="test",
        prompt="Say hello.",
        expected_answer=None,
        metric=MetricKind.OFFICIAL_EVALUATOR,
        source_url="official",
        metadata={
            "official_input_row": {
                "key": "0",
                "prompt": "Say hello.",
                "instruction_id_list": ["format:test"],
                "kwargs": [{}],
            }
        },
    )

    result = score_prediction(example, "hello")

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["evaluator"] == "allenai/IFBench test_instruction_following_loose"


def test_ifbench_evaluator_missing_details_reports_attempted_imports(monkeypatch):
    def fake_import_module(module_name):
        if module_name == "evaluation_lib":
            raise ModuleNotFoundError("No module named 'evaluation_lib'", name="evaluation_lib")
        if module_name == "ifbench.evaluation_lib":
            raise ModuleNotFoundError("No module named 'ifbench'", name="ifbench")
        raise AssertionError(f"unexpected import: {module_name}")

    def missing_distribution(_name):
        raise scoring_module.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(scoring_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(scoring_module.importlib_metadata, "distribution", missing_distribution)

    details = scoring_module.ifbench_evaluator_missing_details()

    assert details == {
        "reason": "official IFBench evaluator is not installed",
        "install": "python -m pip install -e '.[ifbench]'",
        "missing_module": "evaluation_lib",
        "attempted_modules": [
            {"module": "evaluation_lib", "missing_module": "evaluation_lib"},
            {"module": "ifbench.evaluation_lib", "missing_module": "ifbench"},
        ],
    }


def test_ifbench_evaluator_import_does_not_hide_nested_import_errors(monkeypatch):
    def fake_import_module(module_name):
        if module_name == "evaluation_lib":
            raise ModuleNotFoundError(
                "No module named 'instructions_registry'",
                name="instructions_registry",
            )
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(scoring_module.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        scoring_module._import_ifbench_evaluation_lib()

    assert exc_info.value.name == "instructions_registry"


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
