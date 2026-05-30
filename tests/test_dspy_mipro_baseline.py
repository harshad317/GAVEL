from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from pact_el.baselines.dspy_mipro import (
    DSPyMIPROConfig,
    _evaluate_program_with_stats,
    build_dspy_metric,
    build_gepa_metric,
    run_dspy_baseline,
)
from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind


class FakeDSPyExample:
    def __init__(self, **kwargs):
        self._store = dict(kwargs)
        self.input_keys = ()

    def with_inputs(self, *keys):
        self.input_keys = keys
        return self

    def __getattr__(self, key):
        try:
            return self._store[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __getitem__(self, key):
        return self._store[key]


class FakeProgram:
    def __init__(self, signature):
        self.signature = signature
        self.saved_path = None

    def __call__(self, **kwargs):
        question = kwargs["question"]
        if "2+2" in question:
            return SimpleNamespace(answer="4")
        return SimpleNamespace(answer="A")

    def save(self, path):
        self.saved_path = path
        with open(path, "w") as handle:
            handle.write('{"fake": true}')


class SlowProgram(FakeProgram):
    def __call__(self, **kwargs):
        time.sleep(0.05)
        return super().__call__(**kwargs)


class FakeMIPROv2:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compile_kwargs = None
        FakeMIPROv2.instances.append(self)

    def compile(self, program, **kwargs):
        self.compile_kwargs = kwargs
        return program


class FakeGEPA:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.compile_kwargs = None
        FakeGEPA.instances.append(self)

    def compile(self, program, **kwargs):
        self.compile_kwargs = kwargs
        return program


class FakeLM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.history = []


def fake_dspy_module():
    return SimpleNamespace(
        Example=FakeDSPyExample,
        Predict=FakeProgram,
        ChainOfThought=FakeProgram,
        MIPROv2=FakeMIPROv2,
        GEPA=FakeGEPA,
        LM=FakeLM,
        configured=None,
        configure=lambda **kwargs: setattr(fake_dspy_module.module, "configured", kwargs),
    )


fake_dspy_module.module = None


def install_fake_dspy(monkeypatch):
    module = fake_dspy_module()
    fake_dspy_module.module = module
    FakeMIPROv2.instances = []
    FakeGEPA.instances = []
    monkeypatch.setitem(sys.modules, "dspy", module)
    return module


def numeric_example(example_id="gsm8k:test:0"):
    return BenchmarkExample(
        benchmark_id="gsm8k",
        example_id=example_id,
        split="test",
        prompt="What is 2+2?",
        expected_answer="4",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )


def test_dspy_metric_delegates_to_normalized_scorer():
    example = numeric_example()
    dspy_example = FakeDSPyExample(
        question=example.prompt,
        answer=example.expected_answer,
        benchmark_example=example.model_dump(mode="json"),
    ).with_inputs("question")

    metric = build_dspy_metric()

    assert metric(dspy_example, SimpleNamespace(answer="Answer: 4")) == 1.0
    assert metric(dspy_example, SimpleNamespace(answer="Answer: 5")) == 0.0


def test_gepa_metric_returns_feedback_score():
    example = numeric_example()
    dspy_example = FakeDSPyExample(
        question=example.prompt,
        answer=example.expected_answer,
        benchmark_example=example.model_dump(mode="json"),
    ).with_inputs("question")

    metric = build_gepa_metric()
    result = metric(dspy_example, SimpleNamespace(answer="Answer: 4"), None, None, None)

    assert result["score"] == 1.0
    assert result.score == 1.0
    assert sum([result]) == 1.0
    assert "Metric: numeric_exact" in result.feedback


def test_direct_dspy_baseline_writes_predictions_and_scores(tmp_path, monkeypatch):
    module = install_fake_dspy(monkeypatch)
    config = DSPyMIPROConfig(
        optimizer="dspy",
        program="predict",
        model="fake/model",
        output_dir=tmp_path,
        temperature=0.2,
        cache=False,
        workers=2,
    )

    result = run_dspy_baseline(
        [numeric_example("gsm8k:test:0"), numeric_example("gsm8k:test:1")],
        config=config,
    )

    assert result.summary["accuracy"] == 1.0
    assert result.summary["mean_score"] == 1.0
    assert result.summary["cache"] is False
    assert result.summary["workers"] == 2
    assert result.summary["method"] == "dspy"
    assert result.summary["temperature"] == 0.2
    assert result.summary["effective_temperature"] == 0.2
    assert result.summary["split_results"]["test"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["api_calls"] == 2
    assert result.summary["split_results"]["train"]["score"] is None
    assert module.configured["lm"].kwargs["temperature"] == 0.2
    assert result.predictions_path.exists()
    assert result.scores_path.exists()
    assert result.program_path and result.program_path.exists()


def test_dspy_evaluation_uses_worker_concurrency():
    _predictions, _scores, stats = _evaluate_program_with_stats(
        SlowProgram("question -> answer"),
        [numeric_example(f"gsm8k:test:{index}") for index in range(6)],
        workers=3,
        show_progress=False,
    )

    assert stats.requested_workers == 3
    assert stats.max_in_flight == 3


def test_mipro_baseline_invokes_official_compile_shape(tmp_path, monkeypatch):
    install_fake_dspy(monkeypatch)
    config = DSPyMIPROConfig(
        optimizer="mipro",
        program="cot",
        model="fake/model",
        output_dir=tmp_path,
        auto="light",
        seed=123,
        minibatch_size=2,
        workers=16,
    )
    train = [numeric_example("gsm8k:train:0"), numeric_example("gsm8k:train:1")]
    val = [numeric_example("gsm8k:validation:0")]

    result = run_dspy_baseline(
        [numeric_example()],
        config=config,
        train_examples=train,
        val_examples=val,
    )

    [mipro] = FakeMIPROv2.instances
    assert mipro.kwargs["metric"]
    assert mipro.kwargs["auto"] == "light"
    assert mipro.kwargs["seed"] == 123
    assert mipro.kwargs["num_threads"] == 16
    assert len(mipro.compile_kwargs["trainset"]) == 2
    assert len(mipro.compile_kwargs["valset"]) == 1
    assert mipro.compile_kwargs["minibatch_size"] == 2
    assert result.summary["optimizer"] == "mipro"
    assert result.summary["method"] == "miprov2"
    assert result.summary["effective_temperature"] == "provider_default"
    assert result.summary["split_results"]["train"]["score"] == 1.0
    assert result.summary["split_results"]["val"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0
    assert "optimization" in result.summary["split_results"]


def test_gepa_baseline_invokes_official_compile_shape(tmp_path, monkeypatch):
    install_fake_dspy(monkeypatch)
    config = DSPyMIPROConfig(
        optimizer="gepa",
        program="cot",
        model="fake/model",
        output_dir=tmp_path,
        auto="heavy",
        seed=123,
        workers=16,
        reflection_model="fake/reflection",
        temperature=0.1,
        reflection_temperature=0.3,
        reflection_minibatch_size=2,
    )
    train = [numeric_example("gsm8k:train:0"), numeric_example("gsm8k:train:1")]
    val = [numeric_example("gsm8k:validation:0")]

    result = run_dspy_baseline(
        [numeric_example()],
        config=config,
        train_examples=train,
        val_examples=val,
    )

    [gepa] = FakeGEPA.instances
    assert gepa.kwargs["metric"]
    assert gepa.kwargs["auto"] == "heavy"
    assert gepa.kwargs["seed"] == 123
    assert gepa.kwargs["num_threads"] == 16
    assert gepa.kwargs["reflection_minibatch_size"] == 2
    assert gepa.kwargs["reflection_lm"].args[0] == "fake/reflection"
    assert len(gepa.compile_kwargs["trainset"]) == 2
    assert len(gepa.compile_kwargs["valset"]) == 1
    assert result.summary["optimizer"] == "gepa"
    assert result.summary["method"] == "gepa"
    assert result.summary["reflection_model"] == "fake/reflection"
    assert result.summary["temperature"] == 0.1
    assert result.summary["reflection_temperature"] == 0.3
    assert result.summary["effective_reflection_temperature"] == 0.3
    assert gepa.kwargs["reflection_lm"].kwargs["temperature"] == 0.3
    assert result.summary["split_results"]["train"]["score"] == 1.0
    assert result.summary["split_results"]["val"]["score"] == 1.0
    assert result.summary["split_results"]["test"]["score"] == 1.0


def test_dspy_temperature_validation():
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        DSPyMIPROConfig(temperature=-0.1).validate()

    with pytest.raises(ValueError, match="reflection_temperature must be between 0 and 2"):
        DSPyMIPROConfig(reflection_temperature=2.1).validate()
