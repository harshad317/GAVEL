from __future__ import annotations

import pytest

from pact_el.benchmarks.schemas import BenchmarkExample, MetricKind
from pact_el.benchmarks.splitting import (
    prompt_fingerprint,
    select_disjoint_examples,
    select_three_way,
    validate_no_leakage,
)


def make_example(example_id: str, prompt: str) -> BenchmarkExample:
    return BenchmarkExample(
        benchmark_id="gsm8k",
        example_id=example_id,
        split=example_id.split(":")[1],
        prompt=prompt,
        expected_answer="1",
        metric=MetricKind.NUMERIC_EXACT,
        source_url="official",
    )


def test_three_way_selection_uses_disjoint_ids_and_prompt_fingerprints():
    train_pool = [
        make_example("gsm8k:train:0", "Question 0"),
        make_example("gsm8k:train:1", "Question 1"),
        make_example("gsm8k:train:2", "Question 2"),
        make_example("gsm8k:train:3", "Question 3"),
    ]
    test_pool = [
        make_example("gsm8k:test:0", "Question 0"),
        make_example("gsm8k:test:1", "Heldout 1"),
        make_example("gsm8k:test:2", "Heldout 2"),
    ]

    selection = select_three_way(
        train_pool=train_pool,
        validation_pool=None,
        test_pool=test_pool,
        train_n=2,
        val_n=1,
        test_n=2,
        seed=7,
    )

    assert len(selection.train) == 2
    assert len(selection.validation) == 1
    assert len(selection.test) == 2
    assert selection.manifest["leakage_check"]["passed"] is True
    validate_no_leakage(
        {
            "train": selection.train,
            "validation": selection.validation,
            "test": selection.test,
        }
    )


def test_selection_fails_when_requested_size_cannot_avoid_leakage():
    used = set()
    first = make_example("gsm8k:train:0", "same prompt")
    selected = select_disjoint_examples(
        [first],
        name="train",
        n=1,
        used_keys=used,
        seed=1,
    )

    assert selected.selected == [first]
    with pytest.raises(ValueError, match="could only select 0"):
        select_disjoint_examples(
            [make_example("gsm8k:test:0", "same prompt")],
            name="test",
            n=1,
            used_keys=used,
            seed=2,
        )


def test_prompt_fingerprint_normalizes_spacing_and_case():
    left = make_example("gsm8k:train:0", "What  is  2 + 2?")
    right = make_example("gsm8k:test:0", "what is 2 + 2?")

    assert prompt_fingerprint(left) == prompt_fingerprint(right)
