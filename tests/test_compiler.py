from __future__ import annotations

import pytest

from pact_el.compiler import CompilerOutput, build_compiler_messages, strict_parse_json_model
from pact_el.ledger import build_evidence_ledger
from pact_el.schemas import CanaryKind


def test_strict_parse_rejects_markdown_fenced_json(sample_compiler_output):
    payload = sample_compiler_output.model_dump_json()

    with pytest.raises(ValueError):
        strict_parse_json_model(CompilerOutput, f"```json\n{payload}\n```")


def test_compiler_output_requires_four_canary_families(sample_compiler_output):
    parsed = strict_parse_json_model(
        CompilerOutput,
        sample_compiler_output.model_dump_json(),
    )

    assert {canary.kind for canary in parsed.canaries} == {
        CanaryKind.FIX,
        CanaryKind.BOUNDARY,
        CanaryKind.REGRESSION,
        CanaryKind.FORMAT_SCHEMA,
    }


def test_compiler_messages_require_transferable_contracts():
    ledger = build_evidence_ledger(
        logs=[
            {
                "input": "train example",
                "expected": "Expected answer: 4",
                "observed": "Observed output: 5",
                "passed": False,
                "metadata": {"diagnostics": {"summary": "observed_number='5', expected_number='4'"}},
            }
        ]
    )

    messages = build_compiler_messages(
        prompt="Answer math questions.",
        task_spec="GSM8K-style math.",
        rubric="numeric exact",
        ledger=ledger,
    )
    content = messages[-1]["content"]

    assert "transfers to unseen validation and test examples" in content
    assert "Do not put per-example answers" in content
    assert "fresh stress cases rather than copied evidence rows" in content
    assert "diagnostics" in content
