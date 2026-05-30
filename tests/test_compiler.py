from __future__ import annotations

import pytest

from pact_el.compiler import CompilerOutput, strict_parse_json_model
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

