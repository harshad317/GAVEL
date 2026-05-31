from __future__ import annotations

from pact_el.schemas import ValidatorKind, ValidatorSpec
from pact_el.validators import ValidationContext, evaluate_jsonlogic, validate_specs


def test_jsonlogic_var_and_enum_predicate():
    data = {"parsed_output": {"priority": "urgent"}}

    assert evaluate_jsonlogic(
        {"in": [{"var": "parsed_output.priority"}, ["low", "urgent"]]},
        data,
    )


def test_validator_suite_passes_jsonschema_regex_numeric():
    context = ValidationContext(
        input={"ticket": "outage"},
        output='{"priority": "urgent", "score": 0.91}',
    )
    specs = [
        ValidatorSpec(
            validator_id="schema",
            kind=ValidatorKind.JSON_SCHEMA,
            config={
                "schema": {
                    "type": "object",
                    "required": ["priority", "score"],
                    "properties": {
                        "priority": {"type": "string"},
                        "score": {"type": "number"},
                    },
                }
            },
        ),
        ValidatorSpec(
            validator_id="regex",
            kind=ValidatorKind.REGEX,
            config={"pattern": "urgent", "field": "output_text"},
        ),
        ValidatorSpec(
            validator_id="numeric",
            kind=ValidatorKind.NUMERIC,
            config={"field": "parsed_output.score", "op": ">=", "value": 0.9},
        ),
    ]

    results = validate_specs(specs, context)

    assert all(result.passed for result in results)


def test_regex_must_not_match_fails_when_forbidden_text_present():
    context = ValidationContext(input={}, output="Here is extra commentary.")
    spec = ValidatorSpec(
        validator_id="no_commentary",
        kind=ValidatorKind.REGEX,
        config={
            "pattern": "commentary",
            "field": "output_text",
            "must_not_match": True,
        },
    )

    [result] = validate_specs([spec], context)

    assert not result.passed


def test_regex_validator_accepts_string_dotall_flag_alias():
    context = ValidationContext(input={}, output="alpha\nbeta")
    spec = ValidatorSpec(
        validator_id="dotall",
        kind=ValidatorKind.REGEX,
        config={"pattern": "alpha.*beta", "field": "output_text", "flags": "s"},
    )

    [result] = validate_specs([spec], context)

    assert result.passed
