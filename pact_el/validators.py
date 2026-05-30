"""Deterministic validators and JSONLogic-style GuaranteeScript evaluation."""

from __future__ import annotations

import json
import operator
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from pact_el.schemas import (
    Canary,
    CanaryResult,
    CanaryStatus,
    ValidatorKind,
    ValidatorResult,
    ValidatorSpec,
)


class ValidationContext:
    """Runtime data visible to validators."""

    def __init__(
        self,
        input: Any,
        output: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.input = input
        self.output = output
        self.metadata = dict(metadata or {})
        self.parsed_output = parse_json_if_possible(output)

    @property
    def output_text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, sort_keys=True)

    def as_data(self) -> Dict[str, Any]:
        return {
            "input": self.input,
            "output": self.output,
            "output_text": self.output_text,
            "parsed_output": self.parsed_output,
            "metadata": self.metadata,
        }


class JsonLogicError(ValueError):
    pass


def parse_json_if_possible(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def resolve_path(data: Any, path: Optional[str], default: Any = None) -> Any:
    if path is None or path == "":
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return default
            continue
        return default
    return current


def _json_truthy(value: Any) -> bool:
    return bool(value)


def evaluate_jsonlogic(rule: Any, data: Any) -> Any:
    """Evaluate a safe, local subset of JSONLogic.

    This evaluator intentionally avoids dynamic code execution. It supports the
    operators needed for GuaranteeScript clauses and deterministic validators.
    """

    if isinstance(rule, list):
        return [evaluate_jsonlogic(item, data) for item in rule]
    if not isinstance(rule, dict):
        return rule
    if len(rule) != 1:
        return {key: evaluate_jsonlogic(value, data) for key, value in rule.items()}

    op, raw_args = next(iter(rule.items()))

    if op == "var":
        if isinstance(raw_args, list):
            path = raw_args[0] if raw_args else ""
            default = raw_args[1] if len(raw_args) > 1 else None
        else:
            path = raw_args
            default = None
        if path is None or path == "":
            return data
        return resolve_path(data, str(path), default)

    args = raw_args if isinstance(raw_args, list) else [raw_args]

    if op == "if":
        index = 0
        while index + 1 < len(args):
            if _json_truthy(evaluate_jsonlogic(args[index], data)):
                return evaluate_jsonlogic(args[index + 1], data)
            index += 2
        if index < len(args):
            return evaluate_jsonlogic(args[index], data)
        return None

    if op == "and":
        result = None
        for arg in args:
            result = evaluate_jsonlogic(arg, data)
            if not _json_truthy(result):
                return result
        return result

    if op == "or":
        result = None
        for arg in args:
            result = evaluate_jsonlogic(arg, data)
            if _json_truthy(result):
                return result
        return result

    if op in {"!", "not"}:
        return not _json_truthy(evaluate_jsonlogic(args[0], data))

    if op == "!!":
        return _json_truthy(evaluate_jsonlogic(args[0], data))

    evaluated = [evaluate_jsonlogic(arg, data) for arg in args]

    if op in {"==", "==="}:
        return evaluated[0] == evaluated[1]
    if op in {"!=", "!=="}:
        return evaluated[0] != evaluated[1]

    if op in {"<", "<=", ">", ">="}:
        if len(evaluated) < 2:
            raise JsonLogicError(f"operator {op} requires at least two operands")
        cmp_ops = {
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        return all(
            cmp_ops[op](left, right)
            for left, right in zip(evaluated, evaluated[1:])
        )

    if op == "in":
        needle, haystack = evaluated[0], evaluated[1]
        if haystack is None:
            return False
        return needle in haystack

    if op == "missing":
        paths = evaluated
        if len(paths) == 1 and isinstance(paths[0], list):
            paths = paths[0]
        return [path for path in paths if resolve_path(data, str(path)) is None]

    if op == "missing_some":
        minimum = int(evaluated[0])
        paths = list(evaluated[1])
        present = [path for path in paths if resolve_path(data, str(path)) is not None]
        if len(present) >= minimum:
            return []
        return [path for path in paths if path not in present]

    if op == "cat":
        return "".join("" if value is None else str(value) for value in evaluated)

    if op == "substr":
        text = str(evaluated[0])
        start = int(evaluated[1])
        if len(evaluated) > 2:
            length = int(evaluated[2])
            return text[start : start + length]
        return text[start:]

    if op in {"+", "-", "*", "/", "%"}:
        if not evaluated:
            return 0
        if op == "+":
            return sum(evaluated)
        result = evaluated[0]
        for value in evaluated[1:]:
            if op == "-":
                result -= value
            elif op == "*":
                result *= value
            elif op == "/":
                result /= value
            elif op == "%":
                result %= value
        return result

    if op == "max":
        return max(evaluated)
    if op == "min":
        return min(evaluated)

    if op in {"all", "some", "none"}:
        collection = evaluate_jsonlogic(args[0], data)
        predicate = args[1]
        if not isinstance(collection, list):
            collection = []
        predicate_values = [
            _json_truthy(evaluate_jsonlogic(predicate, _scoped_data(data, item)))
            for item in collection
        ]
        if op == "all":
            return bool(collection) and all(predicate_values)
        if op == "some":
            return any(predicate_values)
        return not any(predicate_values)

    raise JsonLogicError(f"unsupported JSONLogic operator: {op}")


def _scoped_data(parent: Any, item: Any) -> Dict[str, Any]:
    if isinstance(item, Mapping):
        scoped = dict(item)
    else:
        scoped = {"value": item}
    scoped[""] = item
    scoped["_parent"] = parent
    return scoped


class ValidatorRegistry:
    """Dispatch table for deterministic canary validators."""

    def validate(self, spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
        if spec.kind == ValidatorKind.JSON_SCHEMA:
            return validate_jsonschema(spec, context)
        if spec.kind == ValidatorKind.REGEX:
            return validate_regex(spec, context)
        if spec.kind == ValidatorKind.EXACT_MATCH:
            return validate_exact_match(spec, context)
        if spec.kind == ValidatorKind.ENUM:
            return validate_enum(spec, context)
        if spec.kind == ValidatorKind.NUMERIC:
            return validate_numeric(spec, context)
        if spec.kind == ValidatorKind.JSON_LOGIC:
            return validate_jsonlogic(spec, context)
        if spec.kind == ValidatorKind.LLM_JUDGE:
            return ValidatorResult(
                validator_id=spec.validator_id,
                kind=spec.kind,
                passed=False,
                message=(
                    "LLM judge validators are counted fallbacks and must be "
                    "executed by a judge client before the gate can accept."
                ),
            )
        raise ValueError(f"unsupported validator kind: {spec.kind}")


def _field_value(config: Mapping[str, Any], context: ValidationContext) -> Any:
    path = config.get("field")
    if path is None:
        return context.output
    return resolve_path(context.as_data(), str(path))


def validate_jsonschema(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    schema = spec.config.get("schema")
    if not isinstance(schema, Mapping):
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="jsonschema validator requires a schema object",
        )
    instance = _field_value(spec.config, context)
    if spec.config.get("field") is None and context.parsed_output is not None:
        instance = context.parsed_output
    validator = Draft202012Validator(schema)
    errors: List[ValidationError] = sorted(
        validator.iter_errors(instance), key=lambda error: list(error.path)
    )
    if errors:
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="JSON schema validation failed",
            observed=instance,
            expected=schema,
            details={
                "errors": [
                    {
                        "path": list(error.path),
                        "message": error.message,
                    }
                    for error in errors
                ]
            },
        )
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=True,
        message="JSON schema validation passed",
        observed=instance,
        expected=schema,
    )


def validate_regex(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    pattern = spec.config.get("pattern")
    if not isinstance(pattern, str):
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="regex validator requires a pattern",
        )
    flags = 0
    for flag in spec.config.get("flags", []):
        if str(flag).lower() == "ignorecase":
            flags |= re.IGNORECASE
        elif str(flag).lower() == "multiline":
            flags |= re.MULTILINE
        elif str(flag).lower() == "dotall":
            flags |= re.DOTALL
    value = _field_value({"field": spec.config.get("field", "output_text")}, context)
    text = "" if value is None else str(value)
    mode = spec.config.get("mode", "search")
    match = (
        re.fullmatch(pattern, text, flags)
        if mode == "fullmatch"
        else re.search(pattern, text, flags)
    )
    must_not_match = bool(spec.config.get("must_not_match", False))
    passed = (match is not None) != must_not_match
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=passed,
        message="regex validation passed" if passed else "regex validation failed",
        observed=text,
        expected=pattern,
        details={"mode": mode, "must_not_match": must_not_match},
    )


def validate_exact_match(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    expected = spec.config.get("expected")
    observed = _field_value(spec.config, context)
    case_sensitive = bool(spec.config.get("case_sensitive", True))
    normalize_whitespace = bool(spec.config.get("normalize_whitespace", False))
    left = observed
    right = expected
    if isinstance(left, str) and isinstance(right, str):
        if normalize_whitespace:
            left = " ".join(left.split())
            right = " ".join(right.split())
        if not case_sensitive:
            left = left.lower()
            right = right.lower()
    passed = left == right
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=passed,
        message="exact match passed" if passed else "exact match failed",
        observed=observed,
        expected=expected,
    )


def validate_enum(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    allowed = spec.config.get("allowed")
    if not isinstance(allowed, list):
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="enum validator requires an allowed list",
        )
    observed = _field_value(spec.config, context)
    passed = observed in allowed
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=passed,
        message="enum validation passed" if passed else "enum validation failed",
        observed=observed,
        expected=allowed,
    )


def validate_numeric(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    observed = _field_value(spec.config, context)
    expected = spec.config.get("value")
    op = spec.config.get("op")
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }
    if op not in ops:
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="numeric validator requires op in <, <=, ==, !=, >=, >",
        )
    try:
        observed_number = float(observed)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="numeric validator observed or expected value is not numeric",
            observed=observed,
            expected=expected,
        )
    passed = bool(ops[op](observed_number, expected_number))
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=passed,
        message="numeric validation passed" if passed else "numeric validation failed",
        observed=observed_number,
        expected={"op": op, "value": expected_number},
    )


def validate_jsonlogic(spec: ValidatorSpec, context: ValidationContext) -> ValidatorResult:
    rule = spec.config.get("rule")
    if not isinstance(rule, Mapping):
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message="jsonlogic validator requires a rule object",
        )
    data = context.as_data()
    try:
        observed = evaluate_jsonlogic(rule, data)
    except Exception as exc:
        return ValidatorResult(
            validator_id=spec.validator_id,
            kind=spec.kind,
            passed=False,
            message=f"jsonlogic evaluation error: {exc}",
            expected=rule,
        )
    passed = bool(observed)
    return ValidatorResult(
        validator_id=spec.validator_id,
        kind=spec.kind,
        passed=passed,
        message="jsonlogic validation passed" if passed else "jsonlogic validation failed",
        observed=observed,
        expected=rule,
    )


def validate_specs(
    specs: Iterable[ValidatorSpec],
    context: ValidationContext,
    registry: Optional[ValidatorRegistry] = None,
) -> List[ValidatorResult]:
    validator_registry = registry or ValidatorRegistry()
    return [validator_registry.validate(spec, context) for spec in specs]


def validate_canary_output(
    canary: Canary,
    output: Any,
    metadata: Optional[Mapping[str, Any]] = None,
    registry: Optional[ValidatorRegistry] = None,
) -> CanaryResult:
    context = ValidationContext(canary.input, output, metadata)
    validator_outputs = validate_specs(canary.validator_specs, context, registry)
    passed = bool(validator_outputs) and all(result.passed for result in validator_outputs)
    if not canary.validator_specs:
        passed = False
    failure_reason = ""
    if not passed:
        failure_reason = "; ".join(
            result.message for result in validator_outputs if not result.passed
        ) or "canary has no validators"
    return CanaryResult(
        canary_id=canary.canary_id,
        kind=canary.kind,
        status=CanaryStatus.PASSED if passed else CanaryStatus.FAILED,
        passed=passed,
        input=canary.input,
        output=output,
        validator_outputs=validator_outputs,
        failure_reason=failure_reason,
        metadata=dict(metadata or {}),
    )


def validate_guarantee_script(
    clauses: Iterable[Mapping[str, Any]],
    input: Any,
    output: Any,
) -> List[ValidatorResult]:
    """Evaluate GuaranteeScript clauses as JSONLogic validators."""

    results: List[ValidatorResult] = []
    context = ValidationContext(input=input, output=output)
    for clause in clauses:
        validator = ValidatorSpec(
            validator_id=str(clause["guarantee_id"]),
            kind=ValidatorKind.JSON_LOGIC,
            config={"rule": clause["predicate"]},
            severity=clause.get("severity", "high"),
            description=clause.get("description", ""),
        )
        results.append(validate_jsonlogic(validator, context))
    return results

