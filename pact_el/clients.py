"""Async optimizer and target client abstractions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from pact_el.renderer import estimate_token_count
from pact_el.schemas import CallRecord, CallRole


@dataclass
class ClientResponse:
    output: Any
    call_record: CallRecord
    raw: Any = None


class OptimizerClient(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        ...


class TargetClient(Protocol):
    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        ...


class ReplayOptimizerClient:
    """Deterministic optimizer client for tests and cached experiments."""

    def __init__(self, outputs: Sequence[Any], model: str = "replay-optimizer"):
        self.outputs = list(outputs)
        self.model = model
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        if self.calls >= len(self.outputs):
            raise RuntimeError("ReplayOptimizerClient has no remaining outputs")
        started_at = datetime.now(timezone.utc)
        output = self.outputs[self.calls]
        self.calls += 1
        content = output if isinstance(output, str) else json.dumps(output)
        prompt_text = "\n".join(message.get("content", "") for message in messages)
        record = CallRecord(
            role=CallRole.OPTIMIZER,
            name="optimizer_compile",
            model=self.model,
            prompt_tokens=estimate_token_count(prompt_text),
            completion_tokens=estimate_token_count(content),
            cached=True,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )
        return ClientResponse(output=content, call_record=record, raw=output)


class ReplayTargetClient:
    """Deterministic target client for local tests and cached canary replay."""

    def __init__(self, outputs: Sequence[Any], model: str = "replay-target"):
        self.outputs = list(outputs)
        self.model = model
        self.calls = 0

    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        if self.calls >= len(self.outputs):
            raise RuntimeError("ReplayTargetClient has no remaining outputs")
        started_at = datetime.now(timezone.utc)
        output = self.outputs[self.calls]
        self.calls += 1
        payload = json.dumps({"prompt": prompt, "input": input}, sort_keys=True)
        output_text = output if isinstance(output, str) else json.dumps(output)
        record = CallRecord(
            role=CallRole.TARGET,
            name="target_canary",
            model=self.model,
            prompt_tokens=estimate_token_count(payload),
            completion_tokens=estimate_token_count(output_text),
            cached=True,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )
        return ClientResponse(output=output, call_record=record, raw=output)


class LiteLLMOptimizerClient:
    """Optimizer client backed by LiteLLM's async completion API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        extra_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.extra_kwargs = dict(extra_kwargs or {})

    async def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        response_schema: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLMOptimizerClient requires `pip install pact-el[litellm]`."
            ) from exc

        kwargs: Dict[str, Any] = dict(self.extra_kwargs)
        generated_response_format: Optional[Dict[str, Any]] = None
        used_json_schema_fallback = False
        if response_schema is not None and "response_format" not in kwargs:
            generated_response_format = _response_format_for_schema(response_schema)
            kwargs["response_format"] = generated_response_format

        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=list(messages),
                temperature=self.temperature,
                **kwargs,
            )
        except Exception as exc:
            if not (
                generated_response_format is not None
                and generated_response_format.get("type") == "json_schema"
                and _is_response_schema_error(exc)
            ):
                raise
            kwargs = dict(kwargs)
            kwargs["response_format"] = {"type": "json_object"}
            used_json_schema_fallback = True
            response = await litellm.acompletion(
                model=self.model,
                messages=list(messages),
                temperature=self.temperature,
                **kwargs,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        message = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        record_metadata = {**dict(metadata or {}), "latency_ms": elapsed_ms}
        if used_json_schema_fallback:
            record_metadata["response_schema_fallback"] = "json_object"
        record = CallRecord(
            role=CallRole.OPTIMIZER,
            name="optimizer_compile",
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(response.get("_hidden_params", {}).get("response_cost") or 0.0),
            cached=False,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            metadata=record_metadata,
        )
        return ClientResponse(output=message, call_record=record, raw=response)


def _response_format_for_schema(response_schema: Mapping[str, Any]) -> Dict[str, Any]:
    if _has_open_object_schema(response_schema):
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pact_el_compiler_output",
            "schema": _openai_strict_response_schema(response_schema),
            "strict": True,
        },
    }


def _openai_strict_response_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a closed JSON schema for OpenAI strict structured outputs."""

    normalized = _normalize_openai_strict_node(schema)
    if not isinstance(normalized, dict):
        raise TypeError("response schema must normalize to a JSON object")
    return normalized


def _normalize_openai_strict_node(schema: Any) -> Any:
    if isinstance(schema, Mapping):
        normalized: Dict[str, Any] = {}
        for key, value in schema.items():
            if key == "default":
                continue
            normalized[key] = _normalize_openai_strict_node(value)
        properties = normalized.get("properties")
        if (
            normalized.get("type") == "object"
            and isinstance(properties, Mapping)
            and normalized.get("additionalProperties") is False
        ):
            normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(schema, list):
        return [_normalize_openai_strict_node(value) for value in schema]
    return schema


def _has_open_object_schema(schema: Any) -> bool:
    if isinstance(schema, Mapping):
        if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
            return True
        return any(_has_open_object_schema(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_has_open_object_schema(value) for value in schema)
    return False


def _is_response_schema_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message and "schema" in message


class LiteLLMTargetClient:
    """Target client backed by LiteLLM's async completion API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        system_prompt: str = "",
        extra_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.extra_kwargs = dict(extra_kwargs or {})

    async def complete(
        self,
        prompt: str,
        input: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ClientResponse:
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "LiteLLMTargetClient requires `pip install pact-el[litellm]`."
            ) from exc

        input_text = input if isinstance(input, str) else json.dumps(input, sort_keys=True)
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text},
            ]
        )
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **self.extra_kwargs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        message = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        record = CallRecord(
            role=CallRole.TARGET,
            name="target_canary",
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(response.get("_hidden_params", {}).get("response_cost") or 0.0),
            cached=False,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            metadata={**dict(metadata or {}), "latency_ms": elapsed_ms},
        )
        return ClientResponse(output=message, call_record=record, raw=response)
