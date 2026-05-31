from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pact_el.clients import LiteLLMOptimizerClient


@pytest.mark.asyncio
async def test_optimizer_client_uses_json_object_for_open_schemas(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=fake_acompletion))
    client = LiteLLMOptimizerClient("openai/fake")

    await client.complete(
        [{"role": "user", "content": "Return JSON."}],
        response_schema={
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": True,
                }
            },
            "additionalProperties": False,
        },
    )

    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_optimizer_client_uses_strict_json_schema_for_closed_schemas(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": '{"answer": "ok"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(acompletion=fake_acompletion))
    client = LiteLLMOptimizerClient("openai/fake")
    response_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    await client.complete(
        [{"role": "user", "content": "Return JSON."}],
        response_schema=response_schema,
    )

    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "pact_el_compiler_output",
            "schema": response_schema,
            "strict": True,
        },
    }
