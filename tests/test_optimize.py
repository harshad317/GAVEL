from __future__ import annotations

import pytest

from pact_el.clients import ReplayOptimizerClient, ReplayTargetClient
from pact_el.optimize import pact_optimize
from pact_el.schemas import GateDecision


@pytest.mark.asyncio
async def test_pact_optimize_accepts_replay_success(sample_compiler_output):
    report = await pact_optimize(
        prompt="Classify tickets.",
        task_spec="Return JSON with priority.",
        rubric="Priority must be valid.",
        logs=[
            {
                "input": {"ticket": "prod outage"},
                "expected": "urgent",
                "observed": "not valid JSON",
                "passed": False,
                "severity": "high",
                "area": "output_schema",
                "prompt_fixability": 0.9,
            }
        ],
        optimizer_client=ReplayOptimizerClient(
            [sample_compiler_output.model_dump_json()]
        ),
        target_client=ReplayTargetClient(
            [
                '{"priority": "urgent"}',
                '{"priority": "medium"}',
                '{"priority": "low"}',
                '{"priority": "high"}',
            ]
        ),
    )

    assert report.accepted
    assert report.decision == GateDecision.ACCEPTED
    assert report.call_ledger.optimizer_calls == 1
    assert report.call_ledger.target_calls == 4


@pytest.mark.asyncio
async def test_pact_optimize_returns_no_patch_when_repair_schema_is_invalid(
    sample_compiler_output,
):
    repair_payload = {
        "graph": sample_compiler_output.graph.model_dump(mode="json"),
        "patch": {
            **sample_compiler_output.patch.model_dump(mode="json"),
            "graph_edits": [
                {
                    "operation": "update_node",
                    "node_id": "task",
                }
            ],
        },
        "retest_canaries": [
            sample_compiler_output.canaries[0].model_dump(mode="json")
        ],
    }

    report = await pact_optimize(
        prompt="Classify tickets.",
        task_spec="Return JSON with priority.",
        rubric="Priority must be valid.",
        logs=[
            {
                "input": {"ticket": "prod outage"},
                "expected": "urgent",
                "observed": "not valid JSON",
                "passed": False,
                "severity": "high",
                "area": "output_schema",
                "prompt_fixability": 0.9,
            }
        ],
        optimizer_client=ReplayOptimizerClient(
            [sample_compiler_output.model_dump_json(), repair_payload]
        ),
        target_client=ReplayTargetClient(
            [
                '{"priority": "invalid"}',
                '{"priority": "invalid"}',
                '{"priority": "invalid"}',
                '{"priority": "invalid"}',
            ]
        ),
    )

    assert not report.accepted
    assert report.decision == GateDecision.REJECTED_NO_PATCH
    assert report.call_ledger.optimizer_calls == 2
    assert report.call_ledger.target_calls == 4
    assert report.no_patch_diagnosis is not None
    assert "RepairOutput schema" in report.no_patch_diagnosis.reason
    assert report.metadata["repair_schema_error"]["type"] == "ValidationError"
