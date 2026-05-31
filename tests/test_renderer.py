from __future__ import annotations

from pact_el.renderer import render_prompt


def test_renderer_uses_graph_and_guarantee_script(sample_compiler_output):
    rendered = render_prompt(
        sample_compiler_output.graph,
        source_prompt="Classify tickets.",
        patch=sample_compiler_output.patch,
    )

    for heading in (
        "## Goal",
        "## Context",
        "## Role",
        "## Input",
        "## Task",
        "## Constraints",
        "## Output Format",
        "## Quality Bar",
    ):
        assert heading in rendered
    assert "id=task" in rendered
    assert "PACT_EL_GUARANTEE_SCRIPT" in rendered
    assert "Soft guidance from the source prompt" in rendered
    assert "Revise until every applicable guarantee clause evaluates true." in rendered
