from __future__ import annotations

from pact_el.renderer import render_prompt


def test_renderer_uses_graph_and_guarantee_script(sample_compiler_output):
    rendered = render_prompt(
        sample_compiler_output.graph,
        source_prompt="Classify tickets.",
        patch=sample_compiler_output.patch,
    )

    assert "# PACT-EL Behavioral Contract" in rendered
    assert "id=task" in rendered
    assert "PACT_EL_GUARANTEE_SCRIPT" in rendered
    assert "Soft Guidance From Source Prompt" in rendered
    assert "Required Self-Check Before Final Answer" in rendered

