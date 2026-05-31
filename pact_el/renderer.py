"""Deterministic prompt renderer for Prompt Axiom Graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from pact_el.graph import PromptAxiomGraphIndex, validate_graph_for_render
from pact_el.schemas import (
    AxiomEdge,
    AxiomEdgeType,
    AxiomNode,
    AxiomNodeType,
    AxiomPatch,
    GuaranteeClause,
    PromptAxiomGraph,
)


NODE_TYPE_LABELS: Dict[AxiomNodeType, str] = {
    AxiomNodeType.TASK_INTENT: "Task Intent",
    AxiomNodeType.OUTPUT_SCHEMA: "Output Schema",
    AxiomNodeType.DECISION_RULE: "Decision Rules",
    AxiomNodeType.REASONING_POLICY: "Reasoning Policy",
    AxiomNodeType.TOOL_CONTRACT: "Tool Contract",
    AxiomNodeType.RETRIEVAL_CONTRACT: "Retrieval Contract",
    AxiomNodeType.SAFETY_BOUNDARY: "Safety Boundary",
    AxiomNodeType.UNCERTAINTY_POLICY: "Uncertainty Policy",
    AxiomNodeType.FORMATTING_RULE: "Formatting Rules",
    AxiomNodeType.PRECEDENCE_RULE: "Precedence Rules",
    AxiomNodeType.DEMONSTRATION_ROLE: "Demonstration Roles",
    AxiomNodeType.ANTI_PATTERN: "Anti-Patterns",
    AxiomNodeType.MODEL_CEILING_NOTICE: "Model Ceiling Notices",
}


@dataclass(frozen=True)
class RenderSettings:
    include_source_prompt: bool = True
    include_rationales: bool = False
    include_graph_edges: bool = True
    include_patch_summary: bool = True
    max_source_prompt_chars: int = 6000


def render_prompt(
    graph: PromptAxiomGraph,
    source_prompt: str = "",
    patch: Optional[AxiomPatch] = None,
    settings: Optional[RenderSettings] = None,
) -> str:
    """Render a final prompt from graph nodes, edges, and GuaranteeScript.

    The optimizer may propose a rendered prompt for diagnostics, but the runtime
    prompt used by PACT-EL is generated here to prevent hidden freeform rewrites.
    """

    settings = settings or RenderSettings()
    index = PromptAxiomGraphIndex(graph)
    warnings = validate_graph_for_render(graph)
    context_lines = [
        "This is a compiled PACT-EL behavioral contract. Follow higher-priority graph clauses over lower-priority prose when they conflict.",
    ]

    if settings.include_patch_summary and patch is not None:
        context_lines.extend(
            [
                f"Active patch id: {patch.patch_id}.",
                f"Patch summary: {patch.summary}",
                f"Rollback rule: {patch.rollback_rule}",
            ]
        )

    if warnings:
        for warning in warnings:
            context_lines.append(f"Graph render warning: {warning}")

    input_lines = [
        "At runtime, use the user's message as the raw input to solve.",
    ]
    if settings.include_source_prompt and source_prompt.strip():
        clipped = source_prompt.strip()[: settings.max_source_prompt_chars]
        if len(source_prompt.strip()) > settings.max_source_prompt_chars:
            clipped += "\n[Source prompt clipped by renderer.]"
        input_lines.extend(
            [
                "Soft guidance from the source prompt follows. Use it only where it does not conflict with this structured contract.",
                clipped,
            ]
        )

    task_lines = [
        "Apply the behavioral contract to the user's input.",
        "Draft an answer that satisfies the task intent and all applicable constraints.",
        "Privately verify the answer against the executable guarantees before finalizing.",
    ]

    constraints_lines: List[str] = []
    for node_type, nodes in index.grouped_ordered_nodes():
        constraints_lines.append(f"{NODE_TYPE_LABELS[node_type]}:")
        for node in nodes:
            constraints_lines.extend(_render_node(node, graph, settings))

    if settings.include_graph_edges and graph.edges:
        constraints_lines.extend(_render_edges(graph.edges, index))

    if graph.guarantee_script.clauses:
        constraints_lines.extend(_render_guarantee_section(graph.guarantee_script.clauses))
        constraints_lines.extend(_render_guarantee_script_block(graph))

    output_format_lines = [
        "Return only the final answer requested by the user.",
        "Do not expose private reasoning, checklists, graph metadata, or GuaranteeScript.",
        "Follow any output schema, formatting rule, or user-specified surface form exactly.",
    ]

    quality_bar_lines = [
        "Draft the answer.",
        "Interpret the GuaranteeScript over the candidate answer using input, output_text, and parsed_output when JSON is present.",
        "Revise until every applicable guarantee clause evaluates true.",
        "If a clause cannot be satisfied because the task lacks data, return the allowed uncertainty or refusal behavior from the contract instead of inventing facts.",
    ]

    sections = [
        ("Goal", ["Satisfy the user's task while obeying the compiled behavioral contract."]),
        ("Context", context_lines),
        ("Role", ["Act as a precise benchmark-solving assistant bound by the PACT-EL contract."]),
        ("Input", input_lines),
        ("Task", task_lines),
        ("Constraints", constraints_lines or ["No additional graph constraints were supplied."]),
        ("Output Format", output_format_lines),
        ("Quality Bar", quality_bar_lines),
    ]
    lines = _render_structured_sections(sections)

    return "\n".join(lines).strip() + "\n"


def estimate_token_count(text: str) -> int:
    """Cheap deterministic token estimate for call-ledger and patch accounting."""

    return max(1, round(len(text) / 4))


def estimate_token_delta(original_prompt: str, rendered_prompt: str) -> int:
    return estimate_token_count(rendered_prompt) - estimate_token_count(original_prompt)


def _render_node(
    node: AxiomNode,
    graph: PromptAxiomGraph,
    settings: RenderSettings,
) -> List[str]:
    prefix = "MUST" if node.required else "SHOULD"
    lines = [
        f"- [{prefix}; priority={node.priority}; id={node.node_id}] {node.title}: {node.statement}"
    ]
    if settings.include_rationales and node.rationale:
        lines.append(f"  Rationale: {node.rationale}")
    if node.guarantee_ids:
        descriptions = {
            clause.guarantee_id: clause.description
            for clause in graph.guarantee_script.clauses
        }
        for guarantee_id in node.guarantee_ids:
            description = descriptions.get(guarantee_id, "missing guarantee")
            lines.append(f"  Guarantee {guarantee_id}: {description}")
    return lines


def _render_edges(
    edges: List[AxiomEdge],
    index: PromptAxiomGraphIndex,
) -> List[str]:
    lines = ["Contract Relations:"]
    for edge in sorted(edges, key=lambda item: (item.edge_type.value, item.edge_id)):
        source = index.node(edge.source_id)
        target = index.node(edge.target_id)
        verb = _edge_verb(edge.edge_type)
        text = f"- {source.node_id} {verb} {target.node_id}"
        if edge.rationale:
            text += f": {edge.rationale}"
        lines.append(text)
    return lines


def _edge_verb(edge_type: AxiomEdgeType) -> str:
    return {
        AxiomEdgeType.SUPPORTS: "supports",
        AxiomEdgeType.DEPENDS_ON: "depends on",
        AxiomEdgeType.OVERRIDES: "overrides",
        AxiomEdgeType.CONTRADICTS: "contradicts",
        AxiomEdgeType.WEAKENS: "weakens",
        AxiomEdgeType.DUPLICATES: "duplicates",
        AxiomEdgeType.RISKS_REGRESSION_IN: "risks regression in",
    }[edge_type]


def _render_guarantee_section(clauses: List[GuaranteeClause]) -> List[str]:
    lines = [
        "Executable Guarantees:",
        "- These hard clauses are represented again below as GuaranteeScript JSON.",
    ]
    for clause in clauses:
        applies_to = (
            f" Applies to: {', '.join(clause.applies_to)}."
            if clause.applies_to
            else ""
        )
        lines.append(
            f"- [{clause.severity.value}; id={clause.guarantee_id}] {clause.description}{applies_to}"
        )
        lines.append(f"  Violation: {clause.violation_message}")
    return lines


def _render_guarantee_script_block(graph: PromptAxiomGraph) -> List[str]:
    payload = graph.guarantee_script.model_dump(mode="json")
    script = json.dumps(payload, indent=2, sort_keys=True)
    return [
        "<!-- PACT_EL_GUARANTEE_SCRIPT",
        script,
        "PACT_EL_GUARANTEE_SCRIPT -->",
    ]


def _render_structured_sections(
    sections: List[tuple[str, List[str]]],
) -> List[str]:
    lines: List[str] = []
    for title, items in sections:
        lines.append(f"## {title}")
        for item in items:
            if not item:
                continue
            if (
                item.startswith("- ")
                or item.startswith("<!--")
                or item.startswith("PACT_EL_GUARANTEE_SCRIPT")
                or item.endswith(":")
            ):
                lines.append(item)
            elif "\n" in item:
                lines.append(item)
            else:
                lines.append(f"- {item}")
        lines.append("")
    return lines
