"""Prompt Axiom Graph utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

from pact_el.schemas import AxiomEdge, AxiomEdgeType, AxiomNode, AxiomNodeType, PromptAxiomGraph


RENDER_PRECEDENCE_EDGES = {
    AxiomEdgeType.DEPENDS_ON,
    AxiomEdgeType.OVERRIDES,
    AxiomEdgeType.SUPPORTS,
}


class PromptAxiomGraphIndex:
    """Index and traversal helper for a strict PromptAxiomGraph model."""

    def __init__(self, graph: PromptAxiomGraph):
        self.graph = graph
        self._nodes_by_id: Dict[str, AxiomNode] = {
            node.node_id: node for node in graph.nodes
        }
        self._edges_by_id: Dict[str, AxiomEdge] = {
            edge.edge_id: edge for edge in graph.edges
        }

    @property
    def nodes_by_id(self) -> Dict[str, AxiomNode]:
        return dict(self._nodes_by_id)

    @property
    def edges_by_id(self) -> Dict[str, AxiomEdge]:
        return dict(self._edges_by_id)

    def node(self, node_id: str) -> AxiomNode:
        return self._nodes_by_id[node_id]

    def nodes_by_type(self, node_type: AxiomNodeType) -> List[AxiomNode]:
        return [
            node
            for node in self.graph.nodes
            if node.node_type == node_type
        ]

    def outgoing(self, node_id: str) -> List[AxiomEdge]:
        return [edge for edge in self.graph.edges if edge.source_id == node_id]

    def incoming(self, node_id: str) -> List[AxiomEdge]:
        return [edge for edge in self.graph.edges if edge.target_id == node_id]

    def contradiction_edges(self) -> List[AxiomEdge]:
        return [
            edge
            for edge in self.graph.edges
            if edge.edge_type == AxiomEdgeType.CONTRADICTS
        ]

    def duplicate_edges(self) -> List[AxiomEdge]:
        return [
            edge
            for edge in self.graph.edges
            if edge.edge_type == AxiomEdgeType.DUPLICATES
        ]

    def regression_risk_edges(self) -> List[AxiomEdge]:
        return [
            edge
            for edge in self.graph.edges
            if edge.edge_type == AxiomEdgeType.RISKS_REGRESSION_IN
        ]

    def to_networkx(self) -> nx.MultiDiGraph:
        nx_graph = nx.MultiDiGraph()
        for node in self.graph.nodes:
            nx_graph.add_node(node.node_id, node=node)
        for edge in self.graph.edges:
            nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.edge_id,
                edge=edge,
                edge_type=edge.edge_type.value,
            )
        return nx_graph

    def precedence_graph(self) -> nx.DiGraph:
        """Return a DAG-like ordering graph for deterministic rendering.

        Direction means "render predecessor before successor".
        """

        order_graph = nx.DiGraph()
        for node in self.graph.nodes:
            order_graph.add_node(node.node_id, priority=node.priority)

        for edge in self.graph.edges:
            if edge.edge_type not in RENDER_PRECEDENCE_EDGES:
                continue
            if edge.edge_type == AxiomEdgeType.DEPENDS_ON:
                order_graph.add_edge(edge.target_id, edge.source_id)
            elif edge.edge_type == AxiomEdgeType.OVERRIDES:
                order_graph.add_edge(edge.source_id, edge.target_id)
            elif edge.edge_type == AxiomEdgeType.SUPPORTS:
                order_graph.add_edge(edge.source_id, edge.target_id)
        return order_graph

    def ordered_nodes(self) -> List[AxiomNode]:
        """Order graph nodes by precedence edges and stable priority fallback."""

        order_graph = self.precedence_graph()
        priority = {node.node_id: node.priority for node in self.graph.nodes}
        try:
            generations = list(nx.topological_generations(order_graph))
            ordered_ids: List[str] = []
            for generation in generations:
                ordered_ids.extend(
                    sorted(generation, key=lambda node_id: (priority[node_id], node_id))
                )
            return [self._nodes_by_id[node_id] for node_id in ordered_ids]
        except nx.NetworkXUnfeasible:
            return sorted(
                self.graph.nodes,
                key=lambda node: (node.priority, node.node_type.value, node.node_id),
            )

    def grouped_ordered_nodes(self) -> List[Tuple[AxiomNodeType, List[AxiomNode]]]:
        grouped: Dict[AxiomNodeType, List[AxiomNode]] = defaultdict(list)
        for node in self.ordered_nodes():
            grouped[node.node_type].append(node)
        return [
            (node_type, grouped[node_type])
            for node_type in AXIOM_RENDER_GROUP_ORDER
            if grouped.get(node_type)
        ]

    def reachable(self, start_id: str, edge_types: Sequence[AxiomEdgeType]) -> List[str]:
        edge_type_values = {edge_type.value for edge_type in edge_types}
        nx_graph = self.to_networkx()
        projected = nx.DiGraph()
        projected.add_nodes_from(nx_graph.nodes)
        for source, target, payload in nx_graph.edges(data=True):
            if payload.get("edge_type") in edge_type_values:
                projected.add_edge(source, target)
        if start_id not in projected:
            return []
        return sorted(nx.descendants(projected, start_id))


AXIOM_RENDER_GROUP_ORDER: List[AxiomNodeType] = [
    AxiomNodeType.PRECEDENCE_RULE,
    AxiomNodeType.TASK_INTENT,
    AxiomNodeType.OUTPUT_SCHEMA,
    AxiomNodeType.DECISION_RULE,
    AxiomNodeType.REASONING_POLICY,
    AxiomNodeType.TOOL_CONTRACT,
    AxiomNodeType.RETRIEVAL_CONTRACT,
    AxiomNodeType.SAFETY_BOUNDARY,
    AxiomNodeType.UNCERTAINTY_POLICY,
    AxiomNodeType.FORMATTING_RULE,
    AxiomNodeType.DEMONSTRATION_ROLE,
    AxiomNodeType.ANTI_PATTERN,
    AxiomNodeType.MODEL_CEILING_NOTICE,
]


def validate_graph_for_render(graph: PromptAxiomGraph) -> List[str]:
    """Return render warnings that should appear in reports, not hard errors."""

    warnings: List[str] = []
    index = PromptAxiomGraphIndex(graph)
    contradiction_edges = index.contradiction_edges()
    if contradiction_edges:
        warnings.append(
            "Graph contains contradiction edges: "
            + ", ".join(edge.edge_id for edge in contradiction_edges)
        )
    try:
        nx.find_cycle(index.precedence_graph())
    except nx.exception.NetworkXNoCycle:
        pass
    else:
        warnings.append("Precedence edges contain a cycle; renderer used priority fallback.")
    return warnings


def apply_patch_metadata(
    graph: PromptAxiomGraph,
    warnings: Iterable[str],
) -> PromptAxiomGraph:
    """Attach non-destructive render metadata to a graph copy."""

    copied = graph.model_copy(deep=True)
    copied.metadata = dict(copied.metadata)
    copied.metadata["render_warnings"] = list(warnings)
    return copied

