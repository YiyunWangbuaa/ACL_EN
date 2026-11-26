# -*- coding: utf-8 -*-
"""Dependency graph for flight blanks 1–59."""
from __future__ import annotations

from dataclasses import dataclass

from domains.aerodynamics.contract_verifiers import input_ids
from domains.aerodynamics.manuscript import questions
from domains.aerodynamics.pipeline_scope import NODE_IDS


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    depends_on: tuple[str, ...]


def nodes() -> dict[str, Node]:
    rows = questions()
    found = {}
    for node in NODE_IDS:
        parents = input_ids(node)
        found[node] = Node(node, rows[node]["label"], parents)
    return found


def edges() -> tuple[tuple[str, str], ...]:
    found = []
    for node, item in nodes().items():
        for parent in item.depends_on:
            found.append((parent, node))
    return tuple(found)


def future_edges() -> tuple[tuple[str, str], ...]:
    """Parent asked after or with the child — forbidden."""
    bad = []
    for parent, child in edges():
        if int(parent) >= int(child):
            bad.append((parent, child))
    return tuple(bad)


def check_graph() -> None:
    graph = nodes()
    if set(graph) != set(NODE_IDS) or len(graph) != 131:
        raise RuntimeError(f"Expected 131 nodes, got {len(graph)}")
    for parent, child in edges():
        if parent not in graph or child not in graph:
            raise RuntimeError(f"Edge {parent}->{child} points at a missing node")
    bad = future_edges()
    if bad:
        raise RuntimeError(f"An edge runs from a later sentence to an earlier one: {bad}")
    print(f"Stage 1 dependency graph: {len(graph)} nodes, {len(edges())} edges, 0 future edges.", flush=True)


def main() -> None:
    check_graph()


if __name__ == "__main__":
    main()
