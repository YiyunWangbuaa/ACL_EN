# -*- coding: utf-8 -*-
"""Dependency graph for robotics theory F*/I*."""
from __future__ import annotations

from dataclasses import dataclass

from domains.robotics.kin_contract_verifiers import input_ids
from domains.robotics.kin_manuscript import questions
from domains.robotics.kin_pipeline_scope import NODE_IDS
from domains.robotics.kin_signed_lists import DERIVED_NODES


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    depends_on: tuple[str, ...]


def _order_key(node: str) -> tuple[int, int]:
    # Forward before inverse; public id ascending within each.
    family = 0 if node.startswith("F") else 1
    return family, int(node[1:])


def nodes() -> dict[str, Node]:
    rows = questions()
    return {node: Node(node, rows[node]["label"], input_ids(node)) for node in NODE_IDS}


def edges() -> tuple[tuple[str, str], ...]:
    found = []
    for node, item in nodes().items():
        for parent in item.depends_on:
            found.append((parent, node))
    return tuple(found)


def future_edges() -> tuple[tuple[str, str], ...]:
    bad = []
    for parent, child in edges():
        if _order_key(parent) >= _order_key(child):
            bad.append((parent, child))
    return tuple(bad)


def check_graph() -> None:
    graph = nodes()
    if set(graph) != set(NODE_IDS) or len(graph) != len(NODE_IDS):
        raise RuntimeError(f"node count should be {len(NODE_IDS)}, got {len(graph)}")
    bad = future_edges()
    if bad:
        raise RuntimeError(f"there is an edge from a later sentence to an earlier one: {bad}")
    wired = {node for node in NODE_IDS if input_ids(node)}
    if wired != set(DERIVED_NODES):
        missing = sorted(set(DERIVED_NODES) - wired)
        extra = sorted(wired - set(DERIVED_NODES))
        raise RuntimeError(f"dependency graph and derived nodes disagree: missing {missing} extra {extra}")
    print(
        f"stage 1 dependency graph: nodes {len(graph)}, edges {len(edges())}, "
        f"derived nodes {len(DERIVED_NODES)}, future edges 0.",
        flush=True,
    )


if __name__ == "__main__":
    check_graph()
