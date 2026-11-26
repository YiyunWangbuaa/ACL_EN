# -*- coding: utf-8 -*-
"""Dependency graph for vision basic 1–66."""
from __future__ import annotations

from dataclasses import dataclass

from domains.computer_vision.contract_verifiers import input_ids
from domains.computer_vision.manuscript import questions
from domains.computer_vision.pipeline_scope import NODE_IDS


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    depends_on: tuple[str, ...]


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
        if int(parent) >= int(child):
            bad.append((parent, child))
    return tuple(bad)


def check_graph() -> None:
    graph = nodes()
    if set(graph) != set(NODE_IDS) or len(graph) != 66:
        raise RuntimeError(f"expected 66 nodes, got {len(graph)}")
    bad = future_edges()
    if bad:
        raise RuntimeError(f"an edge points from a later sentence to an earlier one: {bad}")
    print(f"stage 1 dependency graph: {len(graph)} nodes, {len(edges())} edges, 0 future edges.", flush=True)


if __name__ == "__main__":
    check_graph()
