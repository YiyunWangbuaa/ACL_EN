# -*- coding: utf-8 -*-
"""Dependency graph for robotics theory F*/I*."""
from __future__ import annotations

from dataclasses import dataclass

from domains.robotics.contract_verifiers import input_ids
from domains.robotics.manuscript import questions
from domains.robotics.pipeline_scope import NODE_IDS


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
        raise RuntimeError(f"节点数应为 {len(NODE_IDS)}，得到 {len(graph)}")
    bad = future_edges()
    if bad:
        raise RuntimeError(f"存在后句指向前句的边：{bad}")
    print(f"阶段1 依赖图：节点 {len(graph)}，边 {len(edges())}，未来边 0。", flush=True)


if __name__ == "__main__":
    check_graph()
