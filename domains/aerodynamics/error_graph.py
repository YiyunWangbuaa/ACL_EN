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
        raise RuntimeError(f"节点数应为 131，得到 {len(graph)}")
    for parent, child in edges():
        if parent not in graph or child not in graph:
            raise RuntimeError(f"边 {parent}->{child} 指向缺失节点")
    bad = future_edges()
    if bad:
        raise RuntimeError(f"存在后句指向前句的边：{bad}")
    print(f"阶段1 依赖图：节点 {len(graph)}，边 {len(edges())}，未来边 0。", flush=True)


def main() -> None:
    check_graph()


if __name__ == "__main__":
    main()
