# -*- coding: utf-8 -*-
"""Smallest repair sets. The ceiling is every upstream node of the target.

All four domains call this search. A domain supplies the upstream nodes and
the predicate that restores a subset to gold and recomputes. The search grows
from one node to the full set and stops at the smallest size that works.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence


def order_grade(order: int) -> str:
    if order == 1:
        return "单因素"
    return f"{order}元"


def _sort_key(node: str) -> tuple[int, str]:
    digits = "".join(char for char in str(node) if char.isdigit())
    return (int(digits) if digits else 0, str(node))


def smallest_repairs(
    universe: Sequence[str],
    restores: Callable[[tuple[str, ...]], bool],
) -> tuple[tuple[str, ...], ...]:
    """Every winner at the smallest size. Empty when the full set still fails."""
    nodes = tuple(universe)
    if not nodes:
        return ()
    for order in range(1, len(nodes) + 1):
        winners = [combo for combo in combinations(nodes, order) if restores(combo)]
        if winners:
            return tuple(winners)
    return ()


def _ancestors(target: str, parents_of: Callable[[str], Sequence[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    pending = list(parents_of(target))
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(parents_of(node))
    return tuple(sorted(seen, key=_sort_key))


def _topo(nodes: Sequence[str], parents_of: Callable[[str], Sequence[str]]) -> list[str]:
    remaining = set(nodes)
    ordered: list[str] = []
    while remaining:
        ready = [node for node in remaining if not (set(parents_of(node)) & remaining)]
        if not ready:
            ready = [min(remaining, key=_sort_key)]
        ready.sort(key=_sort_key)
        for node in ready:
            ordered.append(node)
            remaining.remove(node)
    return ordered


def synthetic_minimal_fixes(
    target: str,
    *,
    parents_of: Callable[[str], Sequence[str]],
    answer_under: Callable[[str, dict[str, str]], str | None],
    equals: Callable[[str, str, str], bool],
    gold: dict[str, str],
    alt_of: Callable[[str], str],
) -> tuple[tuple[str, ...], ...]:
    """Design-audit kernel: wrong upstream declarations, derived nodes recomputed.

    The repair atoms are the target's upstream declarations, not only its
    direct parents. Derived upstream blanks follow φ. The last attempt puts
    every upstream declaration back to gold.
    """
    if not parents_of(target):
        return ((target,),)
    ancestors = _ancestors(target, parents_of)
    roots = tuple(node for node in ancestors if not parents_of(node))
    if not roots:
        roots = ancestors
    derived = [node for node in ancestors if parents_of(node)]
    order = _topo([*derived, target], parents_of)

    def restores(repair: tuple[str, ...]) -> bool:
        chosen = set(repair)
        assignment = dict(gold)
        for node in roots:
            assignment[node] = gold[node] if node in chosen else alt_of(node)
        for node in order:
            if node in chosen:
                assignment[node] = gold[node]
                continue
            got = answer_under(node, assignment)
            if got is None:
                return False
            assignment[node] = got
        return equals(target, assignment.get(target, ""), gold[target])

    return smallest_repairs(roots, restores)
