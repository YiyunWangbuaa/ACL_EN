# -*- coding: utf-8 -*-
"""Terminal results used for counterfactual contribution.

An expression with no descendants is a T_final. There are 17. Depth is the
graph's longest dependency. ID178 stays in the set: it is the roll member of
the elastic-compensation family, at depth 2.

A repair restores one locally attributed node to the manuscript gold and
recomputes calculators on the path to the target. Other local faults keep the
model text. Propagation nodes on that path are not frozen, so the repair can
reach the target. Restoring the target itself is a direct hit, not a recompute.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from cloze_score import sympy_equal
from domains.repair_search import order_grade, smallest_repairs
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.multi_slot import Assignment, answer_under
from domains.wind_tunnel.slot_signature import _has_calculator, _manuscript

_OBS = Path(__file__).resolve().parents[2] / "results" / "contract_consistency" / "error_workflow"
_LOCAL = (DIR, RSN, EXE, ALG)

# Process, then the expression sinks of that process. The audit checks this
# against the graph rather than trusting the tuple alone.
T_FINAL: dict[str, tuple[str, ...]] = {
    "风轴系到体轴系": ("80", "81"),
    "天平系到风轴系": ("176", "177", "178"),
    "天平系到体轴系": ("111", "112", "113", "114", "115", "116"),
    "力矩参考点": ("195", "196", "197", "198", "199", "200"),
}


def final_nodes() -> tuple[str, ...]:
    return tuple(node for group in T_FINAL.values() for node in group)


def _bare(value: str) -> str:
    return str(value).removeprefix("ID")


def _same(left: str, right: str) -> bool:
    if left.strip() == right.strip():
        return True
    try:
        return bool(sympy_equal(left, right))
    except Exception:
        return False


def audit_finals() -> None:
    """The constant is exactly the expression sinks, each with a calculator."""
    sinks = {
        node.blank_id
        for node in NODE_BY_ID.values()
        if node.answer_type == "expression" and not node.descendants
    }
    listed = final_nodes()
    if set(listed) != sinks or len(listed) != len(sinks):
        raise RuntimeError(f"T_FINAL {listed} 不是表达式终点 {tuple(sorted(sinks, key=int))}。")
    seen = set()
    for process, nodes in T_FINAL.items():
        for node in nodes:
            if node in seen:
                raise RuntimeError(f"ID{node} 重复。")
            seen.add(node)
            item = NODE_BY_ID[node]
            if item.process != process:
                raise RuntimeError(f"ID{node} 属于 {item.process}，常量写成了 {process}。")
            if not _has_calculator(node):
                raise RuntimeError(f"ID{node} 没有计算器，不能作为重算终点。")


def _observations(sample: str, condition: str) -> dict[str, dict]:
    found = {}
    path = _OBS / f"{sample}_{condition}" / "observations.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[_bare(row["node_id"])] = row
    return found


def _local_faults(sample: str, condition: str) -> tuple[str, ...]:
    from domains.wind_tunnel.error_features import classify_wrongs

    verdicts = classify_wrongs(sample, condition)
    return tuple(node for node, hit in sorted(verdicts.items(), key=lambda item: int(item[0])) if hit.fault in _LOCAL)


def _repairs_of(repair: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(repair, str):
        return (_bare(repair),)
    return tuple(_bare(item) for item in repair)


def _refresh(repairs: tuple[str, ...], target: str, hold: set[str]) -> set[str]:
    """Calculators between the repaired nodes and the target. They must be recomputed."""
    if set(repairs) == {target}:
        return set()
    ancestors = set(NODE_BY_ID[target].ancestors)
    downstream: set[str] = set()
    for repair in repairs:
        if repair != target:
            downstream.update(NODE_BY_ID[repair].descendants)
    found = {
        node
        for node in downstream
        if node in ancestors and node not in hold and node not in repairs and _has_calculator(node)
    }
    found.add(target)
    found.difference_update(repairs)
    return found


def repair_assignment(
    sample: str,
    observed: dict[str, dict],
    repair: str | tuple[str, ...],
    target: str,
    hold: set[str],
) -> Assignment:
    """Model trajectory with every node in `repair` restored to gold."""
    repairs = _repairs_of(repair)
    target = _bare(target)
    gold = _manuscript(sample)
    name = "repair-" + "-".join(f"ID{node}" for node in repairs)
    if set(repairs) == {target}:
        return Assignment(((target, gold[target]),), name=name, kind="repair-direct")
    open_nodes = _refresh(repairs, target, hold)
    pairs = [(node, gold[node]) for node in repairs]
    for node, row in observed.items():
        if node in open_nodes or node in repairs:
            continue
        if not row.get("asked", True):
            continue
        model = str(row.get("model_answer") or "").strip()
        if not model or _same(model, gold.get(node, "")):
            continue
        pairs.append((node, model))
    pairs.sort(key=lambda item: int(item[0]))
    return Assignment(tuple(pairs), name=name, kind="repair")


@dataclass(frozen=True)
class RepairQuery:
    sample: str
    condition: str
    target: str
    repair: str
    direct: bool
    assignment: Assignment


def queries(sample: str, condition: str = "self-trunc") -> tuple[RepairQuery, ...]:
    """One query per wrong T_final and each local fault that reaches it."""
    observed = _observations(sample, condition)
    faults = _local_faults(sample, condition)
    found = []
    for target in final_nodes():
        row = observed.get(target)
        if row is None or row.get("correct"):
            continue
        for repair in faults:
            if repair != target and target not in NODE_BY_ID[repair].descendants:
                continue
            hold = set(faults) - {repair, target}
            assignment = repair_assignment(sample, observed, repair, target, hold)
            found.append(RepairQuery(
                sample, condition, target, repair, repair == target, assignment,
            ))
    return tuple(found)


def repaired_matches(query: RepairQuery) -> bool | None:
    """True when the repaired target matches gold. None when the calculator refuses."""
    gold = _manuscript(query.sample)[query.target]
    if query.direct:
        return True
    text = answer_under(query.target, query.assignment, query.sample)
    if not text:
        return None
    return _same(text, gold)


def _status(result: bool | None) -> str:
    if result is True:
        return "对"
    if result is None:
        return "拒绝"
    return "仍错"


@dataclass(frozen=True)
class TargetFix:
    """Minimal upstream repairs for one wrong terminal.

    Restoring the terminal itself is recorded in ``direct`` and is not a member
    of the upstream set. The search ceiling is every upstream local fault.
    Smaller sets are tried first. If none of them restores the terminal, the
    full upstream set is restored and recomputed.
    """

    sample: str
    target: str
    direct: bool
    upstream: tuple[str, ...]
    single: tuple[tuple[str, str], ...]
    pairs: tuple[tuple[tuple[str, str], str], ...]
    grade: str
    minimal: tuple[tuple[str, ...], ...]


def _outcome(
    sample: str,
    observed: dict[str, dict],
    repairs: tuple[str, ...],
    target: str,
    hold: set[str],
) -> bool | None:
    if set(repairs) == {target}:
        return True
    assignment = repair_assignment(sample, observed, repairs, target, hold)
    text = answer_under(target, assignment, sample)
    if not text:
        return None
    return _same(text, _manuscript(sample)[target])


def minimal_fixes(sample: str, condition: str = "self-trunc", log=None) -> tuple[TargetFix, ...]:
    """Smallest upstream fault sets that make each wrong terminal match gold.

    The ceiling is every upstream local fault of that terminal. Search grows
    from one node to the full set, and stops at the smallest size that
    restores gold. The full set is the last attempt, so a terminal is not
    left open merely because more than two nodes are required.
    """
    observed = _observations(sample, condition)
    faults = _local_faults(sample, condition)
    found = []
    for target in final_nodes():
        row = observed.get(target)
        if row is None or row.get("correct"):
            continue
        upstream = tuple(
            fault for fault in faults
            if fault != target and target in NODE_BY_ID[fault].descendants
        )
        if log is not None:
            log(f"{sample} ID{target} 上游 {len(upstream)} 处")
        single_rows: list[tuple[str, str]] = []
        collected: list[tuple[tuple[str, str], str]] = []

        def restores(combo: tuple[str, ...]) -> bool:
            if log is not None:
                shown = "+".join(f"ID{node}" for node in combo)
                log(f"{sample} ID{target} {len(combo)} 元 {shown}")
            hold = set(faults) - set(combo) - {target}
            result = _outcome(sample, observed, combo, target, hold)
            if len(combo) == 1:
                single_rows.append((combo[0], _status(result)))
            elif len(combo) == 2:
                collected.append((combo, _status(result)))
            return result is True

        minimal = smallest_repairs(upstream, restores)
        pair_rows = tuple(collected)
        if minimal:
            grade = order_grade(len(minimal[0]))
        elif not upstream:
            grade = "无上游"
        elif len(upstream) == 1:
            grade = "单修失败"
        else:
            grade = "全上游仍失败"
        found.append(TargetFix(
            sample, target, target in faults, upstream,
            tuple(single_rows), pair_rows, grade, minimal,
        ))
    return tuple(found)


@dataclass(frozen=True)
class LayerCount:
    order: int
    refused: int
    wrong: int
    fixed: int
    opened: tuple[tuple[str, ...], ...]
    solved: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class OrderScan:
    """How many upstream faults must be restored before a terminal can be recomputed."""

    sample: str
    target: str
    upstream: tuple[str, ...]
    layers: tuple[LayerCount, ...]
    open_at: int | None
    fixed_at: int | None


def scan_orders(
    sample: str,
    targets: tuple[str, ...] = ("80", "81"),
    start: int = 3,
    condition: str = "self-trunc",
    log=None,
) -> tuple[OrderScan, ...]:
    """Raise the repair size until the calculator answers, then until the answer matches gold.

    Orders below ``start`` are not repeated. For the angle terminals those orders
    already refused in full. The search stops once a repair set matches gold.
    """
    observed = _observations(sample, condition)
    faults = _local_faults(sample, condition)
    found = []
    for target in targets:
        upstream = tuple(
            fault for fault in faults
            if fault != target and target in NODE_BY_ID[fault].descendants
        )
        if log is not None:
            log(f"{sample} ID{target} 扫描上游 {'、'.join(f'ID{node}' for node in upstream)}")
        layers = []
        open_at = None
        fixed_at = None
        for order in range(start, len(upstream) + 1):
            refused = wrong = fixed = 0
            opened: list[tuple[str, ...]] = []
            solved: list[tuple[str, ...]] = []
            combos = list(itertools.combinations(upstream, order))
            if log is not None:
                log(f"{sample} ID{target} {order} 元，共 {len(combos)} 组")
            for index, combo in enumerate(combos, start=1):
                if log is not None:
                    shown = "+".join(f"ID{node}" for node in combo)
                    log(f"{sample} ID{target} {order} 元 {index}/{len(combos)} {shown}")
                hold = set(faults) - set(combo) - {target}
                result = _outcome(sample, observed, combo, target, hold)
                if result is None:
                    refused += 1
                elif result:
                    fixed += 1
                    if len(solved) < 8:
                        solved.append(combo)
                else:
                    wrong += 1
                    if len(opened) < 8:
                        opened.append(combo)
            layers.append(LayerCount(order, refused, wrong, fixed, tuple(opened), tuple(solved)))
            if open_at is None and wrong + fixed:
                open_at = order
            if fixed:
                fixed_at = order
                break
        found.append(OrderScan(sample, target, upstream, tuple(layers), open_at, fixed_at))
    return tuple(found)


def audit_repair_shape() -> None:
    """Body ID80, repair ID47: the path recomputes, and the sibling local fault stays."""
    sample = "body-rule"
    observed = _observations(sample, "self-trunc")
    faults = set(_local_faults(sample, "self-trunc"))
    if "47" not in faults or "48" not in faults:
        raise RuntimeError(f"机体自答应把 ID47、ID48 记为本地故障，实际 {sorted(faults, key=int)}。")
    hold = faults - {"47", "80"}
    assignment = repair_assignment(sample, observed, "47", "80", hold)
    slots = {node: text for node, text in assignment.slots}
    gold = _manuscript(sample)
    if slots.get("47") != gold["47"]:
        raise RuntimeError("修复 ID47 时应写回金标。")
    for node in ("55", "59", "60", "80"):
        if node in slots:
            raise RuntimeError(f"ID{node} 在修复路径上，不应冻住模型原文。")
    if "48" not in slots or _same(slots["48"], gold["48"]):
        raise RuntimeError("ID48 是另一处本地错，应保持模型原文。")
    joint = repair_assignment(sample, observed, ("47", "48"), "80", faults - {"47", "48", "80"})
    joint_slots = {node: text for node, text in joint.slots}
    if joint_slots.get("47") != gold["47"] or joint_slots.get("48") != gold["48"]:
        raise RuntimeError("联合修复 ID47 与 ID48 时应同时写回金标。")
    for node in ("55", "59", "60", "80"):
        if node in joint_slots:
            raise RuntimeError(f"联合修复时 ID{node} 仍应留给计算器。")
    if "2" not in joint_slots or _same(joint_slots["2"], gold["2"]):
        raise RuntimeError("联合修复 ID47 与 ID48 时，ID2 应保持模型原文。")
