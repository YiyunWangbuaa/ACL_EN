# -*- coding: utf-8 -*-
"""Multi-slot substitution: one forward operator for joint, chain, and branch.

A assignment writes a set of blanks into the manuscript texts. Calculators on
the path from those blanks to the target then run in topological order. Any
contradiction on that path makes the assignment unsatisfiable and the answer
is the empty string. Joint, chain, and branch differ only by where the slots
sit and by the name on the assignment.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cloze_score import normalize, sympy_equal
from domains.wind_tunnel.answer_form import match_predicate, parse_answer
from domains.wind_tunnel.error_graph import NODE_BY_ID, _DEFINITIONS
from domains.wind_tunnel.slot_signature import (
    _ATTITUDE,
    _execute,
    _has_calculator,
    _inputs,
    _kind,
    _manuscript,
    _model,
    _options,
    _outcome,
    _parent_alts,
    _rows,
    calculator_nodes,
    slots as single_slots,
)

_STRUCTURAL = {"matrix", "vector", "expression"}
_OBS = Path(__file__).resolve().parents[2] / "results" / "contract_consistency" / "error_workflow"
_AXIS_NAMES = {
    ("2", "3", "4"): ("XFD", "YFD", "ZFD"),
    ("5", "6", "7"): ("XTZ", "YTZ", "ZTZ"),
    ("89", "90", "91"): ("XTP", "YTP", "ZYP"),
    ("92", "93", "94"): ("XTZ", "YTZ", "ZTZ"),
    ("120", "121", "122"): ("XFD", "YFD", "ZFD"),
    ("123", "124", "125"): ("XTP", "YTP", "ZYP"),
}
_READS = {item["key"]: tuple(item["reads"]) for item in _DEFINITIONS}
_ID_IN_REASON = re.compile(r"ID(\d+)")
_FORM_CODES = ("P1", "P2", "P3", "P4", "P5")


@dataclass(frozen=True)
class Group:
    key: str
    slots: tuple[str, ...]
    constraint: str


@dataclass(frozen=True)
class Assignment:
    slots: tuple[tuple[str, str], ...]
    name: str = ""
    kind: str = "joint"


@dataclass(frozen=True)
class Explanation:
    sample: str
    target: str
    assignment: Assignment
    path: tuple[str, ...]
    answer: str


# Enumeration domains. 50 and 51 stay RP2 and RP1 for every consistent
# convention, so the free attitude-code domain is 47–49.
_CLUSTERS = (
    Group("G-axes-wind-body", ("2", "3", "4"), "orthonormal_triad"),
    Group("G-axes-body-wind", ("5", "6", "7"), "orthonormal_triad"),
    Group("G-axes-balance", ("89", "90", "91"), "orthonormal_triad"),
    Group("G-axes-body-balance", ("92", "93", "94"), "orthonormal_triad"),
    Group("G-axes-wind-fd", ("120", "121", "122"), "orthonormal_triad"),
    Group("G-axes-balance-wind", ("123", "124", "125"), "orthonormal_triad"),
    Group("G-order-attitude", ("16", "17", "18"), "permutation"),
    Group("G-codes-attitude", ("47", "48", "49"), "permutation"),
    Group("G-codes-load", ("98", "99", "100"), "permutation"),
    Group("G-bind-body", ("105", "106", "107"), "binding"),
    Group("G-bind-balance", ("108", "109", "110"), "binding"),
    Group("G-moment-path", ("185", "186", "187", "188"), "path_order"),
)
_BRANCH_SLOTS = {
    "frame_anchor": ("23", "31", "40"),
    "transform_sense": ("44", "45", "46"),
    "rotation_type": ("44", "45", "46"),
    "angle_order": ("16", "17", "18"),
    "mount_basis": ("83", "84", "96"),
}
_BRANCH_MODALITY = {"angle_order": "implicit"}


def catalog() -> tuple[Group, ...]:
    """Named constraint families. Manuscript-independent."""
    return _CLUSTERS


def groups(node_id: str) -> tuple[Group, ...]:
    """Minimal satisfiable groups covered by this node's direct inputs."""
    bare = str(node_id).removeprefix("ID")
    covered = set(_inputs(bare))
    return tuple(group for group in _minimal_groups("standard-default") if set(group.slots) <= covered)


def changed_definitions(sample_id: str) -> tuple[str, ...]:
    """Definition keys whose spec differs from standard-default."""
    spec = _rows()[sample_id]["spec"]
    base = _rows()["standard-default"]["spec"]
    found = []
    if spec.get("alpha") != base.get("alpha"):
        found.append("mechanism_alpha")
    if spec.get("gamma_m") != base.get("gamma_m"):
        found.append("mechanism_roll")
    if spec.get("placement") != base.get("placement"):
        found.append("placement")
    frame = spec.get("body_frame") or {}
    base_frame = base.get("body_frame") or {}
    if frame.get("euler_order") != base_frame.get("euler_order"):
        found.append("euler_order")
    axes_differ = any(frame.get(key) != base_frame.get(key) for key in ("chirality", "XTZ", "YTZ", "ZTZ"))
    if axes_differ:
        found.append("body_handedness")
    return tuple(found)


def changed_reads(sample_id: str) -> tuple[str, ...]:
    found = []
    for key in changed_definitions(sample_id):
        for slot in _READS.get(key, ()):
            if slot not in found:
                found.append(slot)
    return tuple(found)


def chain_assignments(sample_id: str) -> tuple[Assignment, ...]:
    """One counterfactual pole per changed definition: its reads set to the standard text."""
    standard = _manuscript("standard-default")
    current = _manuscript(sample_id)
    found = []
    for key in changed_definitions(sample_id):
        reads = _READS.get(key, ())
        if not reads or all(standard.get(slot) == current.get(slot) for slot in reads):
            continue
        pairs = tuple((slot, standard.get(slot, "")) for slot in reads)
        found.append(Assignment(pairs, name=key, kind="chain"))
    return tuple(found)


def branch_table(sample_id: str = "standard-default") -> dict[str, Assignment]:
    """The counterfactual pole of each named branch."""
    current = _manuscript(sample_id)
    found = {}
    for name, pairs in _branch_poles(current):
        found[name] = Assignment(pairs, name=name, kind="branch")
    return found


def answer_under(target: str, assignment: Assignment, sample_id: str) -> str:
    """Answer of ``target`` under this assignment. Empty when the path contradicts."""
    bare = str(target).removeprefix("ID")
    answers, failed, _path = _recompute(sample_id, assignment.slots, (bare,))
    if bare in failed:
        return ""
    return answers.get(bare, "")


def answer_value(target: str, assignment: Assignment, sample_id: str):
    """T2 value of ``answer_under``. None when the assignment is unsatisfiable or unparsed."""
    text = answer_under(target, assignment, sample_id)
    if not text:
        return None
    bare = str(target).removeprefix("ID")
    kind = _kind(bare)
    options = _options().get(bare) if kind in {"choice", "sign"} else None
    form = parse_answer(text, kind, options)
    return form.value if form.status == "ok" else None


def subtree_signature(root: str, assignment: Assignment, nodes: tuple[str, ...] | list[str], sample_id: str) -> dict[str, str]:
    """Recomputed text of each requested node. Empty when a contradiction blocks it.

    T5 step 2 uses this for an RSN subtree check. An unnamed joint assignment
    that holds at one node only stays an EXE candidate (``value_form=wrong-order``).
    """
    del root
    wanted = tuple(str(node).removeprefix("ID") for node in nodes)
    answers, failed, _path = _recompute(sample_id, assignment.slots, wanted)
    return {node: "" if node in failed else answers.get(node, "") for node in wanted}


def assignments(node_id: str, sample_id: str) -> tuple[Assignment, ...]:
    """Satisfiable multi-slot assignments that can reach this node, plus chain and branch poles."""
    bare = str(node_id).removeprefix("ID")
    found = []
    seen = set()
    for group, combos in _combos_for(sample_id, bare):
        for pairs in combos:
            if pairs in seen:
                continue
            seen.add(pairs)
            found.append(Assignment(pairs, name=group.key, kind="joint"))
    for item in chain_assignments(sample_id):
        if _reaches(item.slots, bare) and item.slots not in seen:
            seen.add(item.slots)
            found.append(item)
    for item in branch_table(sample_id).values():
        if _reaches(item.slots, bare) and item.slots not in seen:
            seen.add(item.slots)
            found.append(item)
    return tuple(found)


def seed_from_reason(reason: str, node_id: str = "") -> str:
    """Text seed only. The satisfiability search, not this string, decides the group."""
    text = reason or ""
    if "cannot form a set with the other axes" in text:
        match = _ID_IN_REASON.search(text)
        slot = match.group(1) if match else ""
        for group in _CLUSTERS:
            if group.constraint == "orthonormal_triad" and slot in group.slots:
                return group.key
        return ""
    if "transform order is not" in text and "permutation" in text:
        return "G-order-attitude"
    if "moment transfer does not match the template" in text:
        return "G-moment-path"
    if "The six balance components do not each land on one coordinate axis." in text:
        covered = set(_inputs(str(node_id).removeprefix("ID"))) if node_id else set()
        for key in ("G-bind-balance", "G-bind-body"):
            group = next(item for item in _CLUSTERS if item.key == key)
            if not covered or set(group.slots) <= covered:
                return key
        return "G-bind-balance"
    return ""


def _explain_rows(sample_id: str, rows: tuple[dict, ...]) -> tuple[Explanation, ...]:
    found = []
    pending = {row["node"]: row for row in rows}
    for group in _CLUSTERS:
        if not pending:
            break
        consumer = _consumer(group.slots)
        if consumer is None:
            continue
        if group.constraint == "orthonormal_triad":
            reached = tuple(node for node in pending if set(group.slots) <= set(_inputs(node)))
        else:
            reached = tuple(node for node in pending if _slot_reaches(group.slots, node))
        if not reached:
            continue
        combos, _stats = _enumerated(sample_id, group)
        if not combos:
            continue
        if not reached:
            continue
        for pairs in combos:
            if not pending:
                break
            answers, failed, path = _recompute(sample_id, pairs, reached)
            assignment = Assignment(pairs, name=group.key, kind="joint")
            for node in list(pending):
                if node in failed:
                    continue
                answer = answers.get(node, "")
                row = pending[node]
                if _matches(node, answer, row["model"]) and not _matches(node, answer, row["gold"]):
                    target_path = tuple(item for item in path if item == node or item in _ancestors(node))
                    found.append(Explanation(sample_id, node, assignment, target_path or (node,), answer))
                    del pending[node]
    for item in chain_assignments(sample_id):
        _take_fixed(sample_id, item, pending, found)
    for item in branch_table(sample_id).values():
        _take_fixed(sample_id, item, pending, found)
    return tuple(found)


def _take_fixed(sample_id: str, assignment: Assignment, pending: dict, found: list) -> None:
    if not pending:
        return
    reached = tuple(node for node in pending if _slot_reaches(assignment.slots, node))
    if not reached:
        return
    answers, failed, path = _recompute(sample_id, assignment.slots, reached)
    for node in list(pending):
        if node not in reached or node in failed:
            continue
        answer = answers.get(node, "")
        row = pending[node]
        if _matches(node, answer, row["model"]) and not _matches(node, answer, row["gold"]):
            target_path = tuple(item for item in path if item == node or item in _ancestors(node))
            found.append(Explanation(sample_id, node, assignment, target_path or (node,), answer))
            del pending[node]


def _branch_poles(current: dict[str, str]) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    order = (("16", "sideslip"), ("17", "angle of attack"), ("18", "roll"))
    poles = (
        ("frame_anchor", tuple((slot, "wind frame") for slot in _BRANCH_SLOTS["frame_anchor"])),
        ("transform_sense", (("44", "active"), ("45", "moving axes"), ("46", "intrinsic"))),
        ("rotation_type", (("44", "passive"), ("45", "fixed axes"), ("46", "extrinsic"))),
        ("angle_order", order),
        (
            "mount_basis",
            (("83", current.get("83", "positive")), ("84", current.get("84", "positive")), ("96", "inverted")),
        ),
    )
    return tuple(poles)


def _minimal_groups(sample_id: str) -> tuple[Group, ...]:
    cached = _MINIMAL.get(sample_id)
    if cached is not None:
        return cached
    found = []
    for group in _CLUSTERS:
        consumer = _consumer(group.slots)
        if consumer is None or group.constraint == "orthonormal_triad":
            continue
        if _any_singleton(sample_id, consumer, group.slots):
            continue
        domains = _domains(sample_id, group.slots)
        for size in range(2, len(group.slots) + 1):
            for index in itertools.combinations(range(len(group.slots)), size):
                subset = tuple(group.slots[item] for item in index)
                if any(set(earlier.slots) < set(subset) for earlier in found if earlier.key.startswith(group.key)):
                    continue
                sub_domains = [domains[item] for item in index]
                if _domain_satisfiable(sample_id, consumer, subset, sub_domains):
                    found.append(Group(f"{group.key}/" + "-".join(subset), subset, group.constraint))
    published = tuple(found)
    _MINIMAL[sample_id] = published
    return published


_MINIMAL: dict[str, tuple[Group, ...]] = {}
_ENUM: dict[tuple[str, str], tuple] = {}
_RECOMPUTE: dict[tuple, tuple] = {}
_GRAPH: dict | None = None


def _graph() -> dict:
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    parents: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    for node_id, node in NODE_BY_ID.items():
        linked = set(node.depends_on)
        if _has_calculator(node_id):
            linked.update(_inputs(node_id))
        if node_id in _ATTITUDE:
            linked.add(_ATTITUDE[node_id][0])
        for parent in linked:
            parents[node_id].add(parent)
            children[parent].add(node_id)
    _GRAPH = {"parents": parents, "children": children}
    return _GRAPH


def _ancestors(node_id: str) -> set[str]:
    parents = _graph()["parents"]
    seen = set()
    stack = list(parents.get(node_id, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return seen


def _descendants(seeds: tuple[str, ...] | set[str]) -> set[str]:
    children = _graph()["children"]
    seen = set()
    stack = list(seeds)
    while stack:
        current = stack.pop()
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _reaches(pairs: tuple[tuple[str, str], ...], node_id: str) -> bool:
    return _slot_reaches(tuple(slot for slot, _value in pairs), node_id)


def _slot_reaches(slots: tuple[str, ...], node_id: str) -> bool:
    """A slot explains a target only as an input, never by overwriting the target."""
    if node_id in slots:
        return False
    ancestors = _ancestors(node_id)
    return any(slot in ancestors for slot in slots)


# Axis triads are judged by the alignment matrix that requires the triad,
# not by a later formula that merely reads the same three blanks.
_JUDGE = {
    ("2", "3", "4"): "8",
    ("5", "6", "7"): "8",
    ("89", "90", "91"): "95",
    ("92", "93", "94"): "95",
    ("120", "121", "122"): "126",
    ("123", "124", "125"): "126",
}


def _consumer(slots: tuple[str, ...]) -> str | None:
    pinned = _JUDGE.get(slots)
    if pinned is not None:
        return pinned
    want = set(slots)
    best = None
    best_key = None
    for node_id in calculator_nodes():
        inputs = _inputs(node_id)
        if not want <= set(inputs):
            continue
        key = (len(inputs), int(node_id))
        if best_key is None or key < best_key:
            best = node_id
            best_key = key
    return best


def _domains(sample_id: str, slots: tuple[str, ...]) -> list[tuple[str, ...]]:
    if slots in _AXIS_NAMES:
        names = _AXIS_NAMES[slots]
        signed = tuple(sign + name for name in names for sign in ("", "-"))
        return [signed for _slot in slots]
    current = _manuscript(sample_id)
    found = []
    for slot in slots:
        values = list(_options().get(slot) or ())
        if not values:
            values = [item.alt for item in _parent_alts(slot, sample_id, current.get(slot, ""))]
        if current.get(slot) and current[slot] not in values:
            values = [current[slot], *values]
        found.append(tuple(dict.fromkeys(values)))
    return found


def _accepted(sample_id: str, consumer: str, pairs: tuple[tuple[str, str], ...]) -> bool:
    texts = dict(_manuscript(sample_id))
    for slot, value in pairs:
        texts[slot] = value
    execution = _execute(_model(sample_id), consumer, texts)
    return execution is not None and bool(execution.ok)


def _any_singleton(sample_id: str, consumer: str, slots: tuple[str, ...]) -> bool:
    current = _manuscript(sample_id)
    domains = _domains(sample_id, slots)
    for slot, domain in zip(slots, domains):
        gold = current.get(slot, "")
        flipped = gold[1:] if gold.startswith("-") else f"-{gold}"
        candidates = []
        if flipped and flipped != gold:
            candidates.append(flipped)
        candidates.extend(value for value in domain if value != gold and value not in candidates)
        for value in candidates:
            if _accepted(sample_id, consumer, ((slot, value),)):
                return True
    return False


def _domain_satisfiable(sample_id: str, consumer: str, slots: tuple[str, ...], domains: list[tuple[str, ...]]) -> bool:
    current = _manuscript(sample_id)
    for combo in itertools.product(*domains):
        if all(combo[index] == current.get(slots[index], "") for index in range(len(slots))):
            continue
        pairs = tuple((slots[index], combo[index]) for index in range(len(slots)))
        if _accepted(sample_id, consumer, pairs):
            return True
    return False


def _enumerated(sample_id: str, group: Group) -> tuple[tuple[tuple[tuple[str, str], ...], ...], dict]:
    key = (sample_id, group.key)
    cached = _ENUM.get(key)
    if cached is not None:
        return cached
    consumer = _consumer(group.slots)
    stats = {"pair_checks": 0, "full_checks": 0, "accepted": 0, "domain": 0}
    if consumer is None:
        _ENUM[key] = ((), stats)
        return _ENUM[key]
    if group.constraint == "orthonormal_triad":
        combos, checks = _axis_combos(sample_id, consumer, group.slots)
        stats["domain"] = 48
        stats["full_checks"] = checks
        stats["accepted"] = len(combos)
        packed = tuple(combos)
        _ENUM[key] = (packed, stats)
        return _ENUM[key]
    domains = _domains(sample_id, group.slots)
    stats["domain"] = 1
    for domain in domains:
        stats["domain"] *= max(len(domain), 1)
    if stats["domain"] > 400 or any(not domain for domain in domains):
        _ENUM[key] = ((), stats)
        return _ENUM[key]
    # Products this small are judged entirely by _execute. Pairwise checks that
    # hold the other slots at gold drop legal permutations (two angles can swap
    # only when the third stays the remaining name).
    combos, full_checks = _complete(sample_id, consumer, group.slots, domains)
    stats["pair_checks"] = 0
    stats["full_checks"] = full_checks
    stats["accepted"] = len(combos)
    packed = tuple(combos)
    _ENUM[key] = (packed, stats)
    return _ENUM[key]


def _complete(sample_id: str, consumer: str, slots: tuple[str, ...], domains: list[tuple[str, ...]]):
    current = _manuscript(sample_id)
    found = []
    full_checks = 0
    for combo in itertools.product(*domains):
        changed = sum(
            1 for item, value in enumerate(combo) if normalize(value) != normalize(current.get(slots[item], ""))
        )
        if changed < 2:
            continue
        full_checks += 1
        pairs = tuple((slots[item], combo[item]) for item in range(len(slots)))
        if _accepted(sample_id, consumer, pairs):
            found.append(pairs)
        if len(found) >= 200:
            break
    return found, full_checks


def _axis_combos(sample_id: str, consumer: str, slots: tuple[str, ...]):
    current = _manuscript(sample_id)
    names = _AXIS_NAMES[slots]
    found = []
    checks = 0
    for perm in itertools.permutations(names):
        for signs in itertools.product(("", "-"), repeat=len(slots)):
            pairs = tuple((slot, sign + name) for slot, sign, name in zip(slots, signs, perm))
            changed = sum(1 for slot, value in pairs if normalize(value) != normalize(current.get(slot, "")))
            if changed < 2:
                continue
            checks += 1
            if _accepted(sample_id, consumer, pairs):
                found.append(pairs)
    return found, checks


def _combos_for(sample_id: str, node_id: str):
    found = []
    for group in _CLUSTERS:
        if not _slot_reaches(group.slots, node_id):
            continue
        if group.constraint == "orthonormal_triad" and set(group.slots) - set(_inputs(node_id)):
            # Axis campaigns stay on the consumer. Downstream of an alignment
            # matrix is not re-enumerated here; the code campaigns cover the
            # targets whose model answer is a recomputed product.
            continue
        combos, _stats = _enumerated(sample_id, group)
        if combos:
            found.append((group, combos))
    return found


def _execute_node(sample_id: str, node_id: str, texts: dict[str, str], _original: dict[str, str]) -> str:
    execution = _execute(_model(sample_id), node_id, texts)
    if execution is None or not execution.ok:
        return ""
    return execution.matrix


def _recompute(sample_id: str, pairs: tuple[tuple[str, str], ...], targets: tuple[str, ...]):
    key = (sample_id, pairs, targets)
    cached = _RECOMPUTE.get(key)
    if cached is not None:
        return cached
    original = _manuscript(sample_id)
    texts = dict(original)
    edited = {slot for slot, _value in pairs}
    for slot, value in pairs:
        texts[slot] = value
    # A node that merely lists the injected blank is not ready to run: a calculator
    # between that blank and the node still has the manuscript value. Refresh those
    # in topological order. Only an injected blank itself, or a node the injection
    # does not reach, is read straight from the edited text.
    affected = _descendants(edited)
    direct = []
    indirect = []
    for target in targets:
        if target in edited or (target not in affected and edited <= set(_inputs(target))):
            direct.append(target)
        else:
            indirect.append(target)
    answers = {}
    failed = set()
    path = []
    for target in direct:
        if target in edited:
            answers[target] = texts.get(target, "")
            continue
        if not _has_calculator(target):
            failed.add(target)
            answers[target] = ""
            continue
        got = _execute_node(sample_id, target, texts, original)
        path.append(target)
        if not got:
            failed.add(target)
            answers[target] = ""
        else:
            answers[target] = got
    if indirect:
        parents = _graph()["parents"]
        relevant = set()
        descendants = _descendants(edited)
        for target in indirect:
            relevant |= descendants & (_ancestors(target) | {target})
        order = _topological(relevant)
        blocked = set()
        for node_id in order:
            if node_id in edited or not _has_calculator(node_id):
                continue
            if _blocked(node_id, blocked, parents):
                blocked.add(node_id)
                continue
            got = _execute_node(sample_id, node_id, texts, original)
            path.append(node_id)
            if not got:
                blocked.add(node_id)
                continue
            texts[node_id] = got
        for target in indirect:
            if target in blocked or _blocked(target, blocked, parents):
                failed.add(target)
                answers[target] = ""
            else:
                answers[target] = texts.get(target, "")
    packed = (answers, frozenset(failed), tuple(path))
    _RECOMPUTE[key] = packed
    return packed


def _blocked(node_id: str, failed: set[str], parents: dict) -> bool:
    if node_id in failed:
        return True
    seen = set()
    stack = list(parents.get(node_id, ()))
    while stack:
        current = stack.pop()
        if current in failed:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(parents.get(current, ()))
    return False


def _topological(nodes: set[str]) -> list[str]:
    parents = _graph()["parents"]
    children = _graph()["children"]
    indegree = {node_id: 0 for node_id in nodes}
    for node_id in nodes:
        for parent in parents.get(node_id, ()):
            if parent in nodes:
                indegree[node_id] += 1
    ready = sorted((node_id for node_id, degree in indegree.items() if degree == 0), key=int)
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in sorted(children.get(current, ()), key=int):
            if child not in indegree:
                continue
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort(key=int)
    return order


def _matches(node_id: str, left: str, right: str) -> bool:
    if not left or not right:
        return False
    if normalize(left) == normalize(right):
        return True
    kind = _kind(node_id)
    options = _options().get(node_id) if kind in {"choice", "sign"} else None
    pred = parse_answer(left, kind, options)
    gold = parse_answer(right, kind, options)
    if pred.status == "ok" and gold.status == "ok":
        try:
            if pred.value == gold.value:
                return True
        except Exception:
            pass
    if kind in _STRUCTURAL:
        try:
            return bool(sympy_equal(left, right))
        except Exception:
            return False
    return False


def _model_rows() -> tuple[dict, ...]:
    rows = []
    for folder, sample_id in (
        ("body-rule_self-trunc", "body-rule"),
        ("roll-rule_self-trunc", "roll-rule"),
    ):
        path = _OBS / folder / "observations.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("correct"):
                continue
            row["sample"] = sample_id
            row["node"] = str(row["node_id"])
            rows.append(row)
    return tuple(rows)


def _unexplained(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    found = []
    for row in rows:
        node_id = row["node"]
        sample_id = row["sample"]
        kind = _kind(node_id)
        options = row.get("options") or _options().get(node_id)
        pred = parse_answer(row["model_answer"], kind, options)
        gold = parse_answer(row["gold_answer"], kind, options)
        if not _has_calculator(node_id):
            continue
        if any(code in match_predicate(pred, gold) for code in _FORM_CODES):
            continue
        if pred.status == "ok" and _single_slot_hit(sample_id, node_id, row["model_answer"], row["gold_answer"]):
            continue
        found.append({
            "sample": sample_id,
            "node": node_id,
            "model": row["model_answer"],
            "gold": row["gold_answer"],
        })
    return tuple(found)


def _single_slot_hit(sample_id: str, node_id: str, model_text: str, gold_text: str) -> bool:
    for slot in single_slots(node_id, sample_id):
        answer = _outcome(sample_id, node_id, slot.node, slot.alt)[0]
        if answer and _matches(node_id, answer, model_text) and not _matches(node_id, answer, gold_text):
            return True
    return False


def _cluster_consumer(group: Group) -> str | None:
    match = None
    for cluster in _CLUSTERS:
        if set(group.slots) <= set(cluster.slots) and group.key.startswith(cluster.key):
            if match is None or len(cluster.key) > len(match.key):
                match = cluster
    return _consumer(match.slots if match is not None else group.slots)


def _regression(sample_id: str) -> tuple[int, list[str]]:
    mismatched = []
    checked = 0
    for node_id in calculator_nodes():
        for slot in single_slots(node_id, sample_id):
            previous, reason = _outcome(sample_id, node_id, slot.node, slot.alt)
            if reason or not previous:
                continue
            assignment = Assignment(((slot.node, slot.alt),), kind="joint")
            got = answer_under(node_id, assignment, sample_id)
            checked += 1
            if normalize(got) != normalize(previous) and not _matches(node_id, got, previous):
                mismatched.append(f"ID{node_id}←ID{slot.node}")
                if len(mismatched) >= 8:
                    return checked, mismatched
    return checked, mismatched


def _prune_report(sample_id: str) -> list[str]:
    lines = []
    for key in ("G-codes-attitude", "G-codes-load", "G-order-attitude", "G-axes-wind-body"):
        group = next(item for item in _CLUSTERS if item.key == key)
        _combos, stats = _enumerated(sample_id, group)
        lines.append(
            f"  {key} domain={stats['domain']} pair_checks={stats['pair_checks']} "
            f"full_checks={stats['full_checks']} accepted_multi={stats['accepted']}"
        )
    return lines


def audit(dump: bool = False) -> dict:
    started = time.perf_counter()
    minimal = _minimal_groups("standard-default")
    for group in minimal:
        consumer = _cluster_consumer(group)
        if consumer is None:
            raise RuntimeError(f"{group.key} has no decision node.")
        if not _domain_satisfiable("standard-default", consumer, group.slots, _domains("standard-default", group.slots)):
            raise RuntimeError(f"{group.key} is not satisfiable.")
        for slot in group.slots:
            rest = tuple(item for item in group.slots if item != slot)
            if rest and _domain_satisfiable("standard-default", consumer, rest, _domains("standard-default", rest)):
                raise RuntimeError(f"{group.key} is still satisfiable after ID{slot} is removed.")
    body_groups = tuple((group.key, group.slots) for group in _minimal_groups("body-rule"))
    roll_groups = tuple((group.key, group.slots) for group in _minimal_groups("roll-rule"))
    if body_groups != roll_groups:
        raise RuntimeError("B3 failed: the minimal groups of body and roll do not agree.")
    for sample_id in ("body-rule", "roll-rule"):
        allowed = set(changed_reads(sample_id))
        for item in chain_assignments(sample_id):
            outside = [slot for slot, _value in item.slots if slot not in allowed]
            if item.kind != "chain" or outside:
                raise RuntimeError(f"B4 failed: the chain slots {outside or item.slots} of {sample_id} are not in changed.")
    rows = _model_rows()
    unexplained = []
    by_sample = defaultdict(list)
    for row in _unexplained(rows):
        unexplained.append(row)
        by_sample[row["sample"]].append(row)
    explanations = []
    for sample_id, sample_rows in by_sample.items():
        explanations.extend(_explain_rows(sample_id, tuple(sample_rows)))
    print(f"unexplained={len(unexplained)} B1={len(explanations)}", flush=True)
    for item in explanations:
        if not item.answer or not item.path or not item.assignment.slots:
            raise RuntimeError(f"B5 failed: ID{item.target} is missing an injection, a path, or an answer.")
    if explanations:
        sample = explanations[0]
        signature = subtree_signature(sample.target, sample.assignment, sample.path, sample.sample)
        if signature.get(sample.target, "") != sample.answer:
            raise RuntimeError("subtree_signature does not agree with answer_under.")
    checked, mismatched = _regression("body-rule")
    if mismatched:
        raise RuntimeError("The single-slot regression does not agree with T3: " + "; ".join(mismatched))
    elapsed = time.perf_counter() - started
    no_calculator = sum(1 for row in rows if not _has_calculator(row["node"]))
    explained_early = len(rows) - no_calculator - len(unexplained)
    print(
        f"rows={len(rows)} no_calculator={no_calculator} already={explained_early} "
        f"unexplained={len(unexplained)} B1={len(explanations)} seconds={elapsed:.1f}"
    )
    print("changed body=" + ",".join(changed_definitions("body-rule")) + " roll=" + ",".join(changed_definitions("roll-rule")))
    print("minimal_groups")
    for group in minimal:
        print(f"  {group.key} {group.constraint} {' '.join('ID'+slot for slot in group.slots)}")
    if not minimal:
        print("  (no multi-slot group passed minimality)")
    print("seeds")
    for reason in (
        "Coordinate axis 'XTP' of ID120 cannot form a set with the other axes",
        "The transform order is not a permutation of angle of attack, sideslip, and roll.",
        "The six balance components do not each land on one coordinate axis.",
        "The start, the end, or the force location of the moment transfer does not match the template.",
    ):
        print(f"  {seed_from_reason(reason, '111')} ← {reason}")
    print("clusters")
    for group in _CLUSTERS:
        consumer = _consumer(group.slots)
        print(f"  {group.key} consumer={consumer or '-'} slots={' '.join(group.slots)}")
    print("prune")
    for line in _prune_report("body-rule"):
        print(line)
    print("branches " + " ".join(f"{name}:{_BRANCH_MODALITY.get(name, 'explicit')}" for name in _BRANCH_SLOTS))
    chain_hits = sum(1 for item in explanations if item.assignment.kind == "chain")
    print(f"chain_hits={chain_hits} regression={checked}")
    print("hits")
    for item in explanations:
        joined = " ".join(f"ID{slot}={value}" for slot, value in item.assignment.slots)
        print(f"  {item.sample[:4]} ID{item.target} {item.assignment.kind}:{item.assignment.name} {joined}")
    if dump:
        print("dump")
        for item in explanations:
            print(f"  target=ID{item.target} path={' '.join('ID'+node for node in item.path)}")
            print(f"    answer={item.answer[:180]}")
    if len(explanations) < 30:
        print(
            f"B1 did not reach 30 (actual {len(explanations)})."
            "Chain replacements in changed are counted separately. The misses include codes the calculator range cannot reach, and answers that contain γ_tp which an alignment matrix cannot produce."
        )
    return {
        "unexplained": len(unexplained),
        "hits": len(explanations),
        "chain_hits": chain_hits,
        "regression": checked,
        "minimal_groups": len(minimal),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check multi-slot replacement.")
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()
    audit(dump=args.dump)
