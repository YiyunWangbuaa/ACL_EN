# -*- coding: utf-8 -*-
"""Injector-row catalog for older wind-tunnel checks. This is not the D matrix.

Stage 4 is ``domains.wind_tunnel.dmatrix``. Binary reach is ``signature``;
ambiguity groups are ``dmatrix_graph``. This module only lists injector rows.
It does not inject them or record the value a calculator would write.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from domains.wind_tunnel import slot_signature as SS
from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.multi_slot import (
    Assignment,
    _manuscript,
    branch_table,
)
from domains.wind_tunnel.contract_verifiers import validator_of

_ROOT = Path(__file__).resolve().parents[2]
_SAMPLES = _ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
_DEFAULT = "standard-default"
_SPECIALS = frozenset({"24", "32", "41", "26", "34", "43", "88"})
# Component sign. The problem asks only for the sign. The fault is written on the sign blank and does not rewrite the expression that is read.
_COMPONENT_SIGNS = frozenset({"62", "63", "64", "65", "66", "67", "161", "162", "163"})
# Equivalent-moment sign. The path must be the template value. A different path is rejected by the calculator and does not enter the sign judgment.
# The fault is written on the sign blank. It does not change ID184 or ID185-ID188.
_MOMENT_SIGNS = frozenset({"189", "190", "191", "192", "193", "194"})
# Moment numerator force coefficient, numerator coordinate, and denominator arm. The fault is written on this choice and injects another option. It does not recompute upstream.
_MOMENT_PARTS = frozenset(str(number) for number in range(201, 214))
_SIGN_BLANKS = _COMPONENT_SIGNS | _MOMENT_SIGNS | _MOMENT_PARTS
# The right-wing direction is derived from the right-hand rule. It is the same recognition fault as ID35 and ID37 and is not listed again.
_HAND_RULE_WING = frozenset({"36", "38"})
_SUFFIX = {"choice": " choice fault", "sign": " judgment fault", "matrix": " execution fault",
           "vector": " execution fault", "expression": " execution fault"}


@dataclass(frozen=True)
class Row:
    id: str
    klass: str                      # DIR | RSN | EXE | ALG
    label: str                      # semantic row label
    question: tuple[str, ...]       # every blank of the question this row belongs to
    injector: tuple                 # ((blank, alt), ...)
    variant: str                    # std | all-swap | single:<blank> | branch | chain

    @property
    def degree(self) -> int:
        return len(self.injector)


# ---------------------------------------------------------------- questions and dependencies

@lru_cache(maxsize=1)
def _questions() -> dict[str, dict]:
    for line in _SAMPLES.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("id") != _DEFAULT:
            continue
        return {str(q["id"]): q for q in row["questions"]}
    raise RuntimeError("standard-default was not found")


def _bare(x) -> str:
    return str(x).removeprefix("ID")


def _kind(node: str) -> str:
    node_obj = NODE_BY_ID.get(node)
    return "" if node_obj is None else node_obj.answer_type


@lru_cache(maxsize=None)
def calc_inputs(node: str) -> tuple[str, ...]:
    """The inputs the calculator actually reads. These are not the parents on the graph."""
    node = _bare(node)
    return tuple(_bare(i) for i in SS._inputs(node))


@lru_cache(maxsize=None)
def calc_ancestors(node: str) -> frozenset[str]:
    """Closure upward along calculator inputs, to judge whether an injection can be seen."""
    seen: set[str] = set()
    pending = list(calc_inputs(node))
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pending.extend(calc_inputs(item))
    return frozenset(seen)


@lru_cache(maxsize=None)
def graph_ancestors(node: str) -> frozenset[str]:
    """Closure upward along graph dependencies, to judge whether it is structurally reachable."""
    seen: set[str] = set()
    pending = list(NODE_BY_ID[_bare(node)].depends_on)
    while pending:
        item = _bare(pending.pop())
        if item in seen or item not in NODE_BY_ID:
            continue
        seen.add(item)
        pending.extend(NODE_BY_ID[item].depends_on)
    return frozenset(seen)


# ---------------------------------------------------------------- row construction

def _label(prompt: str) -> str:
    text = re.sub(r"[?.\s]+$", "", str(prompt)).strip()
    text = re.sub(r"^To compute[^,]*,\s*", "", text)
    if len(text) > 40:
        text = text.split(".")[0]
    return text + " contract recognition failed"


def e1_rows() -> tuple[Row, ...]:
    """DIR is the deduplicated direct-recognition entry. One stem is one row, and the row injects every blank of that question."""
    from domains.wind_tunnel.fault_instance import e1_instances

    std = _manuscript(_DEFAULT)
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in e1_instances():
        for blank, _v in item.injector:
            grouped[str(_questions()[_bare(blank)]["prompt"]).strip()].add(_bare(blank))
    rows = []
    for prompt, blanks in grouped.items():
        ordered = tuple(sorted(blanks, key=int))
        injector = tuple((b, std.get(b, "")) for b in ordered)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        rows.append(Row(f"{DIR}:{digest}", DIR, _label(prompt), ordered, injector, "std"))
    rows.sort(key=lambda r: int(r.question[0]))
    return tuple(rows)


def application_questions() -> dict[str, dict]:
    """Execution questions: they have a calculator and are not DIR direct-recognition blanks."""
    e1_blanks = {b for row in e1_rows() for b in row.question}
    found = {}
    for node, q in _questions().items():
        if node in e1_blanks:
            continue
        if validator_of(node) is not None or node in _SPECIALS:
            found[node] = q
    return found


def e3_rows(sample: str, scope: str = "core") -> tuple[Row, ...]:
    """EXE is an execution question. Each question gets a single-slot row (full mode) and one all-swap row.

    A sign blank, and a moment force coefficient, coordinate component, or denominator, write only the other side of that question. They do not use the all-swap row.
    """
    std = _manuscript(_DEFAULT)
    gold = _manuscript(sample)
    rows: list[Row] = []
    for node in sorted(application_questions(), key=int):
        q = _questions()[node]
        label = f"{q['role']}{_SUFFIX.get(str(q['answer_type']), ' execution fault')}"
        if node in _HAND_RULE_WING:
            continue
        if node in _SIGN_BLANKS:
            alt = _other_side(node, gold.get(node, ""), sample)
            if alt and alt != gold.get(node, ""):
                kind = "choice" if node in _MOMENT_PARTS else "sign"
                digest = hashlib.sha1(f"{node}|{kind}|{alt}".encode("utf-8")).hexdigest()[:8]
                rows.append(Row(f"{EXE}:{node}|{kind}|{digest}", EXE, label,
                                (node,), ((node, alt),), kind))
            continue
        inputs = calc_inputs(node)
        if not inputs:
            continue
        if scope == "full":
            for blank in inputs:
                for alt in _alternatives(blank, sample):
                    if alt == gold.get(blank):
                        continue
                    digest = hashlib.sha1(f"{node}|{blank}|{alt}".encode("utf-8")).hexdigest()[:8]
                    rows.append(Row(f"{EXE}:{node}|{blank}|{digest}", EXE, label,
                                    (node,), ((blank, alt),), f"single:{blank}"))
        # All-swap: every input takes the other side at once.
        swapped = []
        for blank in inputs:
            alt = _other_side(blank, gold.get(blank, ""), sample)
            if alt is None:
                swapped = []
                break
            swapped.append((blank, alt))
        if swapped and all(alt != gold.get(b) for b, alt in swapped):
            digest = hashlib.sha1(f"{node}|all".encode("utf-8")).hexdigest()[:8]
            rows.append(Row(f"{EXE}:{node}|all|{digest}", EXE, label,
                            (node,), tuple(swapped), "all-swap"))
    return tuple(rows)


def _alternatives(blank: str, sample: str) -> tuple[str, ...]:
    """A replaceable value for this blank. Prefer a sibling answer in the same family, otherwise the standard value."""
    std = _manuscript(_DEFAULT)
    gold = _manuscript(sample)
    found = []
    label = TARGETS.get(blank, ("", ""))[0]
    for other, node in NODE_BY_ID.items():
        if other == blank or TARGETS.get(other, ("", ""))[0] != label:
            continue
        if node.answer_type != NODE_BY_ID[blank].answer_type:
            continue
        value = gold.get(other, "")
        if value and value != gold.get(blank):
            found.append(value)
    standard = std.get(blank, "")
    if standard and standard != gold.get(blank):
        found.append(standard)
    return tuple(dict.fromkeys(found))


def _other_side(blank: str, current: str, sample: str) -> str | None:
    options = tuple(_questions()[blank].get("options") or ())
    if options:
        for option in options:
            if option and option != current:
                return option
        return None
    alts = [a for a in _alternatives(blank, sample) if a != current]
    return alts[0] if alts else None


def e2_rows(sample: str) -> tuple[Row, ...]:
    rows = []
    for name, assignment in branch_table(sample).items():
        if not assignment.slots:
            continue
        rows.append(Row(f"{RSN}:{name}", RSN, f"{name} reasoning recognition",
                        (), tuple(assignment.slots), "branch"))
    return tuple(rows)


def e4_rows(sample: str) -> tuple[Row, ...]:
    """ALG is a matrix, vector, or formula question. The row name stops at the role algebra error. Component signs are not here."""
    from domains.wind_tunnel.fault_instance import instances

    rows = []
    for item in instances(sample):
        if item.klass != ALG:
            continue
        node = item.injector[0][0]
        rows.append(Row(item.id, ALG, item.scope, (node,), tuple(item.injector), "algebra"))
    return tuple(rows)


def rows(sample: str, scope: str = "core") -> tuple[Row, ...]:
    return e1_rows() + e2_rows(sample) + e3_rows(sample, scope) + e4_rows(sample)


if __name__ == "__main__":
    print("This is not the D matrix. The stage-4 entry is domains.wind_tunnel.dmatrix.")
