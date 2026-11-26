# -*- coding: utf-8 -*-
"""Single-slot signatures for nodes that have a calculator.

One slot is replaced at a time. ``answer_under`` is the whole new answer.
``cells`` is the set of atoms that change, and is None for choice and sign.
The base text is the manuscript gold, never a model answer. Propagation
across nodes is left to T5.
"""
from __future__ import annotations

from domains.fault_class import RSN

import json
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cloze_score import normalize, sympy_equal
from body_frame_model import BodyFrameModel
from domains.wind_tunnel.answer_form import cell_delta, parse_answer
from domains.wind_tunnel.contract_executor import (
    ALPHA_SYM,
    BETA_SYM,
    GAMMA_SYM,
    attitude_target,
    generated_answers,
    roll_target,
)
from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.contract_verifiers import derive, input_ids, validator_of
from domains.wind_tunnel.error_graph import NODE_BY_ID

_STRUCTURAL = {"matrix", "vector", "expression"}
_SAMPLES = Path(__file__).resolve().parents[2] / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
_DEFAULT_SAMPLE = "standard-default"
# Attitude and roll matrices are not in the coverage registry. Their inputs are
# the blanks the matrix formula reads. Axis choice and turn pairing are registry
# calculators and must not be listed here as well.
_SPECIAL_INPUTS = {
    "26": ("24", "25"),
    "34": ("32", "33"),
    "43": ("41", "42"),
    "88": ("83", "84", "86", "87", "89", "90", "91"),
}
_ATTITUDE = {
    "26": ("23", "24", "25", ALPHA_SYM),
    "34": ("31", "32", "33", BETA_SYM),
    "43": ("40", "41", "42", GAMMA_SYM),
}
# Declared nodes. They have a calculator and no replaceable parent.
# ID16–ID18 are also the order probes T4 will branch on.
_NO_SLOT_REASON = {
    "16": "implicit contract order slot, no replaceable prefix; RSN reasoning-recognition probe",
    "17": "implicit contract order slot, no replaceable prefix; RSN reasoning-recognition probe",
    "18": "implicit contract order slot, no replaceable prefix; RSN reasoning-recognition probe",
    "20": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "21": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "28": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "29": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "69": "the inverse trigonometric function is fixed by the declaration and has no replaceable prefix",
    "75": "the inverse trigonometric function is fixed by the declaration and has no replaceable prefix",
    "98": "the product slot is fixed by the declared order and has no replaceable prefix",
    "99": "the product slot is fixed by the declared order and has no replaceable prefix",
    "100": "the product slot is fixed by the declared order and has no replaceable prefix",
    "133": "the product slot is fixed by the declared order and has no replaceable prefix",
    "134": "the product slot is fixed by the declared order and has no replaceable prefix",
    "155": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "156": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "179": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "180": "the direction reading is fixed by the declaration and has no replaceable prefix",
    "182": "a fixed constant with no replaceable prefix",
    "201": "the force coefficient is read from the moment name and has no replaceable prefix",
    "202": "the force coefficient is read from the moment name and has no replaceable prefix",
    "203": "the force coefficient is read from the moment name and has no replaceable prefix",
    "204": "the force coefficient is read from the moment name and has no replaceable prefix",
    "205": "the force coefficient is read from the moment name and has no replaceable prefix",
    "206": "the force coefficient is read from the moment name and has no replaceable prefix",
}
BRANCH_PROBES = ("16", "17", "18")


@dataclass(frozen=True)
class Slot:
    node: str
    alt: str
    source: str
    family: str = ""


@dataclass(frozen=True)
class Signature:
    node: str
    slot: Slot
    answer: str
    cells: frozenset | None
    status: str


def calculator_nodes() -> tuple[str, ...]:
    found = [node_id for node_id in NODE_BY_ID if _has_calculator(node_id)]
    return tuple(sorted(found, key=int))


def no_slot_nodes() -> tuple[str, ...]:
    """Declared calculators. The seven registry gaps are not in this list."""
    return tuple(sorted(_NO_SLOT_REASON, key=int))


def no_slot_reason(node_id: str) -> str:
    return _NO_SLOT_REASON.get(_bare(node_id), "")


def masked_parents(node_id: str) -> tuple[str, ...]:
    """Graph parents the calculator does not read. They have no single-slot signature."""
    bare = _bare(node_id)
    if not _has_calculator(bare):
        return ()
    read = set(_inputs(bare))
    parents = NODE_BY_ID[bare].depends_on
    return tuple(parent for parent in parents if parent not in read)


def default_value(blank_id: str) -> str:
    return _standard_answers().get(_bare(blank_id), "")


def manuscript_answers(sample_id: str) -> dict[str, str]:
    """Gold answers of one manuscript."""
    return dict(_manuscript(sample_id))


def slots(node_id: str, sample_id: str = _DEFAULT_SAMPLE) -> tuple[Slot, ...]:
    """Replaceable parents and their alternative values. Empty when the node has none."""
    bare = _bare(node_id)
    if not _has_calculator(bare):
        return ()
    return _slot_table(bare, sample_id)


def answer_under(node_id: str, slot: Slot, sample_id: str) -> str:
    """Whole answer after replacing one parent. Empty when the calculator rejects it."""
    bare = _bare(node_id)
    if not _has_calculator(bare):
        return ""
    return _derived(sample_id, bare, slot.node, slot.alt) or ""


def cells(node_id: str, slot: Slot, sample_id: str) -> frozenset | None:
    """Atoms that change. None means the node has no cell index.

    An empty set means the cells match. A constant expression change is
    ``{("const",)}`` rather than that empty set.
    """
    bare = _bare(node_id)
    kind = _kind(bare)
    if not _has_calculator(bare) or kind not in _STRUCTURAL:
        return None
    answer = answer_under(bare, slot, sample_id)
    if not answer:
        return None
    gold = _manuscript(sample_id).get(bare, "")
    pred = parse_answer(answer, kind)
    gold_form = parse_answer(gold, kind)
    if pred.status != "ok" or gold_form.status != "ok":
        return None
    delta = cell_delta(pred, gold_form)
    if delta is None:
        return None
    if not delta and not _equivalent(answer, gold, kind):
        return frozenset({("const",)})
    return frozenset(delta)


def alt_classes(node_id: str, slot: Slot, sample_id: str) -> dict[str, tuple[str, ...]]:
    """Equivalent answers of one parent. The key is the representative text.

    An empty key collects alternatives the calculator rejected.
    """
    bare = _bare(node_id)
    kind = _kind(bare)
    groups: list[list] = []
    for item in slots(bare, sample_id):
        if item.node != slot.node:
            continue
        answer = answer_under(bare, item, sample_id)
        placed = False
        for group in groups:
            if _equivalent(group[0], answer, kind) or (not group[0] and not answer):
                group[1].append(item.alt)
                placed = True
                break
        if not placed:
            groups.append([answer, [item.alt]])
    return {group[0]: tuple(group[1]) for group in groups}


def inert_slots(sample_id: str) -> tuple[Signature, ...]:
    """One row per target and parent, when every accepted alternative matches gold."""
    found = []
    for node_id in calculator_nodes():
        seen: dict[str, list[Slot]] = {}
        for slot in slots(node_id, sample_id):
            seen.setdefault(slot.node, []).append(slot)
        gold = _manuscript(sample_id).get(node_id, "")
        kind = _kind(node_id)
        for _parent, items in seen.items():
            answers = [answer_under(node_id, item, sample_id) for item in items]
            valid = [answer for answer in answers if answer]
            if not valid:
                continue
            if all(_equivalent(answer, gold, kind) for answer in valid):
                changed = None if kind not in _STRUCTURAL else frozenset()
                found.append(Signature(node_id, items[0], gold, changed, "inert"))
    return tuple(found)


def derive_with_default(node_id: str, contract: str | None, sample_id: str) -> str:
    """Recompute the node with every input set to the standard-default value.

    ``contract`` selects nothing yet. T4 owns joint branch values.
    """
    del contract
    bare = _bare(node_id)
    if not _has_calculator(bare):
        return ""
    texts = dict(_manuscript(sample_id))
    for parent in _inputs(bare):
        texts[parent] = default_value(parent)
    execution = _execute(_model(sample_id), bare, texts)
    if execution is None or not execution.ok:
        return ""
    return execution.matrix


def _bare(node_id: str) -> str:
    return str(node_id).removeprefix("ID")


def _kind(node_id: str) -> str:
    node = NODE_BY_ID.get(node_id)
    return "" if node is None else node.answer_type


def _has_calculator(node_id: str) -> bool:
    return validator_of(node_id) is not None or node_id in _SPECIAL_INPUTS


def _inputs(node_id: str) -> tuple[str, ...]:
    registered = input_ids(node_id)
    if registered:
        return registered
    return _SPECIAL_INPUTS.get(node_id, ())


def _family(blank_id: str) -> str:
    node = NODE_BY_ID[blank_id]
    label, _subtype = TARGETS[blank_id]
    return f"{node.process}|{label}|{node.answer_type}"


@lru_cache(maxsize=1)
def _rows() -> dict[str, dict]:
    found = {}
    for line in _SAMPLES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[row["id"]] = row
    return found


@lru_cache(maxsize=None)
def _model(sample_id: str) -> BodyFrameModel:
    return BodyFrameModel().declare(_rows()[sample_id]["spec"])


@lru_cache(maxsize=None)
def _manuscript(sample_id: str) -> dict[str, str]:
    return generated_answers(_rows()[sample_id]["spec"])


def _standard_answers() -> dict[str, str]:
    return _manuscript(_DEFAULT_SAMPLE)


@lru_cache(maxsize=1)
def _options() -> dict[str, tuple[str, ...]]:
    found = {}
    for question in _rows()[_DEFAULT_SAMPLE]["questions"]:
        options = tuple(
            str(item) for item in (question.get("options") or []) if str(item) not in {"", "none"}
        )
        found[str(question["id"])] = options
    return found


@lru_cache(maxsize=1)
def _families(sample_id: str) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for blank_id in NODE_BY_ID:
        groups.setdefault(_family(blank_id), []).append(blank_id)
    return {key: tuple(sorted(ids, key=int)) for key, ids in groups.items()}


def _remember(found: dict[str, Slot], text: str, slot: Slot) -> None:
    key = normalize(text)
    if not key:
        return
    current = found.get(key)
    if current is not None and current.source == "standard":
        return
    if current is not None and slot.source != "standard":
        return
    found[key] = slot


def _parent_alts(parent: str, sample_id: str, current: str) -> tuple[Slot, ...]:
    kind = _kind(parent)
    family = _family(parent)
    standard = default_value(parent)
    found: dict[str, Slot] = {}
    options = _options().get(parent, ())
    if kind == "sign":
        for word in ("positive", "negative"):
            _remember(found, word, Slot(parent, word, "sign_flip", family))
    elif kind == "choice" and options:
        for option in options:
            source = "standard" if normalize(option) == normalize(standard) else "option"
            _remember(found, option, Slot(parent, option, source, family))
    else:
        answers = _manuscript(sample_id)
        for sibling in _families(sample_id).get(family, ()):
            if sibling == parent:
                continue
            text = answers.get(sibling, "")
            _remember(found, text, Slot(parent, text, "sibling", family))
        if standard:
            _remember(found, standard, Slot(parent, standard, "standard", family))
    current_key = normalize(current)
    found = {key: slot for key, slot in found.items() if key != current_key}
    return tuple(found.values())


@lru_cache(maxsize=None)
def _slot_table(node_id: str, sample_id: str) -> tuple[Slot, ...]:
    answers = _manuscript(sample_id)
    found = []
    for parent in _inputs(node_id):
        found.extend(_parent_alts(parent, sample_id, answers.get(parent, "")))
    return tuple(found)


def _execute(model, node_id: str, texts: dict[str, str]):
    if node_id in _ATTITUDE:
        frame_id, axis_id, turn_id, symbol = _ATTITUDE[node_id]
        return attitude_target(
            model, symbol, texts.get(frame_id, ""), texts.get(axis_id, ""), texts.get(turn_id, ""), texts,
        )
    if node_id == "88":
        return roll_target(
            model,
            texts.get("83", ""), texts.get("84", ""), texts.get("86", ""), texts.get("87", ""),
            texts.get("89", ""), texts.get("90", ""), texts.get("91", ""),
        )
    if validator_of(node_id) is None:
        return None
    return derive(model, node_id, texts)


@lru_cache(maxsize=None)
def _outcome(sample_id: str, node_id: str, parent: str, alt: str) -> tuple[str, str]:
    """Answer text and a rejection reason. A derived answer has an empty reason."""
    texts = dict(_manuscript(sample_id))
    texts[parent] = alt
    execution = _execute(_model(sample_id), node_id, texts)
    if execution is None:
        return "", "no_calculator"
    if not execution.ok:
        return "", execution.reason or "contradictory"
    return execution.matrix, ""


def _derived(sample_id: str, node_id: str, parent: str, alt: str) -> str:
    return _outcome(sample_id, node_id, parent, alt)[0]


def _equivalent(left: str, right: str, kind: str) -> bool:
    if not left or not right:
        return False
    if normalize(left) == normalize(right):
        return True
    if kind not in _STRUCTURAL:
        return False
    return sympy_equal(left, right)


def _parse_options(node_id: str) -> tuple[str, ...] | None:
    if _kind(node_id) not in {"choice", "sign"}:
        return None
    return _options().get(node_id) or None


def audit(sample_id: str = "body-rule") -> Mapping[str, int]:
    """A1–A5 for one manuscript. Raises when a threshold fails."""
    started = time.perf_counter()
    _check_registry()
    nodes = calculator_nodes()
    if len(nodes) != 122:
        raise RuntimeError(f"There should be 122 calculator nodes, but there are {len(nodes)}.")
    missing = []
    for node_id in nodes:
        reason = no_slot_reason(node_id)
        if slots(node_id, sample_id):
            if reason:
                missing.append(f"ID{node_id} has a slot, so it cannot be marked no_slots")
        elif not reason:
            missing.append(f"ID{node_id} has no slot and no no_slots reason")
    if missing:
        raise RuntimeError("; ".join(missing[:8]))
    if tuple(no_slot_nodes()) != tuple(sorted(_NO_SLOT_REASON, key=int)):
        raise RuntimeError("no_slots does not match the declared list.")
    total = 0
    accepted = 0
    invalid = []
    rejected = []
    reasons: Counter[str] = Counter()
    classes = 0
    parents = 0
    cell_mismatches = []
    for node_id in nodes:
        grouped: dict[str, list[Slot]] = {}
        for slot in slots(node_id, sample_id):
            grouped.setdefault(slot.node, []).append(slot)
        parents += len(grouped)
        kind = _kind(node_id)
        gold = _manuscript(sample_id).get(node_id, "")
        for _parent, items in grouped.items():
            seen: list[str] = []
            for slot in items:
                total += 1
                answer, reason = _outcome(sample_id, node_id, slot.node, slot.alt)
                if not answer:
                    rejected.append(f"ID{node_id}←ID{slot.node}:{slot.alt}")
                    reasons[reason.split(".", 1)[0]] += 1
                    continue
                form = parse_answer(answer, kind, None if kind in {"choice", "sign"} else _parse_options(node_id))
                if form.status != "ok":
                    invalid.append(f"ID{node_id}←ID{slot.node}:{slot.alt}")
                    continue
                del reason
                accepted += 1
                if any(_equivalent(answer, previous, kind) for previous in seen):
                    continue
                seen.append(answer)
                classes += 1
                if kind not in _STRUCTURAL or _equivalent(answer, gold, kind):
                    continue
                got = cells(node_id, slot, sample_id)
                pred = parse_answer(answer, kind)
                gold_form = parse_answer(gold, kind)
                delta = cell_delta(pred, gold_form)
                expect = frozenset({("const",)}) if delta is not None and not delta else frozenset(delta or ())
                if got != expect:
                    cell_mismatches.append(f"ID{node_id}←ID{slot.node}")
    elapsed = time.perf_counter() - started
    if total == 0:
        raise RuntimeError("No alternative value was enumerated.")
    comparable = accepted + len(invalid)
    rate = accepted / comparable if comparable else 1.0
    if rate < 0.95:
        preview = "; ".join(invalid[:8])
        raise RuntimeError(f"The answer_under parse rate is {rate:.1%}, below 95%. {preview}")
    if cell_mismatches:
        raise RuntimeError("cells does not agree with cell_delta: " + "; ".join(cell_mismatches[:8]))
    inert = inert_slots(sample_id)
    masked = [(node_id, parent) for node_id in nodes for parent in masked_parents(node_id)]
    print(
        f"sample={sample_id} calculators={len(nodes)} no_slots={len(no_slot_nodes())} "
        f"parents={parents} alts={total} classes={classes} "
        f"parsed={accepted} rejected={len(rejected)} invalid={len(invalid)} rate={rate:.1%}"
    )
    print(f"inert={len(inert)} masked={len(masked)} seconds={elapsed:.1f}")
    print("no_slots " + " ".join(f"ID{node_id}" for node_id in no_slot_nodes()))
    print("branch_probes " + " ".join(f"ID{node_id}" for node_id in BRANCH_PROBES))
    if reasons:
        print("rejected_reasons")
        shown = reasons.most_common(8)
        for reason, count in shown:
            print(f"  {count} {reason}")
        hidden = len(reasons) - len(shown)
        if hidden:
            print(f"  ... {hidden} more reasons")
    if invalid:
        print("invalid_alt")
        for item in invalid:
            print(f"  {item}")
    print("inert_slots")
    for item in inert:
        print(f"  ID{item.node}←ID{item.slot.node} {item.slot.source}")
    print("masked_parents")
    for node_id, parent in masked:
        print(f"  ID{node_id}←ID{parent}")
    return {
        "calculators": len(nodes),
        "no_slots": len(no_slot_nodes()),
        "parents": parents,
        "alts": total,
        "classes": classes,
        "rejected": len(rejected),
        "invalid": len(invalid),
        "inert": len(inert),
        "masked": len(masked),
    }


def _check_registry() -> None:
    empty = [node_id for node_id in NODE_BY_ID if validator_of(node_id) and not input_ids(node_id)]
    if set(empty) != set(_NO_SLOT_REASON):
        raise RuntimeError(f"Declared blanks do not match the registry blanks with no inputs: {sorted(set(empty) ^ set(_NO_SLOT_REASON), key=int)}")
    for node_id, parents in _SPECIAL_INPUTS.items():
        if input_ids(node_id):
            raise RuntimeError(f"ID{node_id} already has registered inputs, so the manual table would repeat them.")
        if _inputs(node_id) != parents:
            raise RuntimeError(f"The manual inputs of ID{node_id} should be {parents}.")
    for node_id in BRANCH_PROBES:
        if RSN not in no_slot_reason(node_id):
            raise RuntimeError(f"ID{node_id} should be marked an RSN reasoning-recognition probe.")
    if cells("16", Slot("16", "angle of attack", "standard"), _DEFAULT_SAMPLE) is not None:
        raise RuntimeError("cells of a declared node should be None.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check single-slot signatures.")
    parser.add_argument("sample", nargs="?", default="body-rule")
    args = parser.parse_args()
    audit(args.sample)
