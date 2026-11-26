# -*- coding: utf-8 -*-
"""System repair for the retired robotics-theory manuscript.

The live kinematics manuscript uses ``kin_t_final``. This module keeps the
same shared search so the two files do not drift.
"""
from __future__ import annotations

from domains.repair_search import synthetic_minimal_fixes
from domains.robotics.answer_form import answers_equal
from domains.robotics.contract_verifiers import answer_under, input_ids
from domains.robotics.manuscript import gold_texts, questions
from domains.robotics.pipeline_scope import NODE_IDS
from domains.robotics.signed_lists import T_FINAL


def _alt(node: str) -> str:
    gold = gold_texts()
    options = questions()[node]["options"]
    if options:
        found = next((item for item in options if item != gold[node]), None)
        if found is not None:
            return found
    return gold[node] + "'"


def _equals(node: str, left: str, right: str) -> bool:
    row = questions()[node]
    fmt = row["format"] if not row["options"] else "expression"
    return answers_equal(left, right, fmt)


def minimal_fixes(target: str) -> tuple[tuple[str, ...], ...]:
    """Smallest set of upstream declarations that restores this terminal."""
    return synthetic_minimal_fixes(
        target,
        parents_of=input_ids,
        answer_under=lambda node, assignment: answer_under(node, assignment),
        equals=_equals,
        gold=gold_texts(),
        alt_of=_alt,
    )


def check_t_final() -> None:
    report = []
    for target in T_FINAL:
        if target not in NODE_IDS:
            raise RuntimeError(f"terminal {target} is out of range")
        fixes = minimal_fixes(target)
        if not fixes:
            report.append(f"{target} has no core (the calculator rejects the faulty assignment)")
        else:
            report.append(f"{target} core {list(fixes)}")
        if not fixes:
            raise RuntimeError(f"terminal {target} has no minimal repair set")
    print("stage 6 system repair: " + "; ".join(report), flush=True)


if __name__ == "__main__":
    check_t_final()
