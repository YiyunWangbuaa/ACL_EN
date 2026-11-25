# -*- coding: utf-8 -*-
"""System repair / minimal fix sets for vision basic endpoints."""
from __future__ import annotations

from domains.computer_vision.answer_form import answers_equal
from domains.computer_vision.contract_verifiers import answer_under, input_ids
from domains.computer_vision.manuscript import gold_texts, questions
from domains.computer_vision.pipeline_scope import NODE_IDS
from domains.computer_vision.signed_lists import T_FINAL
from domains.repair_search import synthetic_minimal_fixes


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
    fmt = row["format"] if not row["options"] else "表达式"
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


def observation_fixes(answers: dict[str, str], *, prefix: str = "model"):
    """Restore local faults on a model trajectory and recompute signed targets."""
    from domains.computer_vision.fault_diagnosis import repairs
    return repairs(answers, prefix=prefix)


def check_t_final() -> None:
    report = []
    for target in T_FINAL:
        if target not in NODE_IDS:
            raise RuntimeError(f"终点 {target} 不在 1–66")
        fixes = minimal_fixes(target)
        if not fixes and target == "66":
            report.append(f"ID{target} 自修 {{ID{target}}}")
            fixes = ((target,),)
        elif not fixes:
            raise RuntimeError(f"终点 ID{target} 没有最小修复集")
        else:
            report.append(f"ID{target} 核 {list(fixes)}")
    print("阶段6 系统修复：" + "；".join(report), flush=True)


if __name__ == "__main__":
    check_t_final()
