# -*- coding: utf-8 -*-
"""Robotics kinematics view of the shared wind-tunnel diagnosis convention."""
from __future__ import annotations

from domains.mbd_kernel import DomainView, audit_stage4
from domains.robotics.kin_answer_form import answers_equal
from domains.robotics.kin_contract_verifiers import answer_under, input_ids, reference_phi
from domains.robotics.kin_manuscript import gold_texts, questions
from domains.robotics.kin_pipeline_scope import NODE_IDS
from domains.robotics.kin_signed_lists import DECLARATION_ROOTS


def _question(node: str) -> dict:
    return questions()[node]


def _equal(left: str, right: str, node: str) -> bool:
    row = questions()[node]
    fmt = row["format"] if not row["options"] else "expression"
    return answers_equal(left, right, fmt)


def view() -> DomainView:
    return DomainView(
        name="robotics kinematics",
        node_ids=tuple(NODE_IDS),
        parents=input_ids,
        gold=gold_texts,
        question=_question,
        answer_under=lambda node, assignment: answer_under(node, assignment),
        equal=_equal,
        declaration_roots=DECLARATION_ROOTS,
        reference=reference_phi,
    )


def check_stage4() -> None:
    audit_stage4(view())


def check_stage5() -> None:
    from domains.robotics.kin_fault_diagnosis import check
    check()
