# -*- coding: utf-8 -*-
"""Per-blank calculators for vision scene / object+stereo 1–165."""
from __future__ import annotations

import sympy as sp

from cloze_score import sympy_equal
from domains.computer_vision.contract_executor import Execution
from domains.computer_vision.frame_apply import mount_agreement, positive_axis, product_names, rotation_sign
from domains.computer_vision.contract_verifiers import (
    _copy_matrix,
    _parse_literal_matrix,
    _product4,
    _show_vector,
)
from domains.computer_vision.scene_answer_form import answers_equal
from domains.computer_vision.scene_manuscript import gold_texts, questions
from domains.computer_vision.scene_pipeline_scope import NODE_IDS
from domains.computer_vision.scene_signed_lists import (
    DECLARATION_ROOTS,
    DERIVED_EXCEPTIONS,
    DERIVED_NODES,
    T_FINAL,
)

_SIGN = {
    "19": (("5", "6", "7"), "16", "18", "15", "axial"),
    "25": (("5", "6", "7"), "22", "24", "21", "axial"),
    "31": (("5", "6", "7"), "28", "30", "27", "lateral"),
    "62": (("48", "49", "50"), "59", "61", "58", "axial"),
    "68": (("48", "49", "50"), "65", "67", "64", "axial"),
    "74": (("48", "49", "50"), "71", "73", "70", "lateral"),
}
_AXIS = {"18": "17", "24": "23", "30": "29", "61": "60", "67": "66", "73": "72"}
_INPUTS = {
    "18": ("17",),
    "19": ("5", "6", "7", "15", "16", "18"),
    "24": ("23",),
    "25": ("5", "6", "7", "21", "22", "24"),
    "30": ("29",),
    "31": ("5", "6", "7", "27", "28", "30"),
    "33": ("2", "3", "4", "12", "13", "14"),
    "34": ("2", "3", "4", "12", "13", "14"),
    "35": ("2", "3", "4", "12", "13", "14"),
    "36": ("2", "3", "4", "12", "13", "14"),
    "61": ("60",),
    "62": ("48", "49", "50", "58", "59", "61"),
    "67": ("66",),
    "68": ("48", "49", "50", "64", "65", "67"),
    "73": ("72",),
    "74": ("48", "49", "50", "70", "71", "73"),
    "76": ("45", "46", "47", "55", "56", "57"),
    "77": ("45", "46", "47", "55", "56", "57"),
    "78": ("45", "46", "47", "55", "56", "57"),
    "79": ("45", "46", "47", "55", "56", "57"),
    "100": ("99",),
    "37": ("32",),
    "38": ("26",),
    "39": ("20",),
    "40": ("11",),
    "41": ("37", "38", "39", "40"),
    "43": ("41",),
    "80": ("75",),
    "81": ("69",),
    "82": ("63",),
    "83": ("54",),
    "84": ("80", "81", "82", "83"),
    "86": ("84",),
}

_X1, _Y1, _Z1 = sp.symbols("x1 y1 z1", real=True)
_X2, _Y2, _Z2 = sp.symbols("x2 y2 z2", real=True)
_C1 = sp.Matrix([_X1, _Y1, _Z1])
_C2 = sp.Matrix([_X2, _Y2, _Z2])


def input_ids(node_id: str) -> tuple[str, ...]:
    return _INPUTS.get(str(node_id), ())


def _t_cw1_from_r(r_text: str) -> str:
    """t_cw1 = -R_cw1·C1, C1=(x1,y1,z1); angles ψ1,θ1,φ1 stay inside R_cw1."""
    rotation = _parse_literal_matrix(r_text)
    return _show_vector(sp.simplify(-rotation * _C1))


def _t_cw2_from_r(r_text: str) -> str:
    """t_cw2 = -R_cw2·C2, C2=(x2,y2,z2); angles ψ2,θ2,φ2 stay inside R_cw2."""
    rotation = _parse_literal_matrix(r_text)
    return _show_vector(sp.simplify(-rotation * _C2))


def derive(_model, node_id: str, texts: dict[str, str]) -> Execution:
    node_id = str(node_id)
    if node_id not in NODE_IDS:
        return Execution("contradictory", reason=f"ID{node_id} 不在 1–165")
    if node_id in DECLARATION_ROOTS or not input_ids(node_id):
        return Execution("derived", gold_texts()[node_id])
    try:
        if node_id in _AXIS:
            return Execution("derived", positive_axis(texts[_AXIS[node_id]]))
        if node_id in _SIGN:
            triad, role, axis, direction, moving = _SIGN[node_id]
            return Execution("derived", rotation_sign(texts, triad, role, axis, direction, moving))
        if node_id in {"33", "34", "35", "36"}:
            names = product_names(texts, ("12", "13", "14"), "1", "2", "3", "4", "RP1")
            return Execution("derived", names[int(node_id) - 33])
        if node_id in {"76", "77", "78", "79"}:
            names = product_names(texts, ("55", "56", "57"), "2", "45", "46", "47", "RP1")
            return Execution("derived", names[int(node_id) - 76])
        if node_id == "100":
            return Execution("derived", mount_agreement(texts["99"]))
        if node_id in {"37", "38", "39", "40", "80", "81", "82", "83"}:
            parent = input_ids(node_id)[0]
            return Execution("derived", _copy_matrix(texts[parent]))
        if node_id == "41":
            return Execution("derived", _product4(texts["37"], texts["38"], texts["39"], texts["40"]))
        if node_id == "43":
            return Execution("derived", _t_cw1_from_r(texts["41"]))
        if node_id == "84":
            return Execution("derived", _product4(texts["80"], texts["81"], texts["82"], texts["83"]))
        if node_id == "86":
            return Execution("derived", _t_cw2_from_r(texts["84"]))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    return Execution("contradictory", reason=f"ID{node_id} 没有计算器")


def reference_phi(node_id: str, assignment: dict[str, str], sample_id: str = "01"):
    """Defined value, contradiction, or None when the rule did not apply."""
    del sample_id
    from domains.contract_cognition import working_texts
    from domains.phi_result import from_execution

    texts = working_texts(gold_texts(), assignment, DECLARATION_ROOTS, input_ids(node_id))
    return from_execution(derive(None, node_id, texts))


def answer_under(node_id: str, assignment: dict[str, str], sample_id: str = "01") -> str | None:
    got = reference_phi(node_id, assignment, sample_id)
    return got if isinstance(got, str) else None


def check_verifiers() -> None:
    gold = gold_texts()
    rows = questions()
    if set(DECLARATION_ROOTS) | DERIVED_NODES != set(NODE_IDS):
        raise RuntimeError("声明根与派生节点没有覆盖 1–165")
    if input_ids("165"):
        raise RuntimeError("ID165 应是声明终点")
    for node in DERIVED_NODES:
        if not input_ids(node):
            raise RuntimeError(f"ID{node} 缺少派生输入")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"ID{node} 金标回归失败：{got.reason}")
        fmt = rows[node]["format"] if not rows[node]["options"] else "表达式"
        if not answers_equal(got.matrix, gold[node], fmt):
            raise RuntimeError(f"ID{node} 金标回归值不一致：{got.matrix!r} vs {gold[node]!r}")
        if node in DERIVED_NODES and not rows[node]["options"] and not sympy_equal(got.matrix, gold[node]):
            raise RuntimeError(f"ID{node} sympy 金标回归不一致：{got.matrix!r} vs {gold[node]!r}")
    changed = answer_under("41", {"37": gold["20"]})
    if changed is None or answers_equal(changed, gold["41"], "矩阵"):
        raise RuntimeError("改写 ID37 后 ID41 没有变化")
    changed_cam2 = answer_under("84", {"80": gold["63"]})
    if changed_cam2 is None or answers_equal(changed_cam2, gold["84"], "矩阵"):
        raise RuntimeError("改写 ID80 后 ID84 没有变化")
    bad = derive(None, "41", {**gold, "37": "not-a-matrix"})
    if bad.ok:
        raise RuntimeError("非法上游仍推出了 ID41")
    bad_cam2 = derive(None, "86", {**gold, "84": "not-a-matrix"})
    if bad_cam2.ok:
        raise RuntimeError("非法上游仍推出了 ID86")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"派生例外未清空：{DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"终点 {node} 不在 1–165")
    print(
        f"阶段3 计算器：1–165 金标回归通过，声明根 {len(DECLARATION_ROOTS)}，"
        f"派生边 {sum(1 for n in NODE_IDS if input_ids(n))}。",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
