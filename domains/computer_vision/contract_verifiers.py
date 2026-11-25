# -*- coding: utf-8 -*-
"""Per-blank calculators for vision basic 1–66."""
from __future__ import annotations

import sympy as sp

from domains.computer_vision.answer_form import answers_equal
from domains.computer_vision.frame_apply import positive_axis, product_names, rotation_sign
from domains.computer_vision.contract_executor import Execution
from domains.computer_vision.manuscript import gold_texts, questions
from domains.computer_vision.pipeline_scope import NODE_IDS
from domains.computer_vision.signed_lists import DECLARATION_ROOTS, DERIVED_EXCEPTIONS, DERIVED_NODES, T_FINAL

_SIGN = {
    "19": (("5", "6", "7"), "16", "18", "15", "axial"),
    "25": (("5", "6", "7"), "22", "24", "21", "axial"),
    "31": (("5", "6", "7"), "28", "30", "27", "lateral"),
}
_AXIS = {"18": "17", "24": "23", "30": "29"}
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
    "37": ("32",),
    "38": ("26",),
    "39": ("20",),
    "40": ("11",),
    "41": ("37", "38", "39", "40"),
    "47": ("41",),
    "51": ("49",),
    "53": ("41", "49"),
    "54": ("51",),
    "55": ("53",),
}

_PSI, _THETA, _PHI = sp.symbols("ψ2 θ2 φ2", real=True)
_X2, _Y2, _Z2, _L = sp.symbols("x2 y2 z2 L", real=True)


def input_ids(node_id: str) -> tuple[str, ...]:
    return _INPUTS.get(str(node_id), ())


def _locals() -> dict:
    return {
        "ψ2": _PSI, "θ2": _THETA, "φ2": _PHI,
        "psi2": _PSI, "theta2": _THETA, "phi2": _PHI,
        "x2": _X2, "y2": _Y2, "z2": _Z2, "L": _L,
        "cos": sp.cos, "sin": sp.sin, "sqrt": sp.sqrt,
    }


def _parse_literal_matrix(text: str) -> sp.Matrix:
    raw = str(text).replace(" ", "")
    if not (raw.startswith("[[") and raw.endswith("]]")):
        raise ValueError(f"不是矩阵：{text}")
    rows = []
    depth = 0
    current = ""
    inner = raw[1:-1]
    for ch in inner:
        if ch == "[":
            depth += 1
            current += ch
        elif ch == "]":
            depth -= 1
            current += ch
            if depth == 0:
                rows.append(current)
                current = ""
        elif ch == "," and depth == 0:
            continue
        else:
            current += ch
    if len(rows) != 3:
        raise ValueError(f"矩阵不是 3×3：{text}")
    data = []
    for row in rows:
        cells = row[1:-1].split(",")
        if len(cells) != 3:
            raise ValueError(f"矩阵行不是三维：{row}")
        data.append([sp.sympify(cell, locals=_locals()) for cell in cells])
    return sp.Matrix(data)


def _parse_vector(text: str) -> sp.Matrix:
    raw = str(text).strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"不是向量：{text}")
    cells = [item.strip() for item in raw[1:-1].split(",")]
    if len(cells) != 3:
        raise ValueError(f"向量不是三维：{text}")
    return sp.Matrix([sp.sympify(cell, locals=_locals()) for cell in cells])


def _show_expr(expr) -> str:
    text = str(sp.simplify(expr))
    repl = {
        "ψ2": "ψ2", "θ2": "θ2", "φ2": "φ2",
        "psi2": "ψ2", "theta2": "θ2", "phi2": "φ2",
    }
    for key, value in repl.items():
        text = text.replace(key, value)
    return text.replace("**", "^")


def _show_matrix(matrix: sp.Matrix) -> str:
    rows = []
    for r in range(matrix.rows):
        cells = [_show_expr(matrix[r, c]) for c in range(matrix.cols)]
        rows.append("[" + ", ".join(cells) + "]")
    return "[" + ", ".join(rows) + "]"


def _show_vector(vector: sp.Matrix) -> str:
    return "[" + ", ".join(_show_expr(vector[i]) for i in range(3)) + "]"


def _copy_matrix(text: str) -> str:
    return _show_matrix(_parse_literal_matrix(text))


def _product4(a: str, b: str, c: str, d: str) -> str:
    matrix = _parse_literal_matrix(a) * _parse_literal_matrix(b) * _parse_literal_matrix(c) * _parse_literal_matrix(d)
    return _show_matrix(sp.simplify(matrix))


def _t21_from_r(r_text: str) -> str:
    r21 = _parse_literal_matrix(r_text)
    c2 = sp.Matrix([_X2, _Y2, _Z2])
    return _show_vector(sp.simplify(-r21 * c2))


def _pc2_from_r_pw(r_text: str, pw_text: str) -> str:
    r21 = _parse_literal_matrix(r_text)
    pw = _parse_vector(pw_text)
    c2 = sp.Matrix([_X2, _Y2, _Z2])
    t21 = -r21 * c2
    return _show_vector(sp.simplify(r21 * pw + t21))


def _skew(vector_text: str) -> str:
    x, y, z = (_parse_vector(vector_text)[i] for i in range(3))
    matrix = sp.Matrix([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])
    return _show_matrix(sp.simplify(matrix))


def derive(_model, node_id: str, texts: dict[str, str]) -> Execution:
    node_id = str(node_id)
    if node_id not in NODE_IDS:
        return Execution("contradictory", reason=f"ID{node_id} 不在 1–66")
    if node_id in DECLARATION_ROOTS or not input_ids(node_id):
        return Execution("derived", gold_texts()[node_id])
    try:
        if node_id in _AXIS:
            return Execution("derived", positive_axis(texts[_AXIS[node_id]]))
        if node_id in _SIGN:
            triad, role, axis, direction, moving = _SIGN[node_id]
            return Execution("derived", rotation_sign(texts, triad, role, axis, direction, moving))
        if node_id in {"33", "34", "35", "36"}:
            names = product_names(texts, ("12", "13", "14"), "2", "2", "3", "4", "RP1")
            return Execution("derived", names[int(node_id) - 33])
        if node_id in {"37", "38", "39", "40"}:
            parent = input_ids(node_id)[0]
            return Execution("derived", _copy_matrix(texts[parent]))
        if node_id == "41":
            return Execution("derived", _product4(texts["37"], texts["38"], texts["39"], texts["40"]))
        if node_id == "47":
            return Execution("derived", _t21_from_r(texts["41"]))
        if node_id == "51":
            return Execution("derived", _show_vector(_parse_vector(texts["49"])))
        if node_id == "53":
            return Execution("derived", _pc2_from_r_pw(texts["41"], texts["49"]))
        if node_id == "54":
            return Execution("derived", _skew(texts["51"]))
        if node_id == "55":
            return Execution("derived", _skew(texts["53"]))
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
        raise RuntimeError("声明根与派生节点没有覆盖 1–66")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"ID{node} 金标回归失败：{got.reason}")
        fmt = rows[node]["format"] if not rows[node]["options"] else "表达式"
        if not answers_equal(got.matrix, gold[node], fmt):
            raise RuntimeError(f"ID{node} 金标回归值不一致：{got.matrix!r} vs {gold[node]!r}")
    changed = answer_under("41", {"37": gold["20"]})
    if changed is None or answers_equal(changed, gold["41"], "矩阵"):
        raise RuntimeError("改写 ID37 后 ID41 没有变化")
    bad = derive(None, "41", {**gold, "37": "not-a-matrix"})
    if bad.ok:
        raise RuntimeError("非法上游仍推出了 ID41")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"派生例外未清空：{DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"终点 {node} 不在 1–66")
    print(
        f"阶段3 计算器：1–66 金标回归通过，声明根 {len(DECLARATION_ROOTS)}，"
        f"派生边 {sum(1 for n in NODE_IDS if input_ids(n))}。",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
