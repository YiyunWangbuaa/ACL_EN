# -*- coding: utf-8 -*-
"""Per-blank calculators for flight blanks 1–131."""
from __future__ import annotations

import re

import sympy as sp

from domains.aerodynamics.answer_form import answers_equal
from domains.aerodynamics.contract_executor import Execution
from domains.aerodynamics.flow_generate import (
    ATTACK,
    AZIMUTH,
    INCLINE,
    NAME,
    PITCH,
    ROLL,
    SIDESLIP,
    YAW,
    _at_degrees,
    _parse_factor,
    _show,
    _vector,
)
from domains.aerodynamics.flow_loads import _matrix
from domains.aerodynamics.manuscript import gold_texts, questions
from domains.aerodynamics.pipeline_scope import NODE_IDS
from domains.aerodynamics.signed_lists import DECLARATION_ROOTS, DERIVED_EXCEPTIONS, DERIVED_NODES, T_FINAL

E1 = sp.Matrix([1, 0, 0])
_DEGREE = re.compile(r"(\d+)°")
_INPUTS = {
    "15": ("12", "13"),
    "16": ("15",),
    "28": ("25",),
    "29": ("28",),
    "31": ("21",),
    "32": ("23",),
    "33": ("23",),
    "34": ("25",),
    "45": ("42", "43"),
    "46": ("2",),
    "47": ("1",),
    "49": ("37", "46"),
    "50": ("36", "47"),
    "55": ("34",),
    "56": ("44", "54"),
    "72": ("35", "36"),
    "73": ("1", "64", "72"),
    "75": ("37",),
    "76": ("2", "63", "75"),
    "79": ("77",),
    "80": ("74",),
    "81": ("79", "80"),
    "83": ("81",),
    "91": ("19",),
    "96": ("63", "70"),
    "107": ("106",),
    "109": ("108",),
    "110": ("109",),
    "114": ("3", "112"),
    "115": ("3", "112"),
    "116": ("3", "112"),
    "117": ("3", "112"),
    "118": ("3", "112"),
    "119": ("3", "112"),
    "124": ("1", "3", "112"),
    "125": ("112",),
    "126": ("112",),
}
_TRACK_ROLL = "气流系 Ya、Za 相对航迹系的方向"
_EXTRA = {
    "114": ("X", 1), "115": ("X", 2),
    "116": ("Y_b", 0), "117": ("Y_b", 2),
    "118": ("Z", 0), "119": ("Z", 1),
}


def input_ids(node_id: str) -> tuple[str, ...]:
    return _INPUTS.get(str(node_id), ())


def validator_of(node_id: str) -> str:
    if str(node_id) in DECLARATION_ROOTS:
        return "declaration"
    if input_ids(node_id):
        return "derived"
    return "declaration"


def _text(texts: dict[str, str], node: str) -> str:
    return str(texts.get(node, "")).strip()


def _same_side(left: str, right: str) -> str:
    if left not in {"上方", "下方", "左方", "右方"} or right not in {"上方", "下方", "左方", "右方"}:
        raise ValueError(f"方向无法对照：{left}，{right}")
    return "正" if left == right else "负"


def _cross(arm: list[int], force: list[int]) -> list[int]:
    return [
        arm[1] * force[2] - arm[2] * force[1],
        arm[2] * force[0] - arm[0] * force[2],
        arm[0] * force[1] - arm[1] * force[0],
    ]


def _arm_of(formula: str) -> list[int]:
    """Positive r_AG gives r_GA = -r_AG. The additional moment uses the chosen product."""
    if formula == "r_GA×F_b":
        return [-1, -1, -1]
    if formula == "r_AG×F_b":
        return [1, 1, 1]
    raise ValueError(f"附加力矩公式不是力的叉乘：{formula}")


def _signed_component(texts: dict[str, str], force: str, index: int) -> str:
    hand = _text(texts, "3")
    if hand not in {"右手系", "左手系"}:
        raise ValueError("坐标系手性不是右手系或左手系。")
    arm = _arm_of(_text(texts, "112"))
    column = {"X": [1, 0, 0], "Y_b": [0, 1, 0], "Z": [0, 0, 1]}[force]
    value = _cross(arm, column)[index]
    if hand == "左手系":
        value = -value
    if value == 0:
        return "零"
    return "正" if value > 0 else "负"


def _nose_pitch(texts: dict[str, str]) -> str:
    """Upward force on a nose-forward offset. +Z follows the stated normal."""
    normal = _text(texts, "1")
    if normal == "下方":
        force_z = -1
    elif normal == "上方":
        force_z = 1
    else:
        raise ValueError("法向正方向不是上方或下方。")
    formula = _text(texts, "112")
    arm = [-1, 0, 0] if formula == "r_GA×F_b" else [1, 0, 0] if formula == "r_AG×F_b" else None
    if arm is None:
        raise ValueError(f"附加力矩公式不是力的叉乘：{formula}")
    pitch = _cross(arm, [0, 0, force_z])[1]
    if _text(texts, "3") == "左手系":
        pitch = -pitch
    if normal == "下方":
        return "抬头方向" if pitch > 0 else "低头方向"
    return "低头方向" if pitch > 0 else "抬头方向"


def _applied(node: str, texts: dict[str, str]) -> str:
    """Application answers read from the recognition blanks already on the page."""
    if node == "31":
        return "转动" if _text(texts, "21") in {"Xg", "Yg"} else "保持不变"
    if node == "32":
        return "不改变" if _text(texts, "23") == "Xp" else "改变"
    if node == "33":
        if _text(texts, "23") == "Xp":
            return _TRACK_ROLL
        return "航迹系 Yp、Zp 轴"
    if node == "46":
        side = _text(texts, "2")
        if side not in {"左方", "右方"}:
            raise ValueError("侧向正方向不是左方或右方。")
        return side
    if node == "47":
        side = _text(texts, "1")
        if side not in {"上方", "下方"}:
            raise ValueError("法向正方向不是上方或下方。")
        return side
    if node == "49":
        return _same_side(_text(texts, "37"), _text(texts, "46"))
    if node == "50":
        return _same_side(_text(texts, "36"), _text(texts, "47"))
    if node == "56":
        direct = _text(texts, "44") == "bRa"
        plain = _text(texts, "54").startswith("不直接")
        if direct and plain:
            return "相同"
        if not direct:
            return "相反"
        return "一般不相关"
    if node == "72":
        if "对称面" not in _text(texts, "35"):
            raise ValueError("稳定性系前方轴不是空速在对称面内的投影。")
        side = _text(texts, "36")
        if side not in {"上方", "下方"}:
            raise ValueError("正迎角的空速偏向不是上方或下方。")
        return side
    if node == "73":
        if _text(texts, "64") != "Z_b":
            return "零"
        return _same_side(_text(texts, "72"), _text(texts, "1"))
    if node == "75":
        side = _text(texts, "37")
        if side not in {"左方", "右方"}:
            raise ValueError("正侧滑的空速偏向不是左方或右方。")
        return side
    if node == "76":
        if _text(texts, "63") != "Y_b":
            return "零"
        return _same_side(_text(texts, "75"), _text(texts, "2"))
    if node == "91":
        if _text(texts, "19") == "逆矩阵":
            return "相同"
        return "一般增大"
    if node == "96":
        if _text(texts, "70") == "+Y_a" and _text(texts, "63") == "Y_b":
            return "重合"
        if _text(texts, "70") in {"+X_a", "+Z_a"}:
            return "相互垂直"
        return "方向无法确定"
    if node == "107":
        direction = _text(texts, "106")
        if direction == "后方":
            return "相反"
        if direction == "前方":
            return "相同"
        raise ValueError("工程轴向正方向不是前方或后方。")
    if node in _EXTRA:
        force, index = _EXTRA[node]
        return _signed_component(texts, force, index)
    if node == "124":
        if _text(texts, "3") not in {"右手系", "左手系"}:
            raise ValueError("坐标系手性不是右手系或左手系。")
        return _nose_pitch(texts)
    if node == "125":
        formula = _text(texts, "112")
        if formula in {"r_GA×F_b", "r_AG×F_b"}:
            return "零"
        raise ValueError("平行向量的附加力矩只对力的叉乘有定义。")
    if node == "126":
        formula = _text(texts, "112")
        if formula in {"r_GA×F_b", "r_AG×F_b"}:
            return "相同"
        return "不能建立关系"
    raise ValueError(node)


def derive(_model, node_id: str, texts: dict[str, str]) -> Execution:
    node_id = str(node_id)
    if node_id not in NODE_IDS:
        return Execution("contradictory", reason=f"ID{node_id} 不在 1–131。")
    if node_id in DECLARATION_ROOTS or not input_ids(node_id):
        return Execution("derived", gold_texts()[node_id])
    try:
        if node_id in _INPUTS and node_id not in {"15", "16", "28", "29", "34", "45", "55", "79", "80", "81", "83", "109", "110"}:
            return Execution("derived", _applied(node_id, texts))
        if node_id == "15":
            return Execution("derived", _nose_ground(texts))
        if node_id == "16":
            return Execution("derived", _vertical_word(texts["15"]))
        if node_id == "34":
            return Execution("derived", _track_speed(texts["25"]))
        if node_id == "28":
            return Execution("derived", _track_z_from_product(texts["25"]))
        if node_id == "29":
            return Execution("derived", _sign_from_normal(texts["28"]))
        if node_id == "45":
            return Execution("derived", _body_speed(texts["42"], texts["43"]))
        if node_id == "55":
            return Execution("derived", str(texts["34"]).strip())
        if node_id == "79":
            return Execution("derived", str(texts["77"]).strip())
        if node_id == "80":
            return Execution("derived", str(texts["74"]).strip())
        if node_id == "81":
            return Execution("derived", _matrix_product(texts["79"], texts["80"]))
        if node_id == "83":
            return Execution("derived", _matrix_transpose(texts["81"]))
        if node_id == "109":
            return Execution("derived", f"[{str(texts['108']).strip()}, y_Ge, z_Ge]")
        if node_id == "110":
            return Execution("derived", _negate_vector(texts["109"]))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    return Execution("contradictory", reason=f"ID{node_id} 没有计算器。")


def _locals() -> dict:
    locals_map = {str(name): symbol for symbol, name in NAME.items()}
    locals_map.update({
        "α": ATTACK, "β": SIDESLIP, "φ": AZIMUTH, "μ": INCLINE,
        "ψ": YAW, "θ": PITCH, "Φ": ROLL,
        "cos": sp.cos, "sin": sp.sin, "sqrt": sp.sqrt, "rad": sp.rad,
    })
    return locals_map


def _parse_product(text: str) -> sp.Matrix:
    raw = str(text).replace(" ", "")
    matrix = sp.eye(3)
    for part in raw.split("*"):
        matrix *= _parse_factor(part)
    return sp.simplify(matrix)


def _parse_vector(text: str) -> sp.Matrix:
    raw = str(text).strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"不是向量：{text}")
    body = raw[1:-1]
    cells = [item.strip() for item in body.split(",")]
    if len(cells) != 3:
        raise ValueError(f"向量不是三维：{text}")
    values = []
    for cell in cells:
        expr_text = cell
        for amount in _DEGREE.findall(cell):
            expr_text = expr_text.replace(f"{amount}°", f"rad({amount})")
        values.append(sp.sympify(expr_text, locals=_locals()))
    return sp.Matrix(values)


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
        body = row[1:-1]
        cells = body.split(",")
        if len(cells) != 3:
            raise ValueError(f"矩阵行不是三维：{row}")
        data.append([sp.sympify(cell, locals=_locals()) for cell in cells])
    return sp.Matrix(data)


def _show_matrix(matrix: sp.Matrix) -> str:
    return _matrix(sp.simplify(matrix), _show)


def _vector_degrees(expr, values: dict) -> str:
    return f"[{', '.join(_at_degrees(expr[i], values) for i in range(3))}]"


def _nose_ground(texts: dict[str, str]) -> str:
    nose = sp.simplify(_parse_product(texts["12"]) * _parse_vector(texts["13"]))
    return _vector_degrees(nose, {YAW: 30, PITCH: 20, ROLL: 15})


def _vertical_word(vector_text: str) -> str:
    vector = _parse_vector(vector_text)
    text = str(vector[2])
    for amount in (15, 20, 30, 40, 10):
        text = text.replace(f"{amount}°", f"({sp.rad(amount)})")
    number = sp.N(sp.sympify(text, locals={"cos": sp.cos, "sin": sp.sin}))
    if abs(number) < 1e-10:
        raise ValueError("法向分量为 0")
    return "下方" if number > 0 else "上方"


def _track_speed(product_text: str) -> str:
    return _vector(sp.simplify(_parse_product(product_text) * E1))


def _track_z_from_product(product_text: str) -> str:
    speed = sp.simplify(_parse_product(product_text) * E1)
    return _at_degrees(speed[2], {AZIMUTH: 40, INCLINE: 10})


def _sign_from_normal(normal_text: str) -> str:
    text = str(normal_text).replace(" ", "")
    expr_text = text
    for amount in _DEGREE.findall(text):
        expr_text = expr_text.replace(f"{amount}°", f"({sp.rad(int(amount))})")
    number = sp.N(sp.sympify(expr_text, locals={"cos": sp.cos, "sin": sp.sin}))
    if abs(number) < 1e-10:
        raise ValueError("法向分量为 0")
    return "下方" if number > 0 else "上方"


def _body_speed(product_text: str, air_vector_text: str) -> str:
    return _vector(sp.simplify(_parse_product(product_text) * _parse_vector(air_vector_text)))


def _matrix_product(left: str, right: str) -> str:
    return _show_matrix(_parse_literal_matrix(left) * _parse_literal_matrix(right))


def _matrix_transpose(text: str) -> str:
    return _show_matrix(_parse_literal_matrix(text).T)


def _negate_vector(text: str) -> str:
    vector = _parse_vector(text)
    cells = []
    for item in -vector:
        shown = _show(item)
        cells.append(shown)
    # Gold uses x_Ge - X_N form for the first component of r_GA.
    joined = ", ".join(cells)
    joined = joined.replace("-(X_N - x_Ge)", "x_Ge - X_N").replace("-X_N + x_Ge", "x_Ge - X_N")
    return f"[{joined}]"


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
        raise RuntimeError("声明根与派生节点没有覆盖 1–131")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"ID{node} 金标回归失败：{got.reason}")
        if rows[node]["options"]:
            if got.matrix != gold[node]:
                raise RuntimeError(f"ID{node} 金标回归值不一致：{got.matrix!r} vs {gold[node]!r}")
        elif not answers_equal(got.matrix, gold[node], rows[node]["format"]):
            raise RuntimeError(f"ID{node} 金标回归值不一致：{got.matrix!r} vs {gold[node]!r}")
    wrong12 = "Cx(Φ)*Cy(θ)*Cz(ψ)"
    if wrong12 == gold["12"]:
        wrong12 = "Cz(ψ)*Cy(θ)*Cx(Φ)"
    changed = answer_under("15", {"12": wrong12})
    if changed is None or answers_equal(changed, gold["15"], "向量"):
        raise RuntimeError("改写 ID12 后 ID15 没有变化。")
    bad = derive(None, "15", {**gold, "12": "not-a-product"})
    if bad.ok:
        raise RuntimeError("非法上游仍推出了 ID15。")
    # Load-block: swapping a factor matrix must move ID81.
    alt79 = gold["74"] if gold["79"] != gold["74"] else "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]"
    changed81 = answer_under("81", {"79": alt79})
    if changed81 is None or answers_equal(changed81, gold["81"], "矩阵"):
        raise RuntimeError("改写 ID79 后 ID81 没有变化。")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"派生例外未清空：{DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"终点 {node} 不在 1–131")
    print(
        f"阶段3 计算器：1–131 金标回归通过，声明根 {len(DECLARATION_ROOTS)}，"
        f"派生边 {sum(1 for n in NODE_IDS if input_ids(n))}。",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
