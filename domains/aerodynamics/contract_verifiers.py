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
    VX,
    VY,
    VZ,
    YAW,
    _at_degrees,
    _formula,
    _parse_factor,
    _recovers,
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
    "12": ("4", "5"),
    "15": ("12", "13"),
    "16": ("15",),
    "25": ("22",),
    "28": ("25",),
    "29": ("28",),
    "31": ("21",),
    "32": ("23",),
    "33": ("23",),
    "34": ("25",),
    "40": ("36",),
    "41": ("37",),
    "42": ("40", "41"),
    "45": ("42", "43"),
    "46": ("2",),
    "47": ("1",),
    "48": ("42", "43"),
    "49": ("37", "46"),
    "50": ("36", "47"),
    "51": ("42", "43"),
    "55": ("34",),
    "56": ("44", "54"),
    "72": ("35", "36"),
    "73": ("1", "64", "72"),
    "74": ("40",),
    "75": ("37",),
    "76": ("2", "63", "75"),
    "77": ("41",),
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
_TRACK_ROLL = "the directions of airflow Ya and Za relative to the track frame"
_FACTOR = re.compile(r"C([xyz])\((.+)\)")
_YAW_ARG = {"right on the ground": "-ψ", "left on the ground": "ψ"}
_PITCH_ARG = {"upward": "-θ", "downward": "θ"}
_INCLINE_ARG = {"upward": "-μ", "downward": "μ"}
_ATTACK_FACTOR = {"downward": "Cy(α)", "upward": "Cy(-α)"}
_SIDESLIP_FACTOR = {"right": "Cz(-β)", "left": "Cz(β)"}
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
    known = {"upward", "downward", "left", "right"}
    if left not in known or right not in known:
        raise ValueError(f"Directions cannot be compared: {left}, {right}")
    return "positive" if left == right else "negative"


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
    raise ValueError(f"The additional-moment formula is not a cross product of a force: {formula}")


def _signed_component(texts: dict[str, str], force: str, index: int) -> str:
    hand = _text(texts, "3")
    if hand not in {"right-handed", "left-handed"}:
        raise ValueError("Frame chirality is not right-handed or left-handed.")
    arm = _arm_of(_text(texts, "112"))
    column = {"X": [1, 0, 0], "Y_b": [0, 1, 0], "Z": [0, 0, 1]}[force]
    value = _cross(arm, column)[index]
    if hand == "left-handed":
        value = -value
    if value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def _nose_pitch(texts: dict[str, str]) -> str:
    """Upward force on a nose-forward offset. +Z follows the stated normal."""
    normal = _text(texts, "1")
    if normal == "downward":
        force_z = -1
    elif normal == "upward":
        force_z = 1
    else:
        raise ValueError("The positive normal is not upward or downward.")
    formula = _text(texts, "112")
    arm = [-1, 0, 0] if formula == "r_GA×F_b" else [1, 0, 0] if formula == "r_AG×F_b" else None
    if arm is None:
        raise ValueError(f"The additional-moment formula is not a cross product of a force: {formula}")
    pitch = _cross(arm, [0, 0, force_z])[1]
    if _text(texts, "3") == "left-handed":
        pitch = -pitch
    if normal == "downward":
        return "nose-up" if pitch > 0 else "nose-down"
    return "nose-down" if pitch > 0 else "nose-up"


def _problem_factor(node: str, axis: str) -> str:
    """Axis factor that no blank asks about. Roll and track azimuth are problem data."""
    for part in gold_texts()[node].split("*"):
        matched = _FACTOR.fullmatch(part)
        if matched and matched.group(1) == axis:
            return part
    raise ValueError(f"ID{node} has no {axis} factor.")


def _one_factor(text: str) -> str:
    factor = text.strip()
    if not _FACTOR.fullmatch(factor):
        raise ValueError(f"A factor cannot be expanded: {factor}")
    return factor


def _ground_body(texts: dict[str, str]) -> str:
    yaw = _YAW_ARG.get(_text(texts, "4"))
    pitch = _PITCH_ARG.get(_text(texts, "5"))
    if yaw is None:
        raise ValueError("The positive yaw direction is not right on the ground or left on the ground.")
    if pitch is None:
        raise ValueError("The positive pitch direction is not upward or downward.")
    return f"Cz({yaw})*Cy({pitch})*{_problem_factor('12', 'x')}"


def _ground_track(texts: dict[str, str]) -> str:
    incline = _INCLINE_ARG.get(_text(texts, "22"))
    if incline is None:
        raise ValueError("The positive track inclination is not upward or downward.")
    return f"{_problem_factor('25', 'z')}*Cy({incline})"


def _attack_factor(texts: dict[str, str]) -> str:
    factor = _ATTACK_FACTOR.get(_text(texts, "36"))
    if factor is None:
        raise ValueError("The airspeed direction for positive angle of attack is not upward or downward.")
    return factor


def _sideslip_factor(texts: dict[str, str]) -> str:
    factor = _SIDESLIP_FACTOR.get(_text(texts, "37"))
    if factor is None:
        raise ValueError("The airspeed direction for positive sideslip is not left or right.")
    return factor


def _factor_product(texts: dict[str, str]) -> str:
    return f"{_one_factor(_text(texts, '40'))}*{_one_factor(_text(texts, '41'))}"


def _transposed_factor(text: str) -> str:
    return _show_matrix(_parse_factor(_one_factor(text)).T)


def _applied(node: str, texts: dict[str, str]) -> str:
    """Application answers read from the recognition blanks already on the page."""
    if node == "12":
        return _ground_body(texts)
    if node == "25":
        return _ground_track(texts)
    if node == "31":
        return "turns" if _text(texts, "21") in {"Xg", "Yg"} else "stays unchanged"
    if node == "32":
        return "does not change" if _text(texts, "23") == "Xp" else "changes"
    if node == "33":
        if _text(texts, "23") == "Xp":
            return _TRACK_ROLL
        return "the track axes Yp and Zp"
    if node == "40":
        return _attack_factor(texts)
    if node == "41":
        return _sideslip_factor(texts)
    if node == "42":
        return _factor_product(texts)
    if node == "46":
        side = _text(texts, "2")
        if side not in {"left", "right"}:
            raise ValueError("The positive lateral direction is not left or right.")
        return side
    if node == "47":
        side = _text(texts, "1")
        if side not in {"upward", "downward"}:
            raise ValueError("The positive normal is not upward or downward.")
        return side
    if node == "49":
        return _same_side(_text(texts, "37"), _text(texts, "46"))
    if node == "50":
        return _same_side(_text(texts, "36"), _text(texts, "47"))
    if node == "56":
        direct = _text(texts, "44") == "bRa"
        plain = _text(texts, "54").startswith("do not enter")
        if direct and plain:
            return "same"
        if not direct:
            return "opposite"
        return "generally unrelated"
    if node == "72":
        if "plane of symmetry" not in _text(texts, "35"):
            raise ValueError("The stability forward axis is not the airspeed projection in the plane of symmetry.")
        side = _text(texts, "36")
        if side not in {"upward", "downward"}:
            raise ValueError("The airspeed direction for positive angle of attack is not upward or downward.")
        return side
    if node == "73":
        if _text(texts, "64") != "Z_b":
            return "zero"
        return _same_side(_text(texts, "72"), _text(texts, "1"))
    if node == "74":
        return _transposed_factor(_text(texts, "40"))
    if node == "75":
        side = _text(texts, "37")
        if side not in {"left", "right"}:
            raise ValueError("The airspeed direction for positive sideslip is not left or right.")
        return side
    if node == "76":
        if _text(texts, "63") != "Y_b":
            return "zero"
        return _same_side(_text(texts, "75"), _text(texts, "2"))
    if node == "77":
        return _transposed_factor(_text(texts, "41"))
    if node == "91":
        if _text(texts, "19") == "inverse":
            return "same"
        return "generally increases"
    if node == "96":
        if _text(texts, "70") == "+Y_a" and _text(texts, "63") == "Y_b":
            return "aligned"
        if _text(texts, "70") in {"+X_a", "+Z_a"}:
            return "perpendicular"
        return "direction indeterminate"
    if node == "107":
        direction = _text(texts, "106")
        if direction == "rearward":
            return "opposite"
        if direction == "forward":
            return "same"
        raise ValueError("The positive engineering axial direction is not forward or rearward.")
    if node in _EXTRA:
        force, index = _EXTRA[node]
        return _signed_component(texts, force, index)
    if node == "124":
        if _text(texts, "3") not in {"right-handed", "left-handed"}:
            raise ValueError("Frame chirality is not right-handed or left-handed.")
        return _nose_pitch(texts)
    if node == "125":
        formula = _text(texts, "112")
        if formula in {"r_GA×F_b", "r_AG×F_b"}:
            return "zero"
        raise ValueError("The additional moment of parallel vectors is defined only for a cross product of a force.")
    if node == "126":
        formula = _text(texts, "112")
        if formula in {"r_GA×F_b", "r_AG×F_b"}:
            return "same"
        return "no relation can be formed"
    raise ValueError(node)


def derive(_model, node_id: str, texts: dict[str, str]) -> Execution:
    node_id = str(node_id)
    if node_id not in NODE_IDS:
        return Execution("contradictory", reason=f"ID{node_id} not in 1-131.")
    if node_id in DECLARATION_ROOTS or not input_ids(node_id):
        return Execution("derived", gold_texts()[node_id])
    try:
        if node_id in _INPUTS and node_id not in {"15", "16", "28", "29", "34", "45", "48", "51", "55", "79", "80", "81", "83", "109", "110"}:
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
        if node_id == "48":
            return Execution("derived", _sideslip_formula(texts["42"], texts["43"]))
        if node_id == "51":
            return Execution("derived", _attack_formula(texts["42"], texts["43"]))
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
    return Execution("contradictory", reason=f"ID{node_id} no calculator.")


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
        raise ValueError(f"Not a vector: {text}")
    body = raw[1:-1]
    cells = [item.strip() for item in body.split(",")]
    if len(cells) != 3:
        raise ValueError(f"vector not three-dimensional: {text}")
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
        raise ValueError(f"Not a matrix: {text}")
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
        raise ValueError(f"matrix not 3×3: {text}")
    data = []
    for row in rows:
        body = row[1:-1]
        cells = body.split(",")
        if len(cells) != 3:
            raise ValueError(f"matrix row not three-dimensional: {row}")
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
        raise ValueError("The normal component is 0")
    return "downward" if number > 0 else "upward"


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
        raise ValueError("The normal component is 0")
    return "downward" if number > 0 else "upward"


def _body_speed(product_text: str, air_vector_text: str) -> str:
    return _vector(sp.simplify(_parse_product(product_text) * _parse_vector(air_vector_text)))


def _recovered(product_text: str, air_vector_text: str, target, candidates: list) -> str:
    speed = sp.simplify(_parse_product(product_text) * _parse_vector(air_vector_text))
    return _formula(_recovers(speed, target, candidates))


def _sideslip_formula(product_text: str, air_vector_text: str) -> str:
    return _recovered(product_text, air_vector_text, SIDESLIP, [
        sp.atan2(VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(-VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(VZ, sp.sqrt(VX**2 + VY**2)),
        sp.atan2(-VZ, sp.sqrt(VX**2 + VY**2)),
    ])


def _attack_formula(product_text: str, air_vector_text: str) -> str:
    return _recovered(product_text, air_vector_text, ATTACK, [
        sp.atan2(VZ, VX),
        sp.atan2(-VZ, VX),
        sp.atan2(VY, VX),
        sp.atan2(-VY, VX),
    ])


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
        raise RuntimeError("Declaration roots and derived nodes do not cover 1-131")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"ID{node} gold regression failed: {got.reason}")
        if rows[node]["options"]:
            if got.matrix != gold[node]:
                raise RuntimeError(f"ID{node} gold regression value disagrees: {got.matrix!r} vs {gold[node]!r}")
        elif not answers_equal(got.matrix, gold[node], rows[node]["format"]):
            raise RuntimeError(f"ID{node} gold regression value disagrees: {got.matrix!r} vs {gold[node]!r}")
    wrong12 = "Cx(Φ)*Cy(θ)*Cz(ψ)"
    if wrong12 == gold["12"]:
        wrong12 = "Cz(ψ)*Cy(θ)*Cx(Φ)"
    changed = answer_under("15", {"12": wrong12})
    if changed is None or answers_equal(changed, gold["15"], "vector"):
        raise RuntimeError("ID15 did not change after ID12 was rewritten.")
    bad = derive(None, "15", {**gold, "12": "not-a-product"})
    if bad.ok:
        raise RuntimeError("An illegal upstream still produced ID15.")
    # Load-block: swapping a factor matrix must move ID81.
    alt79 = gold["74"] if gold["79"] != gold["74"] else "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]"
    changed81 = answer_under("81", {"79": alt79})
    if changed81 is None or answers_equal(changed81, gold["81"], "matrix"):
        raise RuntimeError("ID81 did not change after ID79 was rewritten.")
    flipped_yaw = "left on the ground" if gold["4"] == "right on the ground" else "right on the ground"
    yaw_body = "ψ" if flipped_yaw == "left on the ground" else "-ψ"
    changed12 = answer_under("12", {"4": flipped_yaw})
    if changed12 is None or changed12 == gold["12"] or not changed12.startswith(f"Cz({yaw_body})*"):
        raise RuntimeError("The yaw factor of ID12 did not change after ID4 was rewritten.")
    flipped_pitch = "downward" if gold["5"] == "upward" else "upward"
    pitch_body = "θ" if flipped_pitch == "downward" else "-θ"
    pitched = answer_under("12", {"5": flipped_pitch})
    if pitched is None or f"Cy({pitch_body})*" not in pitched:
        raise RuntimeError("The pitch factor of ID12 did not change after ID5 was rewritten.")
    both = answer_under("12", {"4": flipped_yaw, "5": flipped_pitch})
    roll = gold["12"].split("*")[-1]
    if both != f"Cz({yaw_body})*Cy({pitch_body})*{roll}":
        raise RuntimeError(f"After yaw and pitch are both reversed, ID12 is {both}.")
    climbed = answer_under("25", {"22": "downward"})
    if climbed is None or not climbed.endswith("*Cy(μ)"):
        raise RuntimeError("The inclination factor of ID25 did not change after ID22 was rewritten.")
    attacked = answer_under("40", {"36": "upward"})
    if attacked != "Cy(-α)":
        raise RuntimeError("ID40 did not become Cy(-α) after ID36 was rewritten.")
    slipped = answer_under("41", {"37": "left"})
    if slipped != "Cz(β)":
        raise RuntimeError("ID41 did not become Cz(β) after ID37 was rewritten.")
    recovered = answer_under("48", {"42": "Cy(α)*Cz(β)"})
    if recovered != "atan2(-vy, sqrt(vx^2+vz^2))":
        raise RuntimeError(f"After ID42 was rewritten, ID48 is {recovered}.")
    attacked_alpha = answer_under("51", {"42": "Cy(-α)*Cz(-β)"})
    if attacked_alpha != "atan2(-vz, vx)":
        raise RuntimeError(f"After ID42 was rewritten, ID51 is {attacked_alpha}.")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"Derived exceptions are not empty: {DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"Endpoint {node} not in 1-131")
    print(
        f"Stage 3 calculators: gold regression passed for 1-131, declaration roots {len(DECLARATION_ROOTS)}, "
        f"derived edges {sum(1 for n in NODE_IDS if input_ids(n))}.",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
