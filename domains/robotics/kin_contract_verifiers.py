# -*- coding: utf-8 -*-
"""Per-blank calculators for robotics kinematics F*/I*."""
from __future__ import annotations

import re

import sympy as sp

from domains.robotics.contract_executor import Execution
from domains.robotics.kin_answer_form import answers_equal
from domains.robotics.kin_manuscript import gold_texts, manuscript_path, questions
from domains.robotics.kin_pipeline_scope import NODE_IDS
from domains.robotics.kin_signed_lists import DECLARATION_ROOTS, DERIVED_EXCEPTIONS, DERIVED_NODES, T_FINAL

_DIR = {
    "forward": (1, 0, 0), "rearward": (-1, 0, 0),
    "right": (0, 1, 0), "left": (0, -1, 0),
    "downward": (0, 0, 1), "upward": (0, 0, -1),
}
_OPP = {word: name for word, name in (
    ("forward", "rearward"), ("rearward", "forward"),
    ("right", "left"), ("left", "right"),
    ("downward", "upward"), ("upward", "downward"),
)}
_TURN = {
    "F13": ("F11", "F12", "forward"),
    "F18": ("F16", "F17", "forward"),
    "F23": ("F21", "F22", "upward"),
}
# One edge per blank the question asks the later answer to execute.
# A derived node and its parents are the same set: the graph check locks them.
_INPUTS = {
    "F11": ("F10", "F4", "F5", "F6"),
    "F13": ("F4", "F5", "F6", "F11", "F12"),
    "F14": ("F4", "F5", "F6", "F11", "F12", "F13"),
    "F16": ("F15", "F4", "F5", "F6"),
    "F18": ("F4", "F5", "F6", "F16", "F17"),
    "F19": ("F4", "F5", "F6", "F16", "F17", "F18"),
    "F21": ("F20", "F4", "F5", "F6"),
    "F23": ("F4", "F5", "F6", "F21", "F22"),
    "F24": ("F4", "F5", "F6", "F21", "F22", "F23"),
    "F27": ("F26",),
    "F28": ("F25", "F26"),
    "F29": ("F28",),
    "F30": ("F4", "F5", "F6", "F25"),
    "F31": ("F4", "F5", "F6", "F27"),
    "F32": ("F4", "F5", "F6", "F29"),
    "F33": ("F30", "F31", "F32"),
    "F34": ("F33",),
    "F35": ("F33",),
    "F39": ("F22", "F35"),
    "F40": ("F16", "F17", "F22", "F34", "F39"),
    "F41": ("F14",),
    "F42": ("F19",),
    "F43": ("F24",),
    "F44": ("F33",),
    "F45": ("F37", "F41", "F42", "F43", "F44"),
    "F48": ("F45", "F47"),
    "F50": ("F49",),
    "F52": ("F51",),
    "F53": ("F48", "F50", "F52"),
    "F59": ("F53", "F54"),
    "F60": ("F53", "F55"),
    "F61": ("F9",),
    "F62": ("F1", "F45", "F61"),
    "F63": ("F62",),
    "F72": ("F66",),
    "F73": ("F66", "F72"),
    "F74": ("F71", "F73"),
    "F75": ("F70", "F74"),
    "F77": ("F68",),
    "F78": ("F68", "F77"),
    "F80": ("F78", "F79"),
    "F81": ("F67",),
    "F82": ("F67", "F81"),
    "F84": ("F82", "F83"),
    "F85": ("F72", "F75", "F76", "F77", "F80", "F81", "F84"),
    "F92": ("F88", "F89", "F90"),
    "F94": ("F93",),
    "F96": ("F92", "F94", "F95"),
    "F98": ("F96",),
    "I7": ("I6",),
    "I9": ("I7", "I8"),
    "I12": ("I10", "I11"),
    "I13": ("I10", "I11", "I12"),
    "I16": ("I9", "I13", "I14", "I15"),
}
_ROTATION = {
    "F14": ("F11", "F12", "F13", "forward", "q1"),
    "F19": ("F16", "F17", "F18", "forward", "q2"),
    "F24": ("F21", "F22", "F23", "upward", "q3"),
}
_ROLE_OF = {"normal": "upward", "axial": "forward", "lateral": "right"}
_POINT = {
    "task forward direction": "forward",
    "base up": "upward",
    "base right": "right",
    "base down": "downward",
    "forward": "forward", "rearward": "rearward",
    "upward": "upward", "downward": "downward",
    "right": "right", "left": "left",
}
_LENGTH = {
    "tool nose to the tool base": "L",
    "tool base to the wrist origin": "L_mount",
    "tool nose to the wrist origin": "L + L_mount",
    "the length of the mount inserted into the tool": "L_in",
}
_TOOL_AXIS = {"forward": "X_T", "rearward": "-X_T", "upward": "Y_T", "downward": "-Y_T", "right": "Z_T", "left": "-Z_T"}
_ROOT_SYMBOL = {
    "x_r": "x_r",
    "tip spacing b": "b",
    "leading-edge sweep Λ": "Λ",
    "dihedral Γ": "Γ",
}

_Q1, _Q2, _Q3 = sp.symbols("q1 q2 q3", real=True)
_ANGLE_SUBS = {_Q1: 0, _Q2: sp.rad(30), _Q3: sp.rad(-60)}


def input_ids(node_id: str) -> tuple[str, ...]:
    return _INPUTS.get(str(node_id), ())


def _locals() -> dict:
    return {
        "q1": _Q1, "q2": _Q2, "q3": _Q3,
        "cos": sp.cos, "sin": sp.sin, "sqrt": sp.sqrt,
        "α_T": sp.atan(sp.Rational(1, 2)),
        "β_T": sp.asin(sp.Rational(1, 4)),
        "γ_T": sp.asin(sp.Rational(1, 2)),
        "a": sp.symbols("a", real=True),
        "bv": sp.symbols("bv", real=True),
        "c": sp.symbols("c", real=True),
        "d": sp.symbols("d", real=True),
        "N": sp.symbols("N", positive=True),
    }


def _show_matrix(matrix: sp.Matrix) -> str:
    rows = []
    for r in range(3):
        cells = [str(sp.simplify(matrix[r, c])).replace("**", "^") for c in range(3)]
        rows.append("[" + ", ".join(cells) + "]")
    return "[" + ", ".join(rows) + "]"


def _show_vector(vector: sp.Matrix) -> str:
    cells = [str(sp.simplify(vector[i])).replace("**", "^") for i in range(3)]
    return "[" + ", ".join(cells) + "]"


def _parse_literal_matrix(text: str) -> sp.Matrix:
    raw = str(text).replace(" ", "")
    if not (raw.startswith("[[") and raw.endswith("]]")):
        raise ValueError(f"not a matrix: {text}")
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
        cells = row[1:-1].split(",")
        if len(cells) != 3:
            raise ValueError(f"matrix row not three-dimensional: {row}")
        data.append([sp.sympify(cell, locals=_locals()) for cell in cells])
    return sp.Matrix(data)


def _parse_vector(text: str) -> sp.Matrix:
    raw = str(text).strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"not a vector: {text}")
    cells = [item.strip() for item in raw[1:-1].split(",")]
    if len(cells) != 3:
        raise ValueError(f"vector not three-dimensional: {text}")
    return sp.Matrix([sp.sympify(cell, locals=_locals()) for cell in cells])


def _pose_from_factors(r1: str, r2: str, r3: str, s: str, rule: str = "R*S") -> str:
    factors = [
        _parse_literal_matrix(r1),
        _parse_literal_matrix(r2),
        _parse_literal_matrix(r3),
        _parse_literal_matrix(s),
    ]
    choice = str(rule).replace(" ", "")
    if choice == "S*R":
        ordered = (factors[3], factors[0], factors[1], factors[2])
    elif choice == "R*S":
        ordered = tuple(factors)
    else:
        raise ValueError(f"attitude product is not R*S or S*R: {rule}")
    matrix = ordered[0] * ordered[1] * ordered[2] * ordered[3]
    return _show_matrix(sp.simplify(matrix.subs(_ANGLE_SUBS)))


def _tool_components(d_text: str, rule_text: str) -> str:
    matrix = _parse_literal_matrix(d_text)
    ref = sp.Matrix([1, 0, 0])
    rule = str(rule_text).lower()
    if "transpose" in rule:
        vector = sp.simplify(matrix.T * ref)
    elif rule.startswith("d times"):
        vector = sp.simplify(matrix * ref)
    else:
        raise ValueError(f"cannot interpret the mapping rule: {rule_text}")
    return _show_vector(vector)


def _translate_expr(offset_text: str) -> str:
    return f"D*{str(offset_text).strip()}"


_TASK_ANGLES = re.compile(r"target task angles of this robot task are ([^.]+)\.")
_ROW_SYMBOLS = re.compile(r"Let a=([^,]+), bv=([^,]+), and c=([^.]+)\.")
_D_SYMBOL = re.compile(r"Let d=(.+?) and N=")


def _task_angle(token: str) -> sp.Expr:
    token = token.strip()
    sign = 1
    if token.startswith("-"):
        sign = -1
        token = token[1:].strip()
    name, rest = token.split("(", 1)
    arg = rest[:-1]
    if name == "atan2":
        left, right = arg.split(",")
        value = sp.atan2(sp.sympify(left), sp.sympify(right))
    else:
        value = {"arctan": sp.atan, "arcsin": sp.asin}[name](sp.sympify(arg))
    return sign * value


def _example_symbols() -> dict[str, sp.Expr]:
    """Numeric a, bv, c, d, N at the angles written in the inverse sample."""
    inverse = manuscript_path().read_text(encoding="utf-8").split("# Inverse kinematics", 1)[1]
    alpha, beta, gamma = [_task_angle(part) for part in _TASK_ANGLES.search(inverse).group(1).split(",")]
    a_src, bv_src, c_src = _ROW_SYMBOLS.search(inverse).groups()
    names = {
        "α_T": alpha, "β_T": beta, "γ_T": gamma,
        "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt,
    }
    a = sp.simplify(sp.sympify(a_src, locals=names))
    bv = sp.simplify(sp.sympify(bv_src, locals=names))
    c = sp.simplify(sp.sympify(c_src, locals=names))
    d = sp.simplify(sp.sympify(
        _D_SYMBOL.search(inverse).group(1),
        locals={**names, "a": a, "bv": bv, "c": c},
    ))
    return {
        "a": a, "bv": bv, "c": c, "d": d, "γ_T": gamma,
        "N": sp.sqrt(sp.simplify(a**2 + d**2)),
        "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt,
    }


def _split_row(text: str) -> list[str]:
    body = str(text).strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    cells = []
    depth = 0
    current = ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            cells.append(current.strip())
            current = ""
        else:
            current += ch
    cells.append(current.strip())
    return cells


def _first_row_numeric(choice_text: str) -> str:
    symbols = _example_symbols()
    order = _split_row(choice_text)
    if any(name not in {"a", "bv", "c"} for name in order) or len(order) != 3:
        raise ValueError(f"unknown first-row form: {choice_text}")
    return _show_vector(sp.Matrix([symbols[name] for name in order]))


def _d_from_rows(row1_text: str, row2_choice: str) -> str:
    symbols = _example_symbols()
    row1 = _parse_vector(row1_text)
    row2 = sp.Matrix([
        sp.simplify(sp.sympify(cell, locals=symbols)) for cell in _split_row(row2_choice)
    ])
    row3 = sp.simplify(row1.cross(row2))
    return _show_matrix(sp.Matrix.vstack(row1.T, sp.simplify(row2).T, row3.T))


def _joints_from_pose(d_text: str, erase_choice: str, s_text: str, formula: str) -> str:
    d = _parse_literal_matrix(d_text)
    s = _parse_literal_matrix(s_text)
    choice = erase_choice.replace(" ", "")
    if "S^T*D" in choice:
        h = sp.simplify(s.T * d)
    elif "D*S^T" in choice:
        h = sp.simplify(d * s.T)
    elif choice == "D*S":
        h = sp.simplify(d * s)
    else:
        raise ValueError(f"the product that removes the installation is not recognized: {erase_choice}")
    names = {f"h{i}{j}": h[i - 1, j - 1] for i in range(1, 4) for j in range(1, 4)}
    vals = []
    for part in str(formula).split(";"):
        expr = part.strip().replace("^", "**")
        degree = sp.deg(sp.sympify(expr, locals={**names, "sqrt": sp.sqrt, "atan2": sp.atan2}))
        vals.append(int(round(float(sp.N(degree)))))
    return f"[{vals[0]}, {vals[1]}, {vals[2]}]"


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left, right) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _wrist(texts: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    found = {}
    for node, word in (("F4", "upward"), ("F5", "forward"), ("F6", "right")):
        raw = str(texts.get(node, "")).strip()
        negative = raw.startswith("-")
        name = raw[1:] if negative else raw
        vector = _DIR[word]
        if negative:
            vector = tuple(-item for item in vector)
        if not name or name in found:
            raise ValueError(f"the wrist axis of {node} cannot form a set with the other axes.")
        found[name] = vector
    return found


def _joint_turn(texts: dict[str, str], axis_id: str, hand_id: str, moving_word: str) -> str:
    frame = _wrist(texts)
    axis = str(texts.get(axis_id, "")).strip()
    if axis.startswith("-"):
        axis = axis[1:]
    if axis not in frame:
        raise ValueError(f"joint axis {axis} is not one of the three wrist axes.")
    hand = str(texts.get(hand_id, "")).strip()
    if hand not in {"same", "opposite"}:
        raise ValueError("the joint positive sense is not same or opposite the right-hand rule.")
    moving = _DIR[moving_word]
    hits = [
        word for word, toward in _DIR.items()
        if _dot(_cross(frame[axis], moving), toward) > 0
    ]
    if len(hits) != 1:
        raise ValueError("a positive turn does not have a unique deflection direction.")
    return _OPP[hits[0]] if hand == "opposite" else hits[0]


def _contains(text: str, words: tuple[str, ...]) -> str:
    found = [word for word in words if word in str(text)]
    if len(found) != 1:
        raise ValueError(f"the direction does not contain a unique {'/'.join(words)}: {text}")
    return found[0]


def _component_sign(vector_text: str, index: int) -> int:
    value = sp.simplify(_parse_vector(vector_text)[index])
    if value == 0:
        raise ValueError("a task-direction component is 0")
    positive = value.is_positive
    if positive is None:
        positive = bool(float(value) > 0)
    return 1 if positive else -1


def _and_directions(vertical: str, lateral: str) -> str:
    short = {"upward": "up", "downward": "down", "right": "right", "left": "left"}
    return f"{short[vertical]} and {short[lateral]}"


def _task(node: str, texts: dict[str, str]) -> str:
    if node == "F50":
        return _OPP[str(texts["F49"]).strip()]
    if node == "F52":
        side = str(texts["F51"]).strip()
        if side not in _DIR:
            raise ValueError(f"the component positive direction is not recognized: {side}")
        return side
    if node == "F53":
        vertical = texts["F50"].strip() if _component_sign(texts["F48"], 1) < 0 else _OPP[texts["F50"].strip()]
        lateral = texts["F52"].strip() if _component_sign(texts["F48"], 2) > 0 else _OPP[texts["F52"].strip()]
        return _and_directions(vertical, lateral)
    if node == "F59":
        vertical = _contains(texts["F53"], ("down", "up"))
        stated = str(texts["F54"]).strip()
        return "positive" if vertical in stated or stated in vertical else "negative"
    if node == "F60":
        lateral = _contains(texts["F53"], ("left", "right"))
        stated = str(texts["F55"]).strip()
        return "positive" if lateral == stated or lateral in stated else "negative"
    raise ValueError(node)


def _word(text: str) -> str:
    raw = str(text).strip()
    if raw not in _POINT:
        raise ValueError(f"absolute direction is not recognized: {text}")
    return _POINT[raw]


def _vector(text: str) -> tuple[int, int, int]:
    word = _word(text)
    if word not in _DIR:
        raise ValueError(f"absolute direction has no vector: {text}")
    return _DIR[word]


def _cross_word(left: str, right: str) -> str:
    vec = _cross(_vector(left), _vector(right))
    for word, item in _DIR.items():
        if item == vec:
            return word
    raise ValueError(f"{left} × {right} is not one of the six absolute directions")


def _wrist_frame(texts: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    found = {}
    for node, role in (("F4", "upward"), ("F5", "forward"), ("F6", "right")):
        raw = str(texts.get(node, "")).strip()
        negative = raw.startswith("-")
        name = raw[1:] if negative else raw
        if name not in {"X_W", "Y_W", "Z_W"}:
            raise ValueError(f"{raw} of {node} is not a wrist axis")
        vector = _DIR[role]
        if negative:
            vector = tuple(-item for item in vector)
        if name in found:
            raise ValueError("two absolute directions were written as the same wrist axis")
        found[name] = vector
    if set(found) != {"X_W", "Y_W", "Z_W"}:
        raise ValueError("the three wrist axes are incomplete")
    return found


def _components(frame: dict[str, tuple[int, int, int]], direction: str) -> str:
    target = _vector(direction)
    comps = [_dot(frame[name], target) for name in ("X_W", "Y_W", "Z_W")]
    rebuilt = tuple(
        sum(comps[index] * frame[name][axis] for index, name in enumerate(("X_W", "Y_W", "Z_W")))
        for axis in range(3)
    )
    if rebuilt != target:
        raise ValueError(f"the three wrist axes cannot represent {direction}")
    return "[" + ", ".join(str(item) for item in comps) + "]"


def _wrist_axis_for_role(texts: dict[str, str], role_id: str) -> str:
    role = str(texts.get(role_id, "")).strip()
    slot = {"normal": "F4", "axial": "F5", "lateral": "F6"}.get(role)
    if slot is None:
        raise ValueError(f"the absolute-direction axis is not axial, normal, or lateral: {role}")
    if _ROLE_OF[role] != {"F4": "upward", "F5": "forward", "F6": "right"}[slot]:
        raise ValueError(role)
    raw = str(texts.get(slot, "")).strip()
    name = raw[1:] if raw.startswith("-") else raw
    if name not in {"X_W", "Y_W", "Z_W"}:
        raise ValueError(f"{raw} for {role} is not a wrist axis")
    return name


def _require_same_axis(chosen: str, binding: str) -> None:
    if str(chosen).strip() != str(binding).strip():
        raise ValueError(f"the written axis {chosen} is not the {binding} for that absolute direction")


def _bound_direction(binding: str, role: str) -> str:
    base = _ROLE_OF[role]
    raw = str(binding).strip()
    if raw.startswith("-"):
        return _OPP[base]
    return base


def _tool_slot(text: str) -> tuple[int, int]:
    raw = str(text).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    index = {"X_T": 0, "Y_T": 1, "Z_T": 2}.get(name)
    if index is None:
        raise ValueError(f"not a tool axis: {text}")
    return index, (-1 if negative else 1)


def _apply_sign(expr: str, sign: int) -> str:
    text = str(expr).strip()
    if sign > 0:
        return text
    if text.startswith("-"):
        return text[1:].strip()
    return f"-({text})"


def _join_offset(base: str, extra: str) -> str:
    extra = str(extra).strip()
    if extra.startswith("-"):
        return f"{base} - {extra[1:].strip()}"
    return f"{base} + {extra}"


def _columns_to_matrix(first: str, second: str, third: str) -> str:
    columns = [_parse_vector(item) for item in (first, second, third)]
    return _show_matrix(sp.Matrix.hstack(*columns))


def _component_change(toward: str, axis_direction: str) -> str:
    toward_word = _word(toward)
    axis_word = _word(axis_direction)
    if toward_word == axis_word:
        return "increases"
    if toward_word == _OPP[axis_word]:
        return "decreases"
    raise ValueError("the offset direction is not collinear with the axis positive direction")


def _sweep_expr(parameter: str, change: str) -> str:
    if change == "increases":
        sign = 1
    elif change == "decreases":
        sign = -1
    else:
        raise ValueError(f"the component increase or decrease is not recognized: {change}")
    param = str(parameter).strip()
    if "sweep" in param:
        body = "(b/2)*tan(Λ)"
    elif "dihedral" in param:
        body = "(b/2)*tan(Γ)"
    else:
        raise ValueError(f"the fore-aft position does not use this shape parameter: {parameter}")
    return body if sign > 0 else f"-{body}"


def _lateral_expr(direction: str, parameter: str) -> str:
    word = _word(direction)
    if word == "right":
        sign = 1
    elif word == "left":
        sign = -1
    else:
        raise ValueError(f"the lateral coordinate positive direction is not left or right: {direction}")
    param = str(parameter).strip()
    if param == "tip spacing b":
        body = "b/2"
    elif "dihedral" in param:
        body = "(b/2)*cos(Γ)"
    else:
        raise ValueError(f"the lateral coordinate does not use this shape parameter: {parameter}")
    if sign > 0:
        return body
    return "-b/2" if body == "b/2" else f"-({body})"


def _height_expr(axis_direction: str, higher: str) -> str:
    axis_word = _word(axis_direction)
    higher_word = _word(higher)
    if axis_word == higher_word:
        return "(b/2)*tan(Γ)"
    if axis_word == _OPP[higher_word]:
        return "-(b/2)*tan(Γ)"
    raise ValueError("the vertical offset is not collinear with the normal positive direction")


def _root_term(choice: str) -> str:
    root = _ROOT_SYMBOL.get(str(choice).strip())
    if root is None:
        raise ValueError(f"the wing-root longitudinal coordinate is not recognized: {choice}")
    if root != "x_r":
        return root
    declaration = manuscript_path().read_text(encoding="utf-8")
    if "It is positive rearward." in declaration:
        return "-x_r"
    return root


def _wing_vector(texts: dict[str, str]) -> str:
    root = _root_term(texts["F76"])
    placed: dict[int, str] = {}

    def put(axis: str, expr: str) -> None:
        index, sign = _tool_slot(axis)
        if index in placed:
            raise ValueError("two offsets were written onto the same tool axis")
        placed[index] = _apply_sign(expr, sign)

    put(texts["F72"], _join_offset(root, texts["F75"]))
    put(texts["F81"], texts["F84"])
    put(texts["F77"], texts["F80"])
    if set(placed) != {0, 1, 2}:
        raise ValueError("the tool coordinate does not fill all three axes")
    return "[" + ", ".join(placed[index] for index in range(3)) + "]"


def _distance(texts: dict[str, str]) -> str:
    left = _LENGTH.get(str(texts["F88"]).strip())
    right = _LENGTH.get(str(texts["F89"]).strip())
    if left is None or right is None:
        raise ValueError("the tool length or the mount distance is not recognized")
    if left == right:
        raise ValueError("the tool length and the mount distance measure the same segment")
    if left == "L + L_mount" or right == "L + L_mount":
        expr = "L + L_mount"
    else:
        order = {"L": 0, "L_mount": 1, "L_in": 2}
        expr = " + ".join(sorted((left, right), key=lambda item: order[item]))
    mode = str(texts["F90"]).strip()
    if mode.startswith("already included"):
        return expr
    if mode.startswith("add it outside"):
        return expr if "L_in" in expr else expr + " + L_in"
    if mode.startswith("subtract"):
        return expr + " - L_in"
    raise ValueError(f"the treatment of the inserted mount length is not recognized: {mode}")


def _offset_vector(length: str, axis: str, other: str) -> str:
    index, sign = _tool_slot(axis)
    comps = [str(other).strip(), str(other).strip(), str(other).strip()]
    comps[index] = _apply_sign(length, sign)
    return "[" + ", ".join(comps) + "]"


def _install_from_directions(forward: str, up: str, right: str) -> str:
    frame = _wrist_frame({"F4": "Y_W", "F5": "X_W", "F6": "Z_W"})
    return _columns_to_matrix(
        _components(frame, forward),
        _components(frame, up),
        _components(frame, right),
    )


def _elem_rotation(axis: str, hand: str, symbol: str) -> str:
    positive = {
        "X_W": (
            f"[[1, 0, 0], [0, cos({symbol}), -sin({symbol})], "
            f"[0, sin({symbol}), cos({symbol})]]"
        ),
        "Y_W": (
            f"[[cos({symbol}), 0, sin({symbol})], [0, 1, 0], "
            f"[-sin({symbol}), 0, cos({symbol})]]"
        ),
        "Z_W": (
            f"[[cos({symbol}), -sin({symbol}), 0], [sin({symbol}), cos({symbol}), 0], "
            "[0, 0, 1]]"
        ),
    }
    negative = {
        "X_W": (
            f"[[1, 0, 0], [0, cos({symbol}), sin({symbol})], "
            f"[0, -sin({symbol}), cos({symbol})]]"
        ),
        "Y_W": (
            f"[[cos({symbol}), 0, -sin({symbol})], [0, 1, 0], "
            f"[sin({symbol}), 0, cos({symbol})]]"
        ),
        "Z_W": (
            f"[[cos({symbol}), sin({symbol}), 0], [-sin({symbol}), cos({symbol}), 0], "
            "[0, 0, 1]]"
        ),
    }
    table = positive if hand == "same" else negative if hand == "opposite" else None
    if table is None or axis not in table:
        raise ValueError(f"the joint matrix cannot be written from axis {axis} and positive sense {hand}")
    return table[axis]


def _rotation_matrix(texts: dict[str, str], node: str) -> str:
    axis_id, hand_id, turn_id, moving, symbol = _ROTATION[node]
    expected = _joint_turn(texts, axis_id, hand_id, moving)
    actual = str(texts[turn_id]).strip()
    if actual != expected:
        raise ValueError(f"the destination {actual} disagrees with {expected} implied by the joint axis and positive sense")
    axis = str(texts[axis_id]).strip()
    return _elem_rotation(axis, str(texts[hand_id]).strip(), symbol)


def _copy_matrix(text: str) -> str:
    return _show_matrix(_parse_literal_matrix(text))


def _trig_pair(angle: int) -> tuple[int, int]:
    if angle == 0:
        return 1, 0
    if angle == 90:
        return 0, 1
    if angle == -90:
        return 0, -1
    if angle == 180:
        return -1, 0
    raise ValueError(angle)


def _single_axis_matrix(axis: str, angle: int) -> sp.Matrix:
    cosine, sine = _trig_pair(angle)
    if axis == "X_W":
        return sp.Matrix([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    if axis == "Y_W":
        return sp.Matrix([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    if axis == "Z_W":
        return sp.Matrix([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    raise ValueError(axis)


def _axis_angle(matrix_text: str) -> tuple[str, int]:
    matrix = sp.simplify(_parse_literal_matrix(matrix_text))
    for axis in ("X_W", "Y_W", "Z_W"):
        for angle in (0, 90, -90, 180):
            if sp.simplify(matrix - _single_axis_matrix(axis, angle)) == sp.zeros(3):
                return axis, angle
    raise ValueError("the installation matrix is not a 0°, ±90°, or 180° rotation about one wrist axis")


def _combined_angle(install_text: str, hand: str) -> str:
    raw = str(install_text).strip().replace("°", "")
    if not raw.lstrip("-").isdigit():
        raise ValueError(f"the installation angle is not an integer number of degrees: {install_text}")
    signed = _hand_degree(hand, int(raw))
    return f"{-60 + signed}°"


def _axis_letter(axis: str) -> str:
    raw = str(axis).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    letter = {"X_W": "X", "Y_W": "Y", "Z_W": "Z"}.get(raw)
    if letter is None:
        raise ValueError(f"not a wrist axis: {axis}")
    return letter


def _hand_degree(hand: str, joint_degrees: int) -> int:
    text = str(hand).strip()
    if text == "same":
        return joint_degrees
    if text == "opposite":
        return -joint_degrees
    raise ValueError(f"the joint positive sense is not same or opposite: {hand}")


def _pose_phrase(q2_axis: str, q2_hand: str, install_axis: str, q3_hand: str, combined: str) -> str:
    number = str(combined).strip().replace("°", "")
    if not number.lstrip("-").isdigit():
        raise ValueError(f"the combined angle cannot be written: {combined}")
    q2 = _hand_degree(q2_hand, 30)
    roll = _hand_degree(q3_hand, int(number))
    return (
        f"{q2}° about {_axis_letter(q2_axis)} times {roll}° "
        f"about {_axis_letter(install_axis)}"
    )


def _tool_column(axis: str) -> str:
    raw = str(axis).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    column = {"X_T": "column 1", "Y_T": "column 2", "Z_T": "column 3"}.get(raw)
    if column is None:
        raise ValueError(f"tool lateral did not land on a tool axis: {axis}")
    return column


def _normal_component_sign(matrix_text: str, column_word: str, axis_name: str) -> str:
    matrix = _parse_literal_matrix(matrix_text)
    column = {"column 1": 0, "column 2": 1, "column 3": 2}.get(str(column_word).strip())
    if column is None:
        raise ValueError(f"the column index is not recognized: {column_word}")
    raw = str(axis_name).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    row = {"X_B": 0, "Y_B": 1, "Z_B": 2}.get(name)
    if row is None:
        raise ValueError(f"the base normal did not land on a base axis: {axis_name}")
    value = sp.N(sp.simplify(matrix[row, column]))
    if negative:
        value = -value
    if value == 0:
        raise ValueError("that normal component is 0")
    return "negative" if value < 0 else "positive"


def _normal_bias(sign: str) -> str:
    if sign == "negative":
        return "downward"
    if sign == "positive":
        return "upward"
    raise ValueError(f"the component sign is not positive or negative: {sign}")


def derive(_model, node_id: str, texts: dict[str, str]) -> Execution:
    node_id = str(node_id)
    if node_id not in NODE_IDS:
        return Execution("contradictory", reason=node_id)
    if node_id in DECLARATION_ROOTS or not input_ids(node_id):
        return Execution("derived", gold_texts()[node_id])
    try:
        if node_id in {"F11", "F16", "F21"}:
            role_id = {"F11": "F10", "F16": "F15", "F21": "F20"}[node_id]
            return Execution("derived", _wrist_axis_for_role(texts, role_id))
        if node_id in _TURN:
            axis_id, hand_id, moving = _TURN[node_id]
            return Execution("derived", _joint_turn(texts, axis_id, hand_id, moving))
        if node_id in _ROTATION:
            return Execution("derived", _rotation_matrix(texts, node_id))
        if node_id in {"F41", "F42", "F43", "F44"}:
            parent = input_ids(node_id)[0]
            return Execution("derived", _copy_matrix(texts[parent]))
        if node_id == "F34":
            axis, _angle = _axis_angle(texts["F33"])
            return Execution("derived", axis)
        if node_id == "F35":
            _axis, angle = _axis_angle(texts["F33"])
            return Execution("derived", f"{angle}°")
        if node_id == "F39":
            return Execution("derived", _combined_angle(texts["F35"], texts["F22"]))
        if node_id == "F40":
            return Execution("derived", _pose_phrase(
                texts["F16"], texts["F17"], texts["F34"], texts["F22"], texts["F39"],
            ))
        if node_id == "F61":
            return Execution("derived", _tool_column(texts["F9"]))
        if node_id == "F62":
            return Execution("derived", _normal_component_sign(texts["F45"], texts["F61"], texts["F1"]))
        if node_id == "F63":
            return Execution("derived", _normal_bias(texts["F62"]))
        if node_id == "F27":
            return Execution("derived", _word(texts["F26"]))
        if node_id == "F28":
            return Execution("derived", _cross_word(texts["F25"], texts["F26"]))
        if node_id == "F29":
            return Execution("derived", _word(texts["F28"]))
        if node_id == "F30":
            return Execution("derived", _components(_wrist_frame(texts), texts["F25"]))
        if node_id == "F31":
            return Execution("derived", _components(_wrist_frame(texts), texts["F27"]))
        if node_id == "F32":
            return Execution("derived", _components(_wrist_frame(texts), texts["F29"]))
        if node_id == "F33":
            return Execution("derived", _columns_to_matrix(texts["F30"], texts["F31"], texts["F32"]))
        if node_id in {"F50", "F52", "F53", "F59", "F60"}:
            return Execution("derived", _task(node_id, texts))
        if node_id == "F45":
            return Execution("derived", _pose_from_factors(
                texts["F41"], texts["F42"], texts["F43"], texts["F44"], texts["F37"],
            ))
        if node_id == "F48":
            return Execution("derived", _tool_components(texts["F45"], texts["F47"]))
        if node_id in {"F72", "F77", "F81"}:
            return Execution("derived", str(texts[{"F72": "F66", "F77": "F68", "F81": "F67"}[node_id]]).strip())
        if node_id == "F73":
            _require_same_axis(texts["F72"], texts["F66"])
            return Execution("derived", _bound_direction(texts["F66"], "axial"))
        if node_id == "F74":
            return Execution("derived", _component_change(texts["F71"], texts["F73"]))
        if node_id == "F75":
            return Execution("derived", _sweep_expr(texts["F70"], texts["F74"]))
        if node_id == "F78":
            _require_same_axis(texts["F77"], texts["F68"])
            return Execution("derived", _bound_direction(texts["F68"], "lateral"))
        if node_id == "F80":
            return Execution("derived", _lateral_expr(texts["F78"], texts["F79"]))
        if node_id == "F82":
            _require_same_axis(texts["F81"], texts["F67"])
            return Execution("derived", _bound_direction(texts["F67"], "normal"))
        if node_id == "F84":
            return Execution("derived", _height_expr(texts["F82"], texts["F83"]))
        if node_id == "F85":
            return Execution("derived", _wing_vector(texts))
        if node_id == "F92":
            return Execution("derived", _distance(texts))
        if node_id == "F94":
            direction = _word(texts["F93"])
            if direction not in _TOOL_AXIS:
                raise ValueError(f"the tool offset direction has no axis: {direction}")
            return Execution("derived", _TOOL_AXIS[direction])
        if node_id == "F96":
            return Execution("derived", _offset_vector(texts["F92"], texts["F94"], texts["F95"]))
        if node_id == "F98":
            return Execution("derived", _translate_expr(texts["F96"]))
        if node_id == "I7":
            return Execution("derived", _first_row_numeric(texts["I6"]))
        if node_id == "I9":
            return Execution("derived", _d_from_rows(texts["I7"], texts["I8"]))
        if node_id == "I12":
            return Execution("derived", _cross_word(texts["I10"], texts["I11"]))
        if node_id == "I13":
            return Execution("derived", _install_from_directions(texts["I10"], texts["I11"], texts["I12"]))
        if node_id == "I16":
            return Execution("derived", _joints_from_pose(
                texts["I9"], texts["I14"], texts["I13"], texts["I15"],
            ))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    return Execution("contradictory", reason=f"{node_id} no calculator")


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
        raise RuntimeError("declaration roots and derived nodes do not cover kinematics")
    wired = {node for node in NODE_IDS if input_ids(node)}
    if wired != set(DERIVED_NODES):
        raise RuntimeError("calculator parents and derived nodes disagree")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"{node} gold regression failed: {got.reason}")
        fmt = rows[node]["format"] if not rows[node]["options"] else "expression"
        if not answers_equal(got.matrix, gold[node], fmt):
            raise RuntimeError(f"{node} gold regression value mismatch: {got.matrix!r} vs {gold[node]!r}")
    bad = derive(None, "F45", {**gold, "F41": "not-a-matrix"})
    if bad.ok:
        raise RuntimeError("an illegal upstream still produced F45")
    changed = answer_under("F48", {"F47": "D times the base-frame task direction"})
    if changed is None or answers_equal(changed, gold["F48"], "vector"):
        changed = answer_under("F45", {"F42": gold["F41"]})
        if changed is None or answers_equal(changed, gold["F45"], "matrix"):
            raise RuntimeError("rewriting an upstream node did not change the terminal")
    if answer_under("F11", {"F10": "axial"}) != "X_W":
        raise RuntimeError("the rotation order was not written into the first joint axis")
    if answer_under("F33", {"F30": "[0, 1, 0]"}) == gold["F33"]:
        raise RuntimeError("the installation column was not written into the installation matrix")
    if answer_under("F85", {"F77": "Y_T"}) is not None and answers_equal(
        answer_under("F85", {"F77": "Y_T"}), gold["F85"], "vector",
    ):
        raise RuntimeError("the wing-tip axis was not written into the tool-frame coordinate")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"derived exceptions were not cleared: {DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"terminal {node} is out of range")
    print(
        f"stage 3 calculators: kinematics gold regression passed, declaration roots {len(DECLARATION_ROOTS)}, "
        f"derived edges {sum(1 for n in NODE_IDS if input_ids(n))}.",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
