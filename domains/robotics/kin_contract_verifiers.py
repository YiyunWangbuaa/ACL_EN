# -*- coding: utf-8 -*-
"""Per-blank calculators for robotics kinematics F*/I*."""
from __future__ import annotations

import sympy as sp

from domains.robotics.contract_executor import Execution
from domains.robotics.kin_answer_form import answers_equal
from domains.robotics.kin_manuscript import gold_texts, questions
from domains.robotics.kin_pipeline_scope import NODE_IDS
from domains.robotics.kin_signed_lists import DECLARATION_ROOTS, DERIVED_EXCEPTIONS, DERIVED_NODES, T_FINAL

_DIR = {
    "前方": (1, 0, 0), "后方": (-1, 0, 0),
    "右侧": (0, 1, 0), "左侧": (0, -1, 0),
    "下方": (0, 0, 1), "上方": (0, 0, -1),
}
_OPP = {word: name for word, name in (
    ("前方", "后方"), ("后方", "前方"),
    ("右侧", "左侧"), ("左侧", "右侧"),
    ("下方", "上方"), ("上方", "下方"),
)}
_TURN = {
    "F13": ("F11", "F12", "前方"),
    "F18": ("F16", "F17", "前方"),
    "F23": ("F21", "F22", "上方"),
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
    "F39": ("F35",),
    "F40": ("F16", "F34", "F39"),
    "F41": ("F14",),
    "F42": ("F19",),
    "F43": ("F24",),
    "F44": ("F33",),
    "F45": ("F37", "F41", "F42", "F43", "F44"),
    "F48": ("F45", "F47"),
    "F50": ("F49",),
    "F52": ("F51",),
    "F53": ("F50", "F52"),
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
    "F14": ("F11", "F12", "F13", "前方", "q1"),
    "F19": ("F16", "F17", "F18", "前方", "q2"),
    "F24": ("F21", "F22", "F23", "上方", "q3"),
}
_ROLE_OF = {"法向": "上方", "轴向": "前方", "侧向": "右侧"}
_POINT = {
    "任务主方向": "前方",
    "基座上方": "上方",
    "基座右侧": "右侧",
    "基座下方": "下方",
    "前方": "前方", "后方": "后方",
    "上方": "上方", "下方": "下方",
    "右侧": "右侧", "左侧": "左侧",
}
_LENGTH = {
    "工具顶点到工具底部": "L",
    "工具底部到球腕原点": "L_mount",
    "工具顶点到球腕原点": "L + L_mount",
    "工具安装杆插入工具内部的长度": "L_in",
}
_TOOL_AXIS = {"前方": "X_T", "后方": "-X_T", "上方": "Y_T", "下方": "-Y_T", "右侧": "Z_T", "左侧": "-Z_T"}
_ROOT_SYMBOL = {
    "x_r": "x_r",
    "左右翼尖间距b": "b",
    "前缘后掠角Λ": "Λ",
    "上反角Γ": "Γ",
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
        raise ValueError(f"姿态乘积不是 R*S 或 S*R：{rule}")
    matrix = ordered[0] * ordered[1] * ordered[2] * ordered[3]
    return _show_matrix(sp.simplify(matrix.subs(_ANGLE_SUBS)))


def _tool_components(d_text: str, rule_text: str) -> str:
    matrix = _parse_literal_matrix(d_text)
    ref = sp.Matrix([1, 0, 0])
    rule = str(rule_text)
    if "转置" in rule:
        vector = sp.simplify(matrix.T * ref)
    elif "D乘" in rule.replace(" ", ""):
        vector = sp.simplify(matrix * ref)
    else:
        raise ValueError(f"无法解释映射规则：{rule_text}")
    return _show_vector(vector)


def _translate_expr(offset_text: str) -> str:
    return f"D*{str(offset_text).strip()}"


def _first_row_numeric(choice_text: str) -> str:
    alpha = sp.atan(sp.Rational(1, 2))
    beta = sp.asin(sp.Rational(1, 4))
    a = sp.simplify(sp.cos(alpha) * sp.cos(beta))
    bv = sp.simplify(-sp.sin(alpha) * sp.cos(beta))
    c = sp.simplify(sp.sin(beta))
    mapping = {
        "[a, bv, c]": [a, bv, c],
        "[a, c, bv]": [a, c, bv],
        "[bv, a, c]": [bv, a, c],
    }
    if choice_text not in mapping:
        raise ValueError(f"未知第一行形式：{choice_text}")
    return _show_vector(sp.Matrix(mapping[choice_text]))


def _d_from_rows(row1_text: str, row2_choice: str) -> str:
    row1 = _parse_vector(row1_text)
    a, bv, c = row1[0], row1[1], row1[2]
    gamma = sp.asin(sp.Rational(1, 2))
    d = sp.simplify(bv * sp.cos(gamma) - c * sp.sin(gamma))
    n = sp.sqrt(sp.simplify(a**2 + d**2))
    if row2_choice.startswith("[-d/N"):
        row2 = sp.Matrix([-d / n, a * sp.cos(gamma) / n, -a * sp.sin(gamma) / n])
    elif row2_choice.startswith("[d/N"):
        row2 = sp.Matrix([d / n, a * sp.cos(gamma) / n, a * sp.sin(gamma) / n])
    else:
        row2 = sp.Matrix([-d / n, a * sp.sin(gamma) / n, a * sp.cos(gamma) / n])
    row3 = sp.simplify(row1.cross(row2))
    return _show_matrix(sp.Matrix.vstack(row1.T, sp.simplify(row2).T, row3.T))


def _joints_from_pose(d_text: str, erase_choice: str, s_text: str | None = None) -> str:
    d = _parse_literal_matrix(d_text)
    if s_text is None:
        s = sp.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    else:
        s = _parse_literal_matrix(s_text)
    choice = erase_choice.replace(" ", "")
    if "S^T*D" in choice:
        h = sp.simplify(s.T * d)
    elif "D*S^T" in choice:
        h = sp.simplify(d * s.T)
    elif choice == "D*S":
        h = sp.simplify(d * s)
    else:
        raise ValueError(f"消去安装的乘积无法识别：{erase_choice}")
    q1 = sp.deg(sp.atan2(-h[2, 0], h[0, 0]))
    q2 = sp.deg(sp.atan2(h[1, 0], sp.sqrt(h[0, 0] ** 2 + h[2, 0] ** 2)))
    q3 = sp.deg(sp.atan2(-h[1, 2], h[1, 1]))
    vals = [int(sp.N(q)) for q in (q1, q2, q3)]
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
    for node, word in (("F4", "上方"), ("F5", "前方"), ("F6", "右侧")):
        raw = str(texts.get(node, "")).strip()
        negative = raw.startswith("-")
        name = raw[1:] if negative else raw
        vector = _DIR[word]
        if negative:
            vector = tuple(-item for item in vector)
        if not name or name in found:
            raise ValueError(f"{node} 的球腕坐标轴无法和其他轴组成一组。")
        found[name] = vector
    return found


def _joint_turn(texts: dict[str, str], axis_id: str, hand_id: str, moving_word: str) -> str:
    frame = _wrist(texts)
    axis = str(texts.get(axis_id, "")).strip()
    if axis.startswith("-"):
        axis = axis[1:]
    if axis not in frame:
        raise ValueError(f"关节轴 {axis} 不在球腕三轴里。")
    hand = str(texts.get(hand_id, "")).strip()
    if hand not in {"相同", "相反"}:
        raise ValueError("关节正方向与右手定则的关系不是相同或相反。")
    moving = _DIR[moving_word]
    hits = [
        word for word, toward in _DIR.items()
        if _dot(_cross(frame[axis], moving), toward) > 0
    ]
    if len(hits) != 1:
        raise ValueError("正转没有唯一的偏转方向。")
    return _OPP[hits[0]] if hand == "相反" else hits[0]


def _contains(text: str, words: tuple[str, ...]) -> str:
    found = [word for word in words if word in str(text)]
    if len(found) != 1:
        raise ValueError(f"方向里没有唯一的{'/'.join(words)}：{text}")
    return found[0]


def _task(node: str, texts: dict[str, str]) -> str:
    if node == "F50":
        return _OPP[str(texts["F49"]).strip()]
    if node == "F52":
        side = str(texts["F51"]).strip()
        if side not in _DIR:
            raise ValueError(f"分量正方向无法识别：{side}")
        return side
    if node == "F53":
        return f"{texts['F50'].strip()}且{texts['F52'].strip()}"
    if node == "F59":
        vertical = _contains(texts["F53"], ("上方", "下方"))
        return "正" if vertical == str(texts["F54"]).strip() else "负"
    if node == "F60":
        lateral = _contains(texts["F53"], ("左侧", "右侧"))
        return "正" if lateral == str(texts["F55"]).strip() else "负"
    raise ValueError(node)


def _word(text: str) -> str:
    raw = str(text).strip()
    if raw not in _POINT:
        raise ValueError(f"绝对方向无法识别：{text}")
    return _POINT[raw]


def _vector(text: str) -> tuple[int, int, int]:
    word = _word(text)
    if word not in _DIR:
        raise ValueError(f"绝对方向没有向量：{text}")
    return _DIR[word]


def _cross_word(left: str, right: str) -> str:
    vec = _cross(_vector(left), _vector(right))
    for word, item in _DIR.items():
        if item == vec:
            return word
    raise ValueError(f"{left} × {right} 不是六个绝对方向之一")


def _wrist_frame(texts: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    found = {}
    for node, role in (("F4", "上方"), ("F5", "前方"), ("F6", "右侧")):
        raw = str(texts.get(node, "")).strip()
        negative = raw.startswith("-")
        name = raw[1:] if negative else raw
        if name not in {"X_W", "Y_W", "Z_W"}:
            raise ValueError(f"{node} 的 {raw} 不是球腕坐标轴")
        vector = _DIR[role]
        if negative:
            vector = tuple(-item for item in vector)
        if name in found:
            raise ValueError("两根绝对方向写成了同一根球腕坐标轴")
        found[name] = vector
    if set(found) != {"X_W", "Y_W", "Z_W"}:
        raise ValueError("球腕三轴不齐")
    return found


def _components(frame: dict[str, tuple[int, int, int]], direction: str) -> str:
    target = _vector(direction)
    comps = [_dot(frame[name], target) for name in ("X_W", "Y_W", "Z_W")]
    rebuilt = tuple(
        sum(comps[index] * frame[name][axis] for index, name in enumerate(("X_W", "Y_W", "Z_W")))
        for axis in range(3)
    )
    if rebuilt != target:
        raise ValueError(f"球腕三轴不能表示 {direction}")
    return "[" + ", ".join(str(item) for item in comps) + "]"


def _wrist_axis_for_role(texts: dict[str, str], role_id: str) -> str:
    role = str(texts.get(role_id, "")).strip()
    slot = {"法向": "F4", "轴向": "F5", "侧向": "F6"}.get(role)
    if slot is None:
        raise ValueError(f"绝对方向轴不是轴向、法向或侧向：{role}")
    if _ROLE_OF[role] != {"F4": "上方", "F5": "前方", "F6": "右侧"}[slot]:
        raise ValueError(role)
    raw = str(texts.get(slot, "")).strip()
    name = raw[1:] if raw.startswith("-") else raw
    if name not in {"X_W", "Y_W", "Z_W"}:
        raise ValueError(f"{role} 对应的 {raw} 不是球腕坐标轴")
    return name


def _require_same_axis(chosen: str, binding: str) -> None:
    if str(chosen).strip() != str(binding).strip():
        raise ValueError(f"写入的坐标轴 {chosen} 不是该绝对方向对应的 {binding}")


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
        raise ValueError(f"不是工具坐标轴：{text}")
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
        return "增大"
    if toward_word == _OPP[axis_word]:
        return "减小"
    raise ValueError("偏移方向和坐标轴正方向不共线")


def _sweep_expr(parameter: str, change: str) -> str:
    if change == "增大":
        sign = 1
    elif change == "减小":
        sign = -1
    else:
        raise ValueError(f"分量增减无法识别：{change}")
    param = str(parameter).strip()
    if "后掠" in param:
        body = "(b/2)*tan(Λ)"
    elif "上反" in param:
        body = "(b/2)*tan(Γ)"
    else:
        raise ValueError(f"前后位置不用这个外形参数：{parameter}")
    return body if sign > 0 else f"-{body}"


def _lateral_expr(direction: str, parameter: str) -> str:
    word = _word(direction)
    if word == "右侧":
        sign = 1
    elif word == "左侧":
        sign = -1
    else:
        raise ValueError(f"左右坐标的正方向不是左侧或右侧：{direction}")
    param = str(parameter).strip()
    if param == "左右翼尖间距b":
        body = "b/2"
    elif "上反" in param:
        body = "(b/2)*cos(Γ)"
    else:
        raise ValueError(f"左右坐标不用这个外形参数：{parameter}")
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
    raise ValueError("高低偏移方向和法向正方向不共线")


def _wing_vector(texts: dict[str, str]) -> str:
    root = _ROOT_SYMBOL.get(str(texts["F76"]).strip())
    if root is None:
        raise ValueError(f"翼根纵向坐标无法识别：{texts['F76']}")
    placed: dict[int, str] = {}

    def put(axis: str, expr: str) -> None:
        index, sign = _tool_slot(axis)
        if index in placed:
            raise ValueError("两个偏移写入同一根工具坐标轴")
        placed[index] = _apply_sign(expr, sign)

    put(texts["F72"], _join_offset(root, texts["F75"]))
    put(texts["F81"], texts["F84"])
    put(texts["F77"], texts["F80"])
    if set(placed) != {0, 1, 2}:
        raise ValueError("工具坐标没有写满三根轴")
    return "[" + ", ".join(placed[index] for index in range(3)) + "]"


def _distance(texts: dict[str, str]) -> str:
    left = _LENGTH.get(str(texts["F88"]).strip())
    right = _LENGTH.get(str(texts["F89"]).strip())
    if left is None or right is None:
        raise ValueError("工具长度或安装距离无法识别")
    if left == right:
        raise ValueError("工具长度和安装距离量的是同一段")
    if left == "L + L_mount" or right == "L + L_mount":
        expr = "L + L_mount"
    else:
        order = {"L": 0, "L_mount": 1, "L_in": 2}
        expr = " + ".join(sorted((left, right), key=lambda item: order[item]))
    mode = str(texts["F90"]).strip()
    if "已经含" in mode:
        return expr
    if "要加" in mode:
        return expr if "L_in" in expr else expr + " + L_in"
    if "减去" in mode:
        return expr + " - L_in"
    raise ValueError(f"安装杆插入段的处理无法识别：{mode}")


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
    table = positive if hand == "相同" else negative if hand == "相反" else None
    if table is None or axis not in table:
        raise ValueError(f"关节矩阵无法由轴 {axis} 和正方向 {hand} 写出")
    return table[axis]


def _rotation_matrix(texts: dict[str, str], node: str) -> str:
    axis_id, hand_id, turn_id, moving, symbol = _ROTATION[node]
    expected = _joint_turn(texts, axis_id, hand_id, moving)
    actual = str(texts[turn_id]).strip()
    if actual != expected:
        raise ValueError(f"去向 {actual} 和关节轴、正方向推出的 {expected} 不一致")
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
    raise ValueError("安装矩阵不是绕单根球腕坐标轴的 0°、±90° 或 180°")


def _combined_angle(install_text: str) -> str:
    raw = str(install_text).strip().replace("°", "")
    if not raw.lstrip("-").isdigit():
        raise ValueError(f"安装转角不是整数度：{install_text}")
    return f"{-60 + int(raw)}°"


def _axis_letter(axis: str) -> str:
    raw = str(axis).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    letter = {"X_W": "X", "Y_W": "Y", "Z_W": "Z"}.get(raw)
    if letter is None:
        raise ValueError(f"不是球腕坐标轴：{axis}")
    return letter


def _pose_phrase(q2_axis: str, install_axis: str, combined: str) -> str:
    number = str(combined).strip().replace("°", "")
    if not number.lstrip("-").isdigit():
        raise ValueError(f"合成转角无法写入：{combined}")
    return f"绕 {_axis_letter(q2_axis)} 的 30° 乘绕 {_axis_letter(install_axis)} 的 {number}°"


def _tool_column(axis: str) -> str:
    raw = str(axis).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    column = {"X_T": "一列", "Y_T": "二列", "Z_T": "三列"}.get(raw)
    if column is None:
        raise ValueError(f"工具侧向没有落到工具坐标轴上：{axis}")
    return column


def _normal_component_sign(matrix_text: str, column_word: str, axis_name: str) -> str:
    matrix = _parse_literal_matrix(matrix_text)
    column = {"一列": 0, "二列": 1, "三列": 2}.get(str(column_word).strip())
    if column is None:
        raise ValueError(f"列号无法识别：{column_word}")
    raw = str(axis_name).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    row = {"X_B": 0, "Y_B": 1, "Z_B": 2}.get(name)
    if row is None:
        raise ValueError(f"基座法向没有落到基座坐标轴上：{axis_name}")
    value = sp.N(sp.simplify(matrix[row, column]))
    if negative:
        value = -value
    if value == 0:
        raise ValueError("该法向分量为 0")
    return "负" if value < 0 else "正"


def _normal_bias(sign: str) -> str:
    if sign == "负":
        return "下方"
    if sign == "正":
        return "上方"
    raise ValueError(f"分量符号不是正或负：{sign}")


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
            return Execution("derived", _combined_angle(texts["F35"]))
        if node_id == "F40":
            return Execution("derived", _pose_phrase(texts["F16"], texts["F34"], texts["F39"]))
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
            return Execution("derived", _bound_direction(texts["F66"], "轴向"))
        if node_id == "F74":
            return Execution("derived", _component_change(texts["F71"], texts["F73"]))
        if node_id == "F75":
            return Execution("derived", _sweep_expr(texts["F70"], texts["F74"]))
        if node_id == "F78":
            _require_same_axis(texts["F77"], texts["F68"])
            return Execution("derived", _bound_direction(texts["F68"], "侧向"))
        if node_id == "F80":
            return Execution("derived", _lateral_expr(texts["F78"], texts["F79"]))
        if node_id == "F82":
            _require_same_axis(texts["F81"], texts["F67"])
            return Execution("derived", _bound_direction(texts["F67"], "法向"))
        if node_id == "F84":
            return Execution("derived", _height_expr(texts["F82"], texts["F83"]))
        if node_id == "F85":
            return Execution("derived", _wing_vector(texts))
        if node_id == "F92":
            return Execution("derived", _distance(texts))
        if node_id == "F94":
            direction = _word(texts["F93"])
            if direction not in _TOOL_AXIS:
                raise ValueError(f"工具偏移方向没有坐标轴：{direction}")
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
            return Execution("derived", _joints_from_pose(texts["I9"], texts["I14"], texts["I13"]))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    return Execution("contradictory", reason=f"{node_id} 没有计算器")


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
        raise RuntimeError("声明根与派生节点没有覆盖运动学")
    wired = {node for node in NODE_IDS if input_ids(node)}
    if wired != set(DERIVED_NODES):
        raise RuntimeError("计算器父空和派生节点不一致")
    for node in NODE_IDS:
        got = derive(None, node, gold)
        if not got.ok:
            raise RuntimeError(f"{node} 金标回归失败：{got.reason}")
        fmt = rows[node]["format"] if not rows[node]["options"] else "表达式"
        if not answers_equal(got.matrix, gold[node], fmt):
            raise RuntimeError(f"{node} 金标回归值不一致：{got.matrix!r} vs {gold[node]!r}")
    bad = derive(None, "F45", {**gold, "F41": "not-a-matrix"})
    if bad.ok:
        raise RuntimeError("非法上游仍推出了 F45")
    changed = answer_under("F48", {"F47": "D乘基座系任务方向坐标"})
    if changed is None or answers_equal(changed, gold["F48"], "向量"):
        changed = answer_under("F45", {"F42": gold["F41"]})
        if changed is None or answers_equal(changed, gold["F45"], "矩阵"):
            raise RuntimeError("改写上游后终点没有变化")
    if answer_under("F11", {"F10": "轴向"}) != "X_W":
        raise RuntimeError("旋转顺序没有写进第一根关节轴")
    if answer_under("F33", {"F30": "[0, 1, 0]"}) == gold["F33"]:
        raise RuntimeError("安装列没有写进安装矩阵")
    if answer_under("F85", {"F77": "Y_T"}) is not None and answers_equal(
        answer_under("F85", {"F77": "Y_T"}), gold["F85"], "向量",
    ):
        raise RuntimeError("翼尖坐标轴没有写进工具系坐标")
    if DERIVED_EXCEPTIONS:
        raise RuntimeError(f"派生例外未清空：{DERIVED_EXCEPTIONS}")
    for node in T_FINAL:
        if node not in NODE_IDS:
            raise RuntimeError(f"终点 {node} 不在范围")
    print(
        f"阶段3 计算器：运动学金标回归通过，声明根 {len(DECLARATION_ROOTS)}，"
        f"派生边 {sum(1 for n in NODE_IDS if input_ids(n))}。",
        flush=True,
    )


if __name__ == "__main__":
    check_verifiers()
