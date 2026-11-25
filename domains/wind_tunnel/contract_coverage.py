# -*- coding: utf-8 -*-
"""Calculators for the A-class blanks that are not composed products.

Each one rebuilds the blank from the texts the model stated. A blank whose
wording is fixed by the sample declaration, rather than by an earlier blank,
reads that declaration. Gold texts must reproduce the generator's answer
before a result is used as evidence.
"""
from __future__ import annotations

import sympy as sp

from body_frame_model import (
    ALPHA,
    ALPHA_M,
    ALPHA_V,
    BETA,
    BETA_M,
    BETA_V,
    NEGATIVE_ROTATION,
    POSITIVE_ROTATION,
    _VECTOR,
    _dot,
)
from cloze_score import _parse_expr, _sympy_modules
from domains.wind_tunnel.contract_executor import Execution, _passive_rotation, recognized_left_handed
from domains.wind_tunnel.ref_point_generate import (
    FORCE_OF_ROLE,
    FORCE_SYMBOL,
    L_RT_SYM,
    MOMENT_CODE,
    XG_SYM,
    YG_SYM,
    ZG_SYM,
    _cross_expr,
    _dot_expr,
)
from domains.wind_tunnel.tp_tz_generate import GAMMA_M_SYM, GAMMA_TP_SYM
from domains.wind_tunnel.tz_fd_generate import _matrix_text, _role, _wing_about_nose

_BODY = ("XTZ", "YTZ", "ZTZ")
_WIND = ("XFD", "YFD", "ZFD")
_BALANCE = ("XTP", "YTP", "ZYP")
_FRAME_FAMILY = {
    "中间体轴系": "TZ",
    "机体系": "TZ",
    "体轴系": "TZ",
    "风轴系": "FD",
    "天平系": "TP",
}
_AXIS_FAMILY = {name: "TZ" for name in _BODY} | {name: "FD" for name in _WIND} | {name: "TP" for name in _BALANCE}
# 法向、轴向、侧向在机体绝对方向上的正方向。绑定文本只决定哪根坐标轴指向它。
_ABSOLUTE = {"法向": "body_up", "轴向": "body_forward", "侧向": "body_right"}
_BODY_DIRECTION = {
    "前面": "body_forward", "后面": "body_rearward",
    "上方": "body_up", "下方": "body_down",
    "左侧": "body_left", "右侧": "body_right",
}
# role 只命名。轴名从「当前坐标系那一族」的绑定读出，不固定在机体三元组上。
_AXIS_RULE = {
    "24": ("侧向", "23"),
    "32": ("法向", "31"),
    "41": ("轴向", "40"),
}
_SLOT_ORDER = ("法向", "轴向", "侧向")
_FAMILY_SLOTS = {
    "TZ": {"法向": "5", "轴向": "6", "侧向": "7"},
    "FD": {"法向": "2", "轴向": "3", "侧向": "4"},
    "TP": {"法向": "89", "轴向": "90", "侧向": "91"},
}
_READ_TRIADS = tuple(
    slots[role] for slots in _FAMILY_SLOTS.values() for role in _SLOT_ORDER
)
_ALLOWED = {"TZ": _BODY, "FD": _WIND, "TP": _BALANCE}
_TURN_RULE = {
    "25": ("24", "20", "alpha"),
    "33": ("32", "28", "beta"),
    "42": ("41", "35", "gamma"),
    "87": ("86", "85", "balance"),
}
_ABS = {"轴向": (1, 0, 0), "侧向": (0, 1, 0), "法向": (0, 0, -1)}
_ZH_OPPOSITE = {
    "上方": "下方", "下方": "上方",
    "左侧": "右侧", "右侧": "左侧",
    "前面": "后面", "后面": "前面",
}
_ROLE_OF = {
    "上方": "法向", "下方": "法向",
    "前面": "轴向", "后面": "轴向",
    "右侧": "侧向", "左侧": "侧向",
}
_BODY_ROLES = {"法向": "5", "轴向": "6", "侧向": "7"}
_WIND_ROLES = {"法向": "120", "侧向": "121", "轴向": "122"}
_FIXED = {"69": "arctan", "75": "arcsin", "182": "L + L_t - L_r"}
_DECLARED = {"20": "mechanism-alpha", "21": "theory-alpha", "28": "mechanism-beta", "29": "theory-beta", "155": "theory-alpha", "156": "theory-beta"}
_SIGN_OF = {"XTZ": "62", "YTZ": "63", "ZTZ": "64", "XFD": "161", "YFD": "162", "ZFD": "163"}
_BETA_SIGN = {"XTZ": "65", "YTZ": "66", "ZTZ": "67"}
_SLOT = {"Xg": "XTZ", "Yg": "YTZ", "Zg": "ZTZ"}
_ENGINEERING = {"XTZ": "Xg", "YTZ": "Yg", "ZTZ": "Zg"}
_PATH = (("185", "原点"), ("186", "力矩参考点"), ("187", "原点"), ("188", "力矩参考点"))
_MOMENT_KEY = {str(189 + index): key for index, key in enumerate(("MCN_CA", "MCZ_CA", "MCA_CZ", "MCN_CZ", "MCA_CN", "MCZ_CN"))}
_XR = sp.symbols("Xr", real=True)
_CANON = {"Xg": XG_SYM, "Yg": YG_SYM, "Zg": ZG_SYM, "Xr": _XR, "L_RT": L_RT_SYM, **FORCE_SYMBOL}


class _Bad(RuntimeError):
    pass


def coverage_validator(node_id: str) -> str | None:
    spec = _spec(node_id)
    return None if spec is None else spec[0]


def coverage_inputs(node_id: str) -> tuple[str, ...]:
    spec = _spec(node_id)
    return () if spec is None else spec[1]


def coverage_derive(model, node_id: str, texts: dict[str, str]) -> Execution | None:
    if _spec(node_id) is None:
        return None
    try:
        return Execution("derived", _value(model, node_id, texts))
    except _Bad as exc:
        return Execution("contradictory", reason=str(exc))


def _spec(node_id: str) -> tuple[str, tuple[str, ...]] | None:
    if node_id == "56":
        return ("level_airspeed", ("3", "8", "13"))
    if node_id == "58":
        return ("level_airspeed_unit", ("56",))
    if node_id == "85":
        return ("roll_difference", ("83", "84", "90"))
    if node_id in _AXIS_RULE:
        _role, frame_id = _AXIS_RULE[node_id]
        return ("axis_choice", (frame_id, *_READ_TRIADS))
    if node_id in _TURN_RULE:
        return ("turn_paired", _turn_inputs(node_id))
    if node_id in {"16", "17", "18"}:
        return ("path_order", ())
    if node_id in {"47", "48", "49"}:
        return ("product_order", ("16", "17", "18", "44", "45", "46"))
    if node_id in {"50", "51"}:
        return ("product_order", ("44", "45", "46"))
    if node_id in {"98", "99", "100", "133", "134"}:
        return ("product_order", ())
    if node_id in {"8", "95", "126"}:
        return ("axis_alignment", _alignment_inputs(node_id))
    if node_id == "13":
        return ("mount_matrix", ("2", "3", "4", "5", "6", "7", "12"))
    if node_id == "97":
        return ("mount_matrix", ("89", "90", "91", "92", "93", "94", "96"))
    if node_id == "132":
        return ("installation_matrix", ("89", "90", "91", "130", "131"))
    if node_id == "135":
        return ("balance_forward", ("123", "124", "125"))
    if node_id in {"142", "143", "144", "145"}:
        return ("deformed_axis", ("135", "136", "137", "123", "124", "125"))
    if node_id in {"146", "147", "148"}:
        return ("wind_component", ("126", "132", "133", "134", "145"))
    if node_id in {"62", "63", "64", "65", "66", "67"}:
        return ("expression_sign", (_component_of(node_id),))
    if node_id in {"161", "162", "163"}:
        return ("expression_sign", ({"161": "146", "162": "147", "163": "148"}[node_id],))
    if node_id in {"22", "30", "39", "158", "159"}:
        return ("sense_sign", _sense_inputs(node_id))
    if node_id in {"36", "38"}:
        return ("sense_sign", ("35",) if node_id == "36" else ("37",))
    if node_id in {"152", "153"}:
        return ("sense_sign", ("136", "151") if node_id == "152" else ("137", "151"))
    if node_id == "160":
        return ("sense_sign", ("154", "157"))
    if node_id == "79":
        return ("formula_part", ("75",))
    if node_id == "78":
        return ("formula_part", ("75",))
    if node_id in _DECLARED or node_id in {"179", "180"}:
        return ("declared_direction", ())
    if node_id in _FIXED:
        return ("fixed_term", ())
    if node_id in {"68", "74", "164", "169"}:
        return ("formula_part", {"68": ("22",), "74": ("30",), "164": ("158",), "169": ("159",)}[node_id])
    if node_id in {"70", "72", "76"}:
        return ("formula_part", {"70": ("71", "62", "63", "64"), "72": ("73", "62", "63", "64"), "76": ("77", "65", "66", "67")}[node_id])
    if node_id in {"165", "167", "170", "172"}:
        return ("formula_part", {"165": ("166", "161", "162", "163"), "167": ("168", "161", "162", "163"), "170": ("171", "161", "162", "163"), "172": ("173", "161", "162", "163")}[node_id])
    if node_id in {"71", "77"}:
        return ("formula_part", ("21", "5", "6", "7") if node_id == "71" else ("29", "5", "6", "7"))
    if node_id == "73":
        return ("formula_part", ("6",))
    if node_id in {"166", "171"}:
        return ("formula_part", ("149",) if node_id == "166" else ("150",))
    if node_id in {"168", "173"}:
        return ("formula_part", ("122",))
    if node_id in {"174", "175"}:
        start = 164 if node_id == "174" else 169
        return ("wind_ratio", tuple(str(start + offset) for offset in range(5)))
    if node_id in {"176", "177"}:
        return ("elastic_angle", ("174", "146", "147", "148") if node_id == "176" else ("175", "146", "147", "148"))
    if node_id == "178":
        return ("elastic_angle", ("160",))
    if node_id == "184":
        return ("reference_coordinate", ("183", "5", "6", "7"))
    if node_id in _MOMENT_KEY:
        return ("moment_sign", ("184", "185", "186", "187", "188", "5", "6", "7"))
    return None


def _alignment_inputs(node_id: str) -> tuple[str, ...]:
    if node_id == "8":
        return ("2", "3", "4", "5", "6", "7")
    if node_id == "95":
        return ("89", "90", "91", "92", "93", "94")
    return ("120", "121", "122", "123", "124", "125")


def _sense_inputs(node_id: str) -> tuple[str, ...]:
    return {"22": ("20", "21"), "30": ("28", "29"), "39": ("36", "38"), "158": ("152", "155"), "159": ("153", "156")}[node_id]


def _component_of(node_id: str) -> str:
    return {"62": "59", "63": "60", "64": "61", "65": "59", "66": "60", "67": "61"}[node_id]


def _value(model, node_id: str, texts: dict[str, str]) -> str:
    if node_id == "56":
        return _level_coordinate(texts)
    if node_id == "58":
        return _substitute_unit(texts)
    if node_id == "85":
        return _roll_difference(texts)
    if node_id in _AXIS_RULE:
        return _axis_choice(node_id, texts)
    if node_id in _TURN_RULE:
        return _turn_paired(model, node_id, texts)
    if node_id in {"16", "17", "18"}:
        return _path_order(model, node_id)
    if node_id in {"47", "48", "49", "50", "51", "98", "99", "100", "133", "134"}:
        return _product_order(node_id, texts)
    if node_id in {"8", "95", "126"}:
        return _show_matrix(_alignment(node_id, texts))
    if node_id in {"13", "97"}:
        return _show_matrix(_mount(node_id, texts))
    if node_id == "132":
        return _installation(model, texts)
    if node_id == "135":
        return _forward(texts)
    if node_id in {"142", "143", "144", "145"}:
        return _deformed(model, node_id, texts)
    if node_id in {"146", "147", "148"}:
        return _wind(texts, {"146": 0, "147": 1, "148": 2}[node_id])
    if node_id in {"62", "63", "64", "65", "66", "67"}:
        active = "alpha_M" if node_id in {"62", "63", "64"} else "beta_M"
        return _angle_sign(_parse(texts[_component_of(node_id)]), active)
    if node_id in {"161", "162", "163"}:
        source = {"161": "146", "162": "147", "163": "148"}[node_id]
        return _elastic_sign(_parse(texts[source]))
    if node_id in {"22", "30", "39", "158", "159"}:
        left, right = _sense_inputs(node_id)
        return _sense(texts[left], texts[right])
    if node_id in {"36", "38"}:
        return _wing(texts["35" if node_id == "36" else "37"])
    if node_id in {"152", "153"}:
        return _effect(texts["136" if node_id == "152" else "137"], texts["151"])
    if node_id == "160":
        return _agreement(texts["154"], texts["157"])
    if node_id == "79":
        return _beta_denominator(texts)
    if node_id == "78":
        return _beta_denominator_sign(texts)
    if node_id in _DECLARED:
        return _direction(model, _DECLARED[node_id])
    if node_id in {"179", "180"}:
        return _flow(model, ALPHA if node_id == "179" else BETA, ALPHA_V if node_id == "179" else BETA_V)
    if node_id in _FIXED:
        return _FIXED[node_id]
    if node_id in {"68", "74", "164", "169"}:
        return _outer(texts[{"68": "22", "74": "30", "164": "158", "169": "159"}[node_id]])
    if node_id in {"70", "72"}:
        return _lookup(texts, texts["71" if node_id == "70" else "73"], _SIGN_OF)
    if node_id == "76":
        return _lookup(texts, texts["77"], _BETA_SIGN)
    if node_id in {"165", "167", "170", "172"}:
        chosen = {"165": "166", "167": "168", "170": "171", "172": "173"}[node_id]
        return _lookup(texts, texts[chosen], _SIGN_OF)
    if node_id in {"71", "77"}:
        return _named_axis(texts, texts["21" if node_id == "71" else "29"], _BODY_ROLES, _BODY)
    if node_id == "73":
        return _named_axis(texts, "前面", _BODY_ROLES, _BODY)
    if node_id in {"166", "171"}:
        return _wind_name(texts["149" if node_id == "166" else "150"])
    if node_id in {"168", "173"}:
        return _named_axis(texts, "前面", _WIND_ROLES, _WIND)
    if node_id in {"174", "175"}:
        start = 164 if node_id == "174" else 169
        parts = [texts[str(start + offset)] for offset in range(5)]
        return _ratio(*parts)
    if node_id in {"176", "177"}:
        return _substituted(texts, "174" if node_id == "176" else "175")
    if node_id == "178":
        return _gamma(texts["160"])
    if node_id == "184":
        return _coordinate(texts, texts["183"])
    if node_id in _MOMENT_KEY:
        return _moment_sign(texts, _MOMENT_KEY[node_id])
    raise _Bad(f"ID{node_id} 没有计算器。")


def _directed_component(text: str, allowed: tuple[str, ...]) -> list[int]:
    raw = str(text).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    if name not in allowed:
        raise _Bad(f"“{raw}”不是该坐标系的坐标轴。")
    values = [0, 0, 0]
    values[allowed.index(name)] = -1 if negative else 1
    return values


def _level_coordinate(texts: dict[str, str]) -> str:
    """RP2*RP1 times the wind-frame unit vector of the stated axial direction."""
    wind = sp.Matrix(_directed_component(texts.get("3", ""), _WIND))
    mounted = sp.simplify(_square(texts.get("13", "")) * _square(texts.get("8", "")) * wind)
    return _show_vector(mounted)


def _substitute_unit(texts: dict[str, str]) -> str:
    column = sp.Matrix(_vector(texts.get("56", "")))
    norm = sp.simplify(column.norm())
    if norm == 0:
        raise _Bad("平置空速坐标是零向量。")
    if norm != 1:
        column = sp.simplify(column / norm)
    return _show_vector(column)


def _roll_difference(texts: dict[str, str]) -> str:
    """Balance axial pointing times (body hand·γ_M − installation hand·γ_tp).

    Reversing the body axial axis flips both the parallel test and the
    coordinate sense, so ID93 does not change this expression.
    """
    body = _hand(texts.get("84", ""))
    installation = _hand(texts.get("83", ""))
    pointing = next(value for value in _directed_component(texts.get("90", ""), _BALANCE) if value)
    theta = sp.simplify(pointing * (body * GAMMA_M_SYM - installation * GAMMA_TP_SYM))
    return _show_scalar(theta)


def _hand(text: str) -> int:
    value = str(text).strip()
    if value == "正":
        return 1
    if value == "负":
        return -1
    raise _Bad("正方向不是正或负。")


def _turn_inputs(node_id: str) -> tuple[str, ...]:
    if node_id == "87":
        return ("85", "86", "90")
    axis_node, direction_id, _kind = _TURN_RULE[node_id]
    _role, frame_id = _AXIS_RULE[axis_node]
    return (frame_id, *_READ_TRIADS, axis_node, direction_id)


def _binding(text: str, allowed: tuple[str, ...]) -> tuple[str, int]:
    raw = str(text).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    if name not in allowed:
        raise _Bad(f"“{raw}”不是该坐标系的坐标轴。")
    return name, -1 if negative else 1


def _triad(texts: dict[str, str], slots: tuple[str, ...], allowed: tuple[str, ...]) -> list[tuple[str, int]]:
    """Exactly one X, one Y and one Z, each a signed axis of this family."""
    parsed = []
    letters = []
    for slot in slots:
        name, sign = _binding(texts.get(slot, ""), allowed)
        letters.append(name[0])
        parsed.append((name, sign))
    if len(letters) != 3 or set(letters) != {"X", "Y", "Z"}:
        raise _Bad(f"三根坐标轴必须恰好是 X、Y、Z 各一根，现在是 {''.join(letters) or '空'}。")
    return parsed


def _family_of(frame: str) -> str:
    family = _FRAME_FAMILY.get(str(frame).strip())
    if family is None:
        raise _Bad(f"坐标系“{frame}”不在约定的族里。")
    return family


def _role_slots(frame: str) -> tuple[str, ...]:
    """The triad that belongs to the frame the blank actually named."""
    chosen = _FAMILY_SLOTS[_family_of(frame)]
    return tuple(chosen[role] for role in _SLOT_ORDER)


def _frame_matches(frame: str, axis: str) -> None:
    family = _FRAME_FAMILY.get(str(frame).strip())
    axis_family = _AXIS_FAMILY.get(axis)
    if family is None:
        raise _Bad(f"坐标系“{frame}”不在约定的族里。")
    if family != axis_family:
        raise _Bad(f"坐标系“{frame}”与轴“{axis}”不同族。")


def _declared_axis(texts: dict[str, str], role: str, frame: str) -> tuple[str, tuple[int, ...]]:
    """Axis name and absolute vector implied by this role in the frame's own triad."""
    slots = _role_slots(frame)
    name, sign = _triad(texts, slots, _ALLOWED[_family_of(frame)])[_SLOT_ORDER.index(role)]
    vector = tuple(sign * item for item in _VECTOR[_ABSOLUTE[role]])
    return name, vector


def _axis_choice(node_id: str, texts: dict[str, str]) -> str:
    role, frame_id = _AXIS_RULE[node_id]
    frame = str(texts.get(frame_id, "")).strip()
    name, _vector = _declared_axis(texts, role, frame)
    _frame_matches(frame, name)
    return name


def _rotation_word(sense: str) -> str:
    if sense == POSITIVE_ROTATION:
        return "正"
    if sense == NEGATIVE_ROTATION:
        return "负"
    raise _Bad("这根轴和这个方向定不出转向。")


def _wing_of_hand(hand: str) -> str:
    if hand not in {"相同", "相反"}:
        raise _Bad("右手定则关系不是相同或相反。")
    return _wing_about_nose(hand)


def _balance_turn(texts: dict[str, str]) -> str:
    name, _sign = _binding(texts.get("90", ""), _BALANCE)
    stated = str(texts.get("86", "")).strip()
    if stated != name:
        raise _Bad(f"转向只在轴“{name}”上成立，当前写的是“{stated}”。")
    expr = _parse(texts.get("85", ""))
    coeff = None
    for symbol in getattr(expr, "free_symbols", ()):
        if getattr(symbol, "name", "") == "gamma_M":
            coeff = sp.simplify(expr.coeff(symbol))
    if coeff == 1:
        return "正"
    if coeff == -1:
        return "逆"
    raise _Bad("滚转表达式里 γ_M 的系数不是 1 或 -1。")


def _turn_paired(model, node_id: str, texts: dict[str, str]) -> str:
    if node_id == "87":
        return _balance_turn(texts)
    axis_node, direction_id, kind = _TURN_RULE[node_id]
    role, frame_id = _AXIS_RULE[axis_node]
    frame = str(texts.get(frame_id, "")).strip()
    name, vector = _declared_axis(texts, role, frame)
    _frame_matches(frame, name)
    stated = str(texts.get(axis_node, "")).strip()
    if stated != name:
        raise _Bad(f"转向只在轴“{name}”上成立，当前写的是“{stated}”。")
    if kind == "gamma":
        sense = model._roll_sense(vector, _wing_of_hand(str(texts.get(direction_id, "")).strip()))
    else:
        direction = str(texts.get(direction_id, "")).strip()
        offset = _BODY_DIRECTION.get(direction)
        if offset is None:
            raise _Bad(f"方向“{direction}”不是上下左右前后。")
        sense = model._airspeed_rotation_sense(vector, offset)
    return _rotation_word(sense)


def _path_order(model, node_id: str) -> str:
    """The angle sequence itself is read from the declaration. No earlier blank states it."""
    names = {"alpha": "迎角", "beta": "侧滑角", "gamma": "滚转角"}
    first, second, third = (str(item) for item in model.euler_order)
    declared = {"16": names[first], "17": names[second], "18": names[third]}
    try:
        return declared[node_id]
    except KeyError as exc:
        raise _Bad("机构顺序不在迎角、侧滑、滚转的两种合法顺序里。") from exc


def attitude_convention(texts: dict[str, str]) -> tuple[str, bool]:
    """(主动/被动, 是否外旋). 动轴只配内旋，定轴只配外旋。"""
    sense = str(texts.get("44", "")).strip()
    kind = str(texts.get("45", "")).strip()
    spin = str(texts.get("46", "")).strip()
    if sense not in {"主动", "被动"}:
        raise _Bad("坐标系转换不是主动或被动。")
    paired = {("动轴", "内旋"): False, ("定轴", "外旋"): True}
    if (kind, spin) not in paired:
        raise _Bad("动轴必须配内旋，定轴必须配外旋。")
    return sense, paired[(kind, spin)]


def _product_order(node_id: str, texts: dict[str, str]) -> str:
    """R1–R3 follow the stated order, then intrinsic/extrinsic and active/passive.

    Passive intrinsic keeps the earliest angle on the right. Passive extrinsic
    reverses that order. Active uses the transpose of the passive product, which
    is the same names written backwards; the total matrix transposes each factor.
    R4 and R5 are the prepose names RP2 and RP1. They do not read the convention,
    so an illegal pair must not blank them.
    """
    fixed = {
        "50": "RP2", "51": "RP1",
        "98": "RTP_TZ2", "99": "RTP_TZ1", "100": "R_(γ_M-γ_tp)",
        "133": "RP_TP_FD", "134": "R_γ_tp",
    }
    if node_id in fixed:
        return fixed[node_id]
    sense, extrinsic = attitude_convention(texts)
    codes = {"迎角": "R_α_M", "侧滑角": "R_β_M", "滚转角": "R_γ_M"}
    ordered = [codes.get(str(texts.get(item, "")).strip()) for item in ("16", "17", "18")]
    if any(item is None for item in ordered) or len(set(ordered)) != 3:
        raise _Bad("变换顺序不是迎角、侧滑角、滚转角的一个排列。")
    first, second, third = ordered
    slots = (first, second, third) if extrinsic else (third, second, first)
    if sense == "主动":
        slots = tuple(reversed(slots))
    return {"47": slots[0], "48": slots[1], "49": slots[2]}[node_id]


def _alignment(node_id: str, texts: dict[str, str]):
    if node_id == "8":
        rows, cols = _assign(texts, (("5", "法向"), ("6", "轴向"), ("7", "侧向")), _BODY), _assign(texts, (("2", "法向"), ("3", "轴向"), ("4", "侧向")), _WIND)
        return _matrix(rows, cols, _BODY, _WIND)
    if node_id == "95":
        rows = _assign(texts, (("92", "法向"), ("93", "轴向"), ("94", "侧向")), _BODY)
        cols = _assign(texts, (("89", "法向"), ("90", "轴向"), ("91", "侧向")), _BALANCE)
        return _matrix(rows, cols, _BODY, _BALANCE)
    rows = _assign(texts, (("120", "法向"), ("121", "侧向"), ("122", "轴向")), _WIND)
    cols = _assign(texts, (("123", "法向"), ("124", "侧向"), ("125", "轴向")), _BALANCE)
    return _matrix(rows, cols, _WIND, _BALANCE)


def _mount(node_id: str, texts: dict[str, str]):
    if node_id == "13":
        relation = str(texts.get("12", "")).strip()
        inverted = {"相同": False, "相反": True}.get(relation)
        rows = _assign(texts, (("5", "法向"), ("6", "轴向"), ("7", "侧向")), _BODY)
        cols = _assign(texts, (("2", "法向"), ("3", "轴向"), ("4", "侧向")), _WIND)
        row_names, col_names = _BODY, _WIND
    else:
        relation = str(texts.get("96", "")).strip()
        inverted = {"正装": False, "倒装": True}.get(relation)
        rows = _assign(texts, (("92", "法向"), ("93", "轴向"), ("94", "侧向")), _BODY)
        cols = _assign(texts, (("89", "法向"), ("90", "轴向"), ("91", "侧向")), _BALANCE)
        row_names, col_names = _BODY, _BALANCE
    if inverted is None:
        raise _Bad("安装关系不是正装、倒装、相同或相反。")
    alignment = _matrix(rows, cols, row_names, col_names)
    if alignment.det() == 0:
        raise _Bad("坐标轴绑定不能组成可逆的对齐矩阵。")
    if not inverted:
        return sp.eye(3)
    placed = _matrix(rows, cols, row_names, col_names, invert_rows=True)
    return sp.simplify(placed * alignment.inv())


def _installation(model, texts: dict[str, str]) -> str:
    """Installation matrix in the balance triad the answers wrote, not the gold chirality."""
    del model
    axis = str(texts.get("130", "")).strip()
    turn = str(texts.get("131", "")).strip()
    if axis not in _BALANCE:
        raise _Bad(f"{axis} 不是天平坐标轴。")
    if turn == "正":
        theta = GAMMA_TP_SYM
    elif turn == "逆":
        theta = -GAMMA_TP_SYM
    else:
        raise _Bad("安装角转向不是正或逆。")
    try:
        left = recognized_left_handed(texts, "TP")
    except ValueError as exc:
        raise _Bad(str(exc)) from exc
    matrix = sp.simplify(_passive_rotation(axis, theta, left))
    return _show_matrix(matrix)


def _forward(texts: dict[str, str]) -> str:
    assigned = _assign(texts, (("123", "法向"), ("124", "侧向"), ("125", "轴向")), _BALANCE)
    values = []
    for name in _BALANCE:
        role, sign = assigned[name]
        values.append(sign if role == "轴向" else 0)
    return _show_vector(values)


def _deformed(model, node_id: str, texts: dict[str, str]) -> str:
    del model
    base = _vector(texts["135"])
    alpha = _shift(texts, texts["136"])
    beta = _shift(texts, texts["137"])
    symbols = _symbols()
    values = [
        base[index] + alpha[index] * symbols["alpha_e"] + beta[index] * symbols["beta_e"]
        for index in range(3)
    ]
    if node_id == "145":
        return _show_vector(values)
    index = {"142": 0, "143": 1, "144": 2}[node_id]
    sign = _small_sign(values[index], ("alpha_e", "beta_e"))
    if sign is None:
        raise _Bad("变形分量的符号无法确定。")
    return sign


def _wind(texts: dict[str, str], index: int) -> str:
    names = {"RP_TP_FD": "126", "R_γ_tp": "132"}
    factors = []
    seen = []
    for slot in ("133", "134"):
        name = str(texts.get(slot, "")).strip()
        matrix_id = names.get(name)
        if matrix_id is None:
            raise _Bad(f"连乘槽中的“{name}”不是 RP_TP_FD 或 R_γ_tp。")
        if name in seen:
            raise _Bad("连乘码重复，不是一个排列。")
        seen.append(name)
        factors.append(_square(texts[matrix_id]))
    if set(seen) != set(names):
        raise _Bad("连乘码不是 RP_TP_FD 与 R_γ_tp 的排列。")
    vector = _column(texts["145"])
    product = factors[0] * factors[1] * vector
    return _show_scalar(product[index])


def _sense(left: str, right: str) -> str:
    left, right = str(left).strip(), str(right).strip()
    if left not in _ZH_OPPOSITE or right not in _ZH_OPPOSITE:
        raise _Bad("方向不是上下、左右或前后。")
    if left == right:
        return "正"
    if _ZH_OPPOSITE[left] == right:
        return "负"
    return "不确定"


def _wing(relation: str) -> str:
    try:
        return _role(_wing_about_nose(str(relation).strip()))
    except Exception as exc:
        raise _Bad("滚转与右手定则的关系不是相同或相反。") from exc


def _effect(bend: str, mount: str) -> str:
    bend, mount = str(bend).strip(), str(mount).strip()
    if bend not in _ZH_OPPOSITE or mount not in {"正装", "反装"}:
        raise _Bad("弯曲方向或安装方式无法确定。")
    placed = bend if mount == "正装" else _ZH_OPPOSITE[bend] if bend in {"上方", "下方", "左侧", "右侧"} else bend
    return _ZH_OPPOSITE[placed]


def _agreement(relation: str, hand: str) -> str:
    pair = (str(relation).strip(), str(hand).strip())
    if pair in {("相同", "正"), ("相反", "负")}:
        return "正"
    if pair in {("相同", "负"), ("相反", "正")}:
        return "负"
    raise _Bad("弹性滚转和理论滚转的方向关系无法比较。")


def _direction(model, kind: str) -> str:
    if kind == "mechanism-alpha":
        toward = model.mechanism_angles[ALPHA_M].positive_airspeed_toward
    elif kind == "mechanism-beta":
        toward = model.mechanism_angles[BETA_M].positive_airspeed_toward
    elif kind == "theory-alpha":
        toward = model.angles[ALPHA].positive_airspeed_toward
    else:
        toward = model.angles[BETA].positive_airspeed_toward
    if not toward:
        raise _Bad("声明里没有这个方向。")
    return _zh(toward)


def _flow(model, theory_name: str, flow_name: str) -> str:
    theory = model.angles[theory_name].positive_airspeed_toward
    flow = model.flow_angles[flow_name].positive_airspeed_toward
    if not theory or not flow:
        raise _Bad("气流偏角或理论角没有声明方向。")
    return "正" if flow == theory else "负"


def _outer(text: str) -> str:
    sign = str(text).strip()
    if sign not in {"正", "负"}:
        raise _Bad("外层符号不是正或负。")
    return sign


def _beta_denominator(texts: dict[str, str]) -> str:
    """arcsin writes the ratio over 1. arctan would need another body component."""
    inverse = str(texts.get("75", "")).strip()
    if inverse == "arcsin":
        return "1"
    if inverse == "arctan":
        raise _Bad("侧滑写成 arctan 时，分母是体轴分量，不能沿用 arcsin 的常数 1。")
    raise _Bad("侧滑反三角函数不是 arcsin 或 arctan。")


def _beta_denominator_sign(texts: dict[str, str]) -> str:
    """The sign is asked before the component, so it follows the inverse function."""
    inverse = str(texts.get("75", "")).strip()
    if inverse == "arcsin":
        return "正"
    if inverse == "arctan":
        raise _Bad("侧滑写成 arctan 时，分母是体轴分量，符号不能沿用常数 1 的正号。")
    raise _Bad("侧滑反三角函数不是 arcsin 或 arctan。")


def _lookup(texts: dict[str, str], axis: str, table: dict[str, str]) -> str:
    node_id = table.get(str(axis).strip())
    if node_id is None:
        raise _Bad(f"分量“{axis}”没有对应的符号。")
    sign = str(texts.get(node_id, "")).strip()
    if sign not in {"正", "负"}:
        raise _Bad(f"ID{node_id} 的符号不是正或负。")
    return sign


def _named_axis(texts: dict[str, str], direction: str, roles: dict[str, str], allowed: tuple[str, ...]) -> str:
    role = _ROLE_OF.get(str(direction).strip())
    if role is None:
        raise _Bad(f"方向“{direction}”不能对应到法向、轴向或侧向。")
    raw = str(texts.get(roles[role], "")).strip()
    name = raw[1:] if raw.startswith("-") else raw
    if name not in allowed:
        raise _Bad(f"“{raw}”不是该坐标系的坐标轴。")
    return name


def _wind_name(text: str) -> str:
    name = str(text).strip()
    if name not in _WIND:
        raise _Bad(f"“{name}”不是风轴分量。")
    return name


def _ratio(outer: str, num_sign: str, num_axis: str, den_sign: str, den_axis: str) -> str:
    outer, num_sign, den_sign = (item.strip() for item in (outer, num_sign, den_sign))
    num_axis, den_axis = num_axis.strip(), den_axis.strip()
    if outer not in {"正", "负"} or num_sign not in {"正", "负"} or den_sign not in {"正", "负"}:
        raise _Bad("分式的符号不是正或负。")
    if num_axis not in _WIND or den_axis not in _WIND:
        raise _Bad("分子或分母不是风轴分量。")
    body = f"{_factor(num_sign, num_axis)}/{_factor(den_sign, den_axis)}"
    return f"-{body}" if outer == "负" else body


def _factor(sign: str, axis: str) -> str:
    return f"({axis})" if sign == "正" else f"(-{axis})"


def _substituted(texts: dict[str, str], formula_id: str) -> str:
    expr = _parse(texts[formula_id])
    pieces = {name: _parse(texts[node]) for name, node in (("XFD", "146"), ("YFD", "147"), ("ZFD", "148"))}
    replaced = expr.xreplace({
        symbol: pieces[symbol.name]
        for symbol in getattr(expr, "free_symbols", ())
        if symbol.name in pieces
    })
    return _show_scalar(replaced)


def _gamma(sign: str) -> str:
    sign = str(sign).strip()
    if sign == "正":
        return "γ_e"
    if sign == "负":
        return "-γ_e"
    raise _Bad("弹性滚转增量的符号不是正或负。")


def _coordinate(texts: dict[str, str], slot: str) -> str:
    axis = _SLOT.get(str(slot).strip())
    if axis is None:
        raise _Bad("不同的工程分量不是 Xg、Yg 或 Zg。")
    _vectors, roles, signs = _body_geometry(texts)
    if roles.get(axis) != "axial":
        raise _Bad(f"{slot} 对应的体轴不在绝对轴向上。")
    forward = signs[axis]
    cells = []
    for name in _BODY:
        engineering = _ENGINEERING[name]
        if name != axis:
            cells.append(engineering)
            continue
        arm = f"Xr - {engineering}"
        cells.append(arm if forward == 1 else f"-({arm})")
    return "[" + ", ".join(cells) + "]"


def _moment_sign(texts: dict[str, str], key: str) -> str:
    for node_id, expected in _PATH:
        if str(texts.get(node_id, "")).strip() != expected:
            raise _Bad("力矩等效的起点、终点或力的位置与模板不一致。")
    coords = [_adopt(item) for item in _vector(texts["184"])]
    expr = moment_term(texts, coords, key)
    numer, denom = sp.fraction(sp.together(sp.simplify(expr)))
    if sp.simplify(denom - L_RT_SYM) != 0:
        raise _Bad("力矩项没有除以 L_RT。")
    force = next((name for name, symbol in FORCE_SYMBOL.items() if symbol in numer.free_symbols), None)
    if force is None:
        raise _Bad("力矩项里没有 CA、CZ 或 CN。")
    arm = _adopt(sp.expand(sp.simplify(numer / FORCE_SYMBOL[force])))
    leading = sp.simplify(sp.Poly(sp.expand(arm), _XR, XG_SYM, YG_SYM, ZG_SYM).LC())
    if leading == 1:
        return "正"
    if leading == -1:
        return "负"
    raise _Bad("力矩符号的主系数不是 1 或 -1。")


_ROLE_EN = {"法向": "normal", "轴向": "axial", "侧向": "lateral"}


def _body_geometry(texts: dict[str, str]):
    """Coordinate vectors implied by the current body-axis bindings, not the sample frame."""
    assigned = _assign(texts, (("5", "法向"), ("6", "轴向"), ("7", "侧向")), _BODY)
    vectors = {}
    roles = {}
    signs = {}
    for name, (role, sign) in assigned.items():
        base = _ABS[role]
        vectors[name] = tuple(sign * item for item in base)
        roles[name] = _ROLE_EN[role]
        signs[name] = sign
    return vectors, roles, signs


def _embed_vectors(coords, vectors: dict[str, tuple[int, ...]]):
    total = [sp.Integer(0), sp.Integer(0), sp.Integer(0)]
    for coord, name in zip(coords, _BODY):
        vector = vectors[name]
        for index in range(3):
            total[index] += coord * vector[index]
    return total


def moment_term(texts: dict[str, str], coords, key: str):
    """Moment term from the stated reference coordinates and the stated body triad."""
    vectors, roles, _signs = _body_geometry(texts)
    found = None
    for force_role, force_symbol in FORCE_OF_ROLE.items():
        force_coords = [FORCE_SYMBOL[force_symbol] if roles[name] == force_role else 0 for name in _BODY]
        moment = _cross_expr(_embed_vectors(force_coords, vectors), _embed_vectors(coords, vectors))
        for name in _BODY:
            if f"M{force_symbol}_{MOMENT_CODE[roles[name]]}" == key:
                found = sp.simplify(_dot_expr(moment, vectors[name]) / L_RT_SYM)
    if found is None:
        raise _Bad(f"没有矩分量 {key}。")
    if sp.simplify(found) == 0:
        raise _Bad("力臂长度为零。")
    return found


def _assign(texts: dict[str, str], pairs: tuple[tuple[str, str], ...], names: tuple[str, ...]) -> dict[str, tuple[str, int]]:
    found: dict[str, tuple[str, int]] = {}
    for node_id, role in pairs:
        raw = str(texts.get(node_id, "")).strip()
        sign = -1 if raw.startswith("-") else 1
        name = raw[1:] if raw.startswith("-") else raw
        if name not in names or name in found:
            raise _Bad(f"ID{node_id} 的坐标轴“{raw}”不能和其他轴组成一组。")
        found[name] = (role, sign)
    if set(found) != set(names):
        raise _Bad("三根坐标轴没有各对应一个绝对方向。")
    return found


def _matrix(rows, cols, row_names, col_names, invert_rows: bool = False):
    data = []
    for row_name in row_names:
        vector = _abs_vector(*rows[row_name])
        if invert_rows:
            vector = (vector[0], -vector[1], -vector[2])
        data.append([_dot(vector, _abs_vector(*cols[col_name])) for col_name in col_names])
    return sp.Matrix(data)


def _abs_vector(role: str, sign: int) -> tuple[int, int, int]:
    base = _ABS[role]
    return (sign * base[0], sign * base[1], sign * base[2])


def _shift(texts: dict[str, str], direction: str) -> list[int]:
    sense = {
        "上方": ("法向", 1), "下方": ("法向", -1),
        "右侧": ("侧向", 1), "左侧": ("侧向", -1),
        "前面": ("轴向", 1), "后面": ("轴向", -1),
    }.get(str(direction).strip())
    if sense is None:
        raise _Bad(f"弯曲方向“{direction}”不是上下、左右或前后。")
    role, sign = sense
    assigned = _assign(texts, (("123", "法向"), ("124", "侧向"), ("125", "轴向")), _BALANCE)
    values = []
    for name in _BALANCE:
        bound, pointing = assigned[name]
        values.append(sign * pointing if bound == role else 0)
    return values


def _angle_sign(expr, active: str) -> str:
    values = {"alpha_M": 0, "beta_M": 0, "gamma_M": 0, active: sp.Rational(1, 10)}
    number = sp.simplify(_replace_named(expr, values))
    if number == 0:
        return "不确定"
    if getattr(number, "is_number", False) and number.is_real:
        if number > 0:
            return "正"
        if number < 0:
            return "负"
    raise _Bad("分量在单角下的符号无法确定。")


def _elastic_sign(expr) -> str:
    reduced = _replace_named(expr, {"gamma_tp": 0})
    sign = _small_sign(reduced, ("alpha_e", "beta_e"))
    if sign is None:
        raise _Bad("风轴分量在安装角为零时的符号无法确定。")
    return sign


def _numeric_sign(value) -> int | None:
    """1、0、-1 when the expression is a real number. None when the sign is open."""
    number = sp.simplify(value)
    if not (getattr(number, "is_number", False) and bool(number.is_real)):
        return None
    if number == 0:
        return 0
    return 1 if number > 0 else -1


def _small_sign(expr, names: tuple[str, ...]) -> str | None:
    expanded = sp.expand(expr)
    constant = expanded
    usable = []
    for name in names:
        coefficient = _coefficient(expanded, name)
        constant = sp.simplify(_replace_named(constant, {name: 0}))
        decision = _numeric_sign(coefficient)
        if decision is None:
            return None
        if decision == 0:
            continue
        usable.append(decision)
    constant_sign = _numeric_sign(constant)
    if constant_sign is None:
        return None
    if constant_sign != 0:
        return "正" if constant_sign > 0 else "负"
    if len(set(usable)) != 1:
        return "不确定"
    return "正" if usable[0] > 0 else "负"


def _coefficient(expr, name: str):
    for symbol in getattr(expr, "free_symbols", ()):
        if symbol.name == name:
            return sp.simplify(expr.coeff(symbol))
    return 0


def _replace_named(expr, values: dict):
    replacement = {}
    for symbol in getattr(expr, "free_symbols", ()):
        if symbol.name in values:
            replacement[symbol] = values[symbol.name]
    return expr.xreplace(replacement)


def _vector(text: str) -> list:
    parsed = _parse(text)
    sympy, _locals = _sympy_modules()
    if not isinstance(parsed, sympy.MatrixBase) or parsed.shape not in {(3, 1), (1, 3)}:
        raise _Bad(f"“{text}”不是三维向量。")
    return list(parsed)


def _column(text: str):
    parsed = _vector(text)
    return sp.Matrix(parsed)


def _square(text: str):
    parsed = _parse(text)
    if getattr(parsed, "shape", None) != (3, 3):
        raise _Bad("矩阵不是 3×3。")
    return parsed


def _parse(text: str):
    sympy, locals_map = _sympy_modules()
    try:
        return _parse_expr(str(text), sympy, locals_map)
    except Exception as exc:
        raise _Bad(f"无法解析“{text}”。") from exc


def _symbols() -> dict[str, sp.Expr]:
    _sympy, locals_map = _sympy_modules()
    return locals_map


def _adopt(expr):
    return expr.xreplace({
        symbol: _CANON[symbol.name]
        for symbol in getattr(expr, "free_symbols", ())
        if symbol.name in _CANON
    })


def _zh(direction: str) -> str:
    return {
        "forward": "前面", "rearward": "后面",
        "up": "上方", "down": "下方",
        "left": "左侧", "right": "右侧",
    }[direction.split("_", 1)[1]]


def _show_matrix(matrix) -> str:
    return _matrix_text(sp.Matrix(matrix)).replace("gamma_tp", "γ_tp").replace("alpha_e", "α_e").replace("beta_e", "β_e")


def _show_scalar(expr) -> str:
    rendered = str(sp.simplify(expr))
    for source, target in (
        ("alpha_e", "α_e"), ("beta_e", "β_e"), ("gamma_e", "γ_e"), ("gamma_tp", "γ_tp"),
        ("alpha_M", "α_M"), ("beta_M", "β_M"), ("gamma_M", "γ_M"),
    ):
        rendered = rendered.replace(source, target)
    return rendered


def _show_vector(values) -> str:
    return "[" + ", ".join(_show_scalar(value) for value in values) + "]"
