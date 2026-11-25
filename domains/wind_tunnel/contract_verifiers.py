# -*- coding: utf-8 -*-
"""Deterministic verifiers for composed targets.

Each function rebuilds one answer from the texts the model actually stated.
A gold-contract check must succeed before the diagnosis treats the result as evidence.
"""
from __future__ import annotations

import itertools

import sympy as sp

from cloze_score import sympy_equal
from cloze_score import _parse_expr, _sympy_modules
from domains.wind_tunnel.contract_executor import Execution
from domains.wind_tunnel.ref_point_generate import (
    ARM_OPTIONS,
    DENOM_OPTIONS,
    FORCE_OPTIONS,
    FORCE_SYMBOL,
    L_RT_SYM,
    MOMENT_KEYS,
    XG_SYM,
    YG_SYM,
    ZG_SYM,
)
from domains.wind_tunnel.contract_coverage import (
    attitude_convention,
    coverage_derive,
    coverage_inputs,
    coverage_validator,
    moment_term,
)
from domains.wind_tunnel.tz_fd_generate import DISPLAY, _matrix_text, _text

VERSION = "v1"
_NAME_OF_MATRIX = {
    "RP1": "8",
    "RP2": "13",
    "R_α_M": "26",
    "R_β_M": "34",
    "R_γ_M": "43",
    "R_(γ_M-γ_tp)": "88",
    "RTP_TZ1": "95",
    "RTP_TZ2": "97",
}
_ATTITUDE_NAMES = ("RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M")
_LOAD_NAMES = ("R_(γ_M-γ_tp)", "RTP_TZ1", "RTP_TZ2")
_ATTITUDE_POOL = set(_ATTITUDE_NAMES)
_LOAD_POOL = set(_LOAD_NAMES)
_COMPONENT = {"XTZ": "59", "YTZ": "60", "ZTZ": "61"}
_ANGLE = {
    "80": ("68", "69", "70", "71", "72", "73"),
    "81": ("74", "75", "76", "77", "78", "79"),
}
_MOMENT_FORCE = {str(201 + index): key for index, key in enumerate(MOMENT_KEYS)}
_MOMENT_ARM = {str(207 + index): key for index, key in enumerate(MOMENT_KEYS)}
_MOMENT_DENOM = "213"
_MOMENT = {
    str(195 + index): (str(189 + index), str(201 + index), str(207 + index), _MOMENT_DENOM, key)
    for index, key in enumerate(MOMENT_KEYS)
}
_AIRSPEED = {"59": 0, "60": 1, "61": 2}
_LOAD = {
    "111": "CN", "112": "CA", "113": "CZ",
    "114": "MN", "115": "MA", "116": "MZ",
}
_BODY_BIND = (("105", 0), ("106", 1), ("107", 2))
_BALANCE_BIND = (("108", 0), ("109", 1), ("110", 2))
_BODY_PAIR = {
    "CN_TZ,MN_TZ": ("CN", "MN"),
    "CA_TZ,MA_TZ": ("CA", "MA"),
    "CZ_TZ,MZ_TZ": ("CZ", "MZ"),
}
_BALANCE_PAIR = {
    "CN_TP,MN_TP": ("CN_TP", "MN_TP"),
    "CA_TP,MA_TP": ("CA_TP", "MA_TP"),
    "CZ_TP,MZ_TP": ("CZ_TP", "MZ_TP"),
}
_XR = sp.symbols("Xr", real=True)
_CANON = {
    "Xg": XG_SYM, "Yg": YG_SYM, "Zg": ZG_SYM, "Xr": _XR, "L_RT": L_RT_SYM,
    **FORCE_SYMBOL,
}


def validator_of(node_id: str) -> str | None:
    if node_id in {"55", "104"}:
        return "total_matrix_product"
    if node_id in _ANGLE:
        return "angle_formula"
    if node_id in _MOMENT_FORCE or node_id in _MOMENT_ARM or node_id == _MOMENT_DENOM:
        return "formula_part"
    if node_id in _MOMENT:
        return "moment_coefficient"
    if node_id in _AIRSPEED:
        return "airspeed_component"
    if node_id in _LOAD:
        return "load_component"
    return coverage_validator(node_id)


def input_ids(node_id: str) -> tuple[str, ...]:
    """Texts this verifier reads. Counterfactuals only swap these nodes."""
    if node_id == "55":
        return ("44", "45", "46", "47", "48", "49") + tuple(
            _NAME_OF_MATRIX[name] for name in _ATTITUDE_NAMES
        )
    if node_id == "104":
        return ("98", "99", "100") + tuple(_NAME_OF_MATRIX[name] for name in _LOAD_NAMES)
    if node_id in _ANGLE:
        return _ANGLE[node_id] + tuple(_COMPONENT.values())
    if node_id in _MOMENT_FORCE:
        return ()
    if node_id in _MOMENT_ARM:
        return ("184", "5", "6", "7")
    if node_id == _MOMENT_DENOM:
        return ("182",)
    if node_id in _MOMENT:
        sign, force, arm, denom, _key = _MOMENT[node_id]
        return (sign, force, arm, denom)
    if node_id in _AIRSPEED:
        return ("55", "58")
    if node_id in _LOAD:
        return ("104", "105", "106", "107", "108", "109", "110")
    covered = coverage_inputs(node_id)
    return covered if covered else ()


def derive(model, node_id: str, texts: dict[str, str]) -> Execution | None:
    kind = validator_of(node_id)
    if kind == "total_matrix_product":
        slots = ("47", "48", "49") if node_id == "55" else ("98", "99", "100")
        return _product(texts, slots)
    if kind == "angle_formula":
        return _angle(texts, _ANGLE[node_id])
    if node_id in _MOMENT_FORCE:
        return Execution("derived", _MOMENT_FORCE[node_id].split("_", 1)[0][1:])
    if node_id in _MOMENT_ARM:
        return _moment_arm(texts, _MOMENT_ARM[node_id])
    if node_id == _MOMENT_DENOM:
        return _moment_denom(texts)
    if kind == "moment_coefficient":
        sign, force, arm, denom, _key = _MOMENT[node_id]
        return _compose_moment(texts, sign, force, arm, denom)
    if kind == "airspeed_component":
        return _airspeed(texts, _AIRSPEED[node_id])
    if kind == "load_component":
        return _load(texts, _LOAD[node_id])
    return coverage_derive(model, node_id, texts)


def _agrees(derived: str, gold_text: str) -> bool:
    if str(derived).strip() == str(gold_text).strip():
        return True
    return sympy_equal(derived, gold_text)


def _require_change(model, gold: dict[str, str], node_id: str, edits: dict[str, str], reason: str) -> None:
    edited = dict(gold)
    edited.update(edits)
    derived = derive(model, node_id, edited)
    if derived is not None and derived.ok and _agrees(derived.matrix, gold[node_id]):
        raise RuntimeError(reason)


def check_verifiers(model, gold: dict[str, str]) -> None:
    """Refuse to diagnose with a verifier that cannot reproduce this sample's gold."""
    from domains.wind_tunnel.contract_labels import TARGETS

    missing = [
        node_id for node_id, (label, _subtype) in TARGETS.items()
        if label.startswith("A-") and validator_of(node_id) is None and node_id not in {"26", "34", "43", "88"}
    ]
    if missing:
        raise RuntimeError("这些 A 类空还没有计算器：" + "、".join(f"ID{item}" for item in sorted(missing, key=int)))
    for node_id in sorted((item for item in TARGETS if validator_of(item)), key=int):
        derived = derive(model, node_id, gold)
        if derived is None or not derived.ok or not _agrees(derived.matrix, gold[node_id]):
            reason = "" if derived is None else derived.reason or derived.matrix
            raise RuntimeError(f"ID{node_id} 的验证器没有由金标契约推出金标答案：{reason}")
    _require_change(model, gold, "80", {"68": "负" if gold["68"] == "正" else "正"}, "改写 ID68 的符号后，ID80 的公式没有变化。")
    _require_change(model, gold, "55", {"47": "RP1" if gold["47"] != "RP1" else "RP2"}, "改写 ID47 的矩阵名后，ID55 的总矩阵没有变化。")
    _require_change(
        model, gold, "59",
        {"58": "[0, 0, 1]" if gold["58"].replace(" ", "") == "[1,0,0]" else "[1, 0, 0]"},
        "改写 ID58 的空速向量后，ID59 没有变化。",
    )
    swapped = dict(gold)
    swapped["105"], swapped["106"] = gold["106"], gold["105"]
    _require_change(model, gold, "111", {"105": swapped["105"], "106": swapped["106"]}, "对调 ID105 与 ID106 后，ID111 没有变化。")
    _require_change(model, gold, "13", {"12": "相反" if gold["12"] == "相同" else "相同"}, "改写 ID12 后，ID13 没有变化。")
    _require_change(model, gold, "8", {"2": gold["4"], "4": gold["2"]}, "对调 ID2 与 ID4 后，ID8 没有变化。")
    _require_change(model, gold, "36", {"35": "相反" if gold["35"] == "相同" else "相同"}, "改写 ID35 后，ID36 没有变化。")
    if gold["62"] in {"正", "负"}:
        component = gold["59"].strip()
        negated = component[1:] if component.startswith("-") else f"-({component})"
        _require_change(model, gold, "62", {"59": negated}, "改写 ID59 的符号后，ID62 没有变化。")
    _require_change(model, gold, "178", {"160": "负" if gold["160"] == "正" else "正"}, "改写 ID160 后，ID178 没有变化。")
    _require_change(model, gold, "184", {"183": "Zg" if gold["183"] == "Xg" else "Xg"}, "改写 ID183 后，ID184 没有变化。")
    other_force = "CA" if gold["202"] != "CA" else "CN"
    other_arm = "Zg" if gold["208"] != "Zg" else "Yg"
    edited = dict(gold)
    edited["202"] = other_force
    composed = derive(model, "196", edited)
    expected = f"{'-' if gold['190'] == '负' else ''}{other_force}·{gold['208']}/L_RT"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"改写 ID202 后，ID196 应为 {expected}。")
    edited = dict(gold)
    edited["208"] = other_arm
    composed = derive(model, "196", edited)
    expected = f"{'-' if gold['190'] == '负' else ''}{gold['202']}·{other_arm}/L_RT"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"改写 ID208 后，ID196 应为 {expected}。")
    composed = derive(model, "196", {**gold, "213": "L"})
    expected = f"{'-' if gold['190'] == '负' else ''}{gold['202']}·{gold['208']}/L"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"改写 ID213 后，ID196 应为 {expected}。")
    _require_change(
        model, gold, "56",
        {"3": "YFD" if gold["3"].lstrip("-") != "YFD" else "ZFD"},
        "改写 ID3 的风轴轴向后，ID56 没有变化。",
    )
    _require_change(
        model, gold, "58",
        {"56": "[0, 1, 0]" if gold["56"].replace(" ", "") == "[1,0,0]" else "[1, 0, 0]"},
        "改写 ID56 后，ID58 没有变化。",
    )
    _require_change(
        model, gold, "85",
        {"84": "负" if gold["84"] == "正" else "正"},
        "改写 ID84 后，ID85 没有变化。",
    )
    balance_axis = str(gold["90"]).strip()
    _require_change(
        model, gold, "85",
        {"90": balance_axis[1:] if balance_axis.startswith("-") else f"-{balance_axis}"},
        "改写 ID90 的指向后，ID85 没有变化。",
    )
    if input_ids("16") != () or input_ids("47") != ("16", "17", "18", "44", "45", "46"):
        raise RuntimeError("变换顺序和 R1 的前置关系不对。")
    names = {"alpha": "迎角", "beta": "侧滑角", "gamma": "滚转角"}
    order = [str(item) for item in model.euler_order]
    for node_id, angle in zip(("16", "17", "18"), order):
        derived = derive(model, node_id, {})
        if derived is None or not derived.ok or derived.matrix != names[angle]:
            raise RuntimeError(f"ID{node_id} 没有按定义声明推出变换顺序。")
    _require_change(
        model, gold, "47",
        {"16": gold["18"], "18": gold["16"]},
        "对调变换顺序的首尾后，ID47 没有变化。",
    )
    _require_change(
        model, gold, "49",
        {"16": gold["18"], "18": gold["16"]},
        "对调变换顺序的首尾后，ID49 没有变化。",
    )
    extrinsic = derive(model, "47", {**gold, "46": "外旋"})
    if extrinsic is not None and extrinsic.ok:
        raise RuntimeError("动轴配外旋时仍推出了 R1。")
    _check_transform_rules(model, gold)
    _check_axis_rules(model, gold)
    _check_composed_rules(model, gold)


def _unsigned(text: str) -> str:
    raw = str(text).strip()
    return raw[1:] if raw.startswith("-") else raw


def _flip_sign(text: str) -> str:
    raw = str(text).strip()
    return raw[1:] if raw.startswith("-") else f"-{raw}"


def _must_answer(model, texts: dict[str, str], node_id: str) -> str:
    derived = derive(model, node_id, texts)
    if derived is None or not derived.ok:
        reason = "" if derived is None else derived.reason
        raise RuntimeError(f"ID{node_id} 应给出答案，实际为矛盾：{reason}")
    return str(derived.matrix).strip()


def _must_reject(model, texts: dict[str, str], node_id: str, what: str) -> None:
    derived = derive(model, node_id, texts)
    if derived is None or derived.ok or not derived.reason:
        raise RuntimeError(what)


def _check_axis_rules(model, gold: dict[str, str]) -> None:
    """Batch 1: binding lookup, same-family frame, turn only on the declared axis."""
    cycled = {**gold, "5": gold["6"], "6": gold["7"], "7": gold["5"]}
    expected = {"24": _unsigned(gold["5"]), "32": _unsigned(gold["6"]), "41": _unsigned(gold["7"])}
    for node_id, want in expected.items():
        got = _must_answer(model, cycled, node_id)
        if got != want:
            raise RuntimeError(f"轴循环后 ID{node_id} 应为 {want}，实际是 {got}。")
        if got == str(gold[node_id]).strip():
            raise RuntimeError(f"轴循环后 ID{node_id} 仍是金标。")
    _must_reject(model, {**gold, "5": gold["6"]}, "24", "重复轴没有判成矛盾。")
    wind_axis = _must_answer(model, {**gold, "23": "风轴系"}, "24")
    if wind_axis != _unsigned(gold["4"]):
        raise RuntimeError(f"风轴系下迎角轴应读风轴侧向 {_unsigned(gold['4'])}，实际是 {wind_axis}。")
    if wind_axis == str(gold["24"]).strip():
        raise RuntimeError("风轴系下迎角轴仍是金标。")
    balance_axis = _must_answer(model, {**gold, "31": "天平系"}, "32")
    if balance_axis != _unsigned(gold["89"]):
        raise RuntimeError(f"天平系下侧滑轴应读天平法向 {_unsigned(gold['89'])}，实际是 {balance_axis}。")
    _must_reject(model, {**gold, "24": "XFD"}, "25", "中间体轴系上写风轴没有判成矛盾。")
    opposite = {"上方": "下方", "下方": "上方", "左侧": "右侧", "右侧": "左侧", "前面": "后面", "后面": "前面"}
    for node_id, binding, direction in (("25", "7", "20"), ("33", "5", "28")):
        flipped = {**gold, binding: _flip_sign(gold[binding])}
        got = _must_answer(model, flipped, node_id)
        if got == str(gold[node_id]).strip() or got not in {"正", "负"}:
            raise RuntimeError(f"对调 ID{binding} 的指向后，ID{node_id} 没有变成另一个合法转向。")
        turned = {**gold, direction: opposite[str(gold[direction]).strip()]}
        got = _must_answer(model, turned, node_id)
        if got == str(gold[node_id]).strip() or got not in {"正", "负"}:
            raise RuntimeError(f"对调 ID{direction} 后，ID{node_id} 没有变成另一个合法转向。")
    hand = "相反" if str(gold["35"]).strip() == "相同" else "相同"
    got = _must_answer(model, {**gold, "35": hand}, "42")
    if got == str(gold["42"]).strip() or got not in {"正", "负"}:
        raise RuntimeError("对调右手定则后，ID42 没有变成另一个合法转向。")
    other = next(name for name in ("XTZ", "YTZ", "ZTZ") if name != str(gold["24"]).strip())
    _must_reject(model, {**gold, "24": other}, "25", "换轴后转向没有判成矛盾。")
    expr = str(gold["85"]).strip()
    negated = expr[1:].strip() if expr.startswith("-") else f"-({expr})"
    got = _must_answer(model, {**gold, "85": negated}, "87")
    if got == str(gold["87"]).strip() or got not in {"正", "逆"}:
        raise RuntimeError("对调滚转表达式的符号后，ID87 没有变成另一个合法转向。")
    other_balance = next(name for name in ("XTP", "YTP", "ZYP") if name != str(gold["86"]).strip())
    _must_reject(model, {**gold, "86": other_balance}, "87", "换到另一根天平轴后转向没有判成矛盾。")
    role_slot = {"24": "7", "32": "5", "41": "6"}
    turn_of = {"25": "24", "33": "32", "42": "41"}
    for perm in itertools.permutations(("XTZ", "YTZ", "ZTZ")):
        for signs in itertools.product(("", "-"), repeat=3):
            named = {"5": signs[0] + perm[0], "6": signs[1] + perm[1], "7": signs[2] + perm[2]}
            texts = {**gold, **named}
            chosen = {}
            for node_id, slot in role_slot.items():
                got = _must_answer(model, texts, node_id)
                want = _unsigned(named[slot])
                if got != want:
                    raise RuntimeError(f"绑定 {named[slot]} 后 ID{node_id} 应为 {want}，实际是 {got}。")
                chosen[node_id] = got
            paired = {**texts, "24": chosen["24"], "32": chosen["32"], "41": chosen["41"]}
            for turn_id, axis_id in turn_of.items():
                got = _must_answer(model, paired, turn_id)
                if got not in {"正", "负"}:
                    raise RuntimeError(f"合法绑定下 ID{turn_id} 的转向不是正或负：{got}。")
                slot = role_slot[axis_id]
                if named[slot] == str(gold[slot]).strip():
                    continue
                axis_differs = chosen[axis_id] != str(gold[axis_id]).strip()
                turn_differs = got != str(gold[turn_id]).strip()
                if not axis_differs and not turn_differs:
                    raise RuntimeError(
                        f"ID{slot} 从 {gold[slot]} 换成 {named[slot]} 后，ID{axis_id} 与 ID{turn_id} 仍是金标。"
                    )


def _check_composed_rules(model, gold: dict[str, str]) -> None:
    """Batches 2–7: matrices, products, components, and moment arms follow the stated texts."""
    from domains.wind_tunnel.contract_executor import attitude_target, roll_target
    from domains.wind_tunnel.tz_fd_generate import ALPHA_SYM

    other = "YTZ" if str(gold["24"]).strip() != "YTZ" else "XTZ"
    changed = attitude_target(model, ALPHA_SYM, gold["23"], other, gold["25"], gold)
    if not changed.ok or _agrees(changed.matrix, gold["26"]):
        raise RuntimeError("换一根机体轴后，ID26 没有变成另一个合法矩阵。")
    clash = attitude_target(model, ALPHA_SYM, gold["23"], "XFD", gold["25"], gold)
    if clash.ok or not clash.reason:
        raise RuntimeError("轴与坐标系不同族时，ID26 没有判成矛盾。")
    cycled = {**gold, "89": gold["90"], "90": gold["91"], "91": gold["89"]}
    new_axis = _unsigned(cycled["90"])
    derived = None
    for turn in (gold["87"], "逆" if str(gold["87"]).strip() == "正" else "正"):
        derived = roll_target(
            model, gold["83"], gold["84"], new_axis, turn,
            cycled["89"], cycled["90"], cycled["91"],
        )
        if derived.ok:
            break
    if derived is None or not derived.ok or _agrees(derived.matrix, gold["88"]):
        reason = "" if derived is None else derived.reason or derived.matrix
        raise RuntimeError(f"天平轴循环后，ID88 没有变成另一个合法矩阵：{reason}")
    foreign = roll_target(
        model, gold["83"], gold["84"], "XTZ", gold["87"], gold["89"], gold["90"], gold["91"],
    )
    if foreign.ok or not foreign.reason:
        raise RuntimeError("滚转轴不是天平轴时，ID88 没有判成矛盾。")
    flipped_hand = "负" if str(gold["84"]).strip() == "正" else "正"
    opposed = roll_target(
        model, gold["83"], flipped_hand, gold["86"], gold["87"], gold["89"], gold["90"], gold["91"],
    )
    if opposed.ok:
        raise RuntimeError("手性与转向不一致时，ID88 没有判成矛盾。")
    flipped_axis = {**gold, "6": _flip_sign(gold["6"])}
    got = _must_answer(model, flipped_axis, "184")
    if _agrees(got, gold["184"]):
        raise RuntimeError("对调轴向指向后，ID184 没有变化。")
    moment = derive(model, "189", flipped_axis)
    if moment is None or not moment.ok or str(moment.matrix).strip() == str(gold["189"]).strip():
        raise RuntimeError("对调轴向指向后，ID189 没有变成另一个合法符号。")
    _must_reject(model, {**gold, "48": gold["47"]}, "55", "连乘码重复没有判成矛盾。")
    _must_reject(model, {**gold, "26": "[1, 0, 0]"}, "55", "上游不是 3×3 时总矩阵没有判成矛盾。")
    tilted = {**gold, "126": "[[0, 1, 0], [0, 0, 1], [1, 0, 0]]"}
    base = derive(model, "146", tilted)
    swapped = {**tilted, "133": gold["134"], "134": gold["133"]}
    wind = derive(model, "146", swapped)
    if base is None or wind is None or not base.ok or not wind.ok or _agrees(wind.matrix, base.matrix):
        reason = "" if wind is None else wind.reason
        raise RuntimeError(f"对调连乘码后，ID146 没有变成另一个合法分量：{reason}")
    _must_reject(model, {**gold, "134": gold["133"]}, "146", "连乘码重复没有判成矛盾。")
    direction = str(gold["136"]).strip()
    slot = {"上方": "123", "下方": "123", "左侧": "124", "右侧": "124", "前面": "125", "后面": "125"}.get(direction)
    if slot is None:
        raise RuntimeError(f"ID136 的弯曲方向无法对应到天平轴：{direction}")
    bent = {**gold, slot: _flip_sign(gold[slot])}
    deformed = derive(model, "145", bent)
    if deformed is None or not deformed.ok or _agrees(deformed.matrix, gold["145"]):
        reason = "" if deformed is None else deformed.reason
        raise RuntimeError(f"对调弯曲方向所在轴的指向后，ID145 没有变化：{reason}")
    component = _COMPONENT.get(str(gold["77"]).strip())
    if str(gold["75"]).strip() == "arcsin" and component:
        _must_reject(model, {**gold, component: "2"}, "81", "反正弦超出定义域没有判成矛盾。")


def _airspeed(texts: dict[str, str], index: int) -> Execution:
    try:
        matrix = _parse(texts["55"])
        vector = _column(_parse(texts["58"]))
    except Exception:
        return Execution("contradictory", reason="总矩阵或空速向量无法解析。")
    if getattr(matrix, "shape", None) != (3, 3) or vector is None:
        return Execution("contradictory", reason="R_t 不是 3×3，或空速不是三维向量。")
    return Execution("derived", _scalar_text((matrix * vector)[index]))


def _load(texts: dict[str, str], symbol: str) -> Execution:
    try:
        matrix = _parse(texts["104"])
    except Exception:
        return Execution("contradictory", reason="载荷总矩阵无法解析。")
    if getattr(matrix, "shape", None) != (3, 3):
        return Execution("contradictory", reason="载荷总矩阵不是 3×3。")
    force = symbol in {"CN", "CA", "CZ"}
    try:
        row = _bound_row(texts, symbol, _BODY_BIND, _BODY_PAIR)
        vector = _bound_vector(texts, force)
    except RuntimeError as exc:
        return Execution("contradictory", reason=str(exc))
    return Execution("derived", _scalar_text((matrix * vector)[row]))


def _bound_row(texts: dict[str, str], symbol: str, slots: tuple[tuple[str, int], ...], pairs: dict[str, tuple[str, str]]) -> int:
    found = []
    for node_id, index in slots:
        pair = pairs.get(str(texts.get(node_id, "")).strip())
        if pair is None:
            raise RuntimeError(f"ID{node_id} 的分量对应不是六分量配对。")
        if symbol in pair:
            found.append(index)
    if len(found) != 1:
        raise RuntimeError(f"{symbol} 没有唯一的体轴行。")
    return found[0]


def _bound_vector(texts: dict[str, str], force: bool):
    values = [None, None, None]
    for node_id, index in _BALANCE_BIND:
        pair = _BALANCE_PAIR.get(str(texts.get(node_id, "")).strip())
        if pair is None:
            raise RuntimeError(f"ID{node_id} 的天平分量对应不是六分量配对。")
        values[index] = sp.Symbol(pair[0] if force else pair[1])
    if any(item is None for item in values) or len(set(values)) != 3:
        raise RuntimeError("天平六分量没有各自落到一根坐标轴上。")
    return sp.Matrix(values)


def _column(parsed):
    sympy, _locals = _sympy_modules()
    if not isinstance(parsed, sympy.MatrixBase):
        return None
    if parsed.shape == (3, 1):
        return parsed
    if parsed.shape == (1, 3):
        return parsed.reshape(3, 1)
    return None


def _scalar_text(expr) -> str:
    rendered = _text(sp.simplify(expr))
    return rendered.replace("gamma_tp", "γ_tp").replace("gamma_M", "γ_M")


def _check_transform_rules(model, gold: dict[str, str]) -> None:
    """Active intrinsic is the transpose of the gold product. Passive extrinsic reverses the order."""
    active_texts = {**gold, "44": "主动", "45": "动轴", "46": "内旋"}
    extrinsic_texts = {**gold, "44": "被动", "45": "定轴", "46": "外旋"}
    active_codes = {slot: _must_answer(model, active_texts, slot) for slot in ("47", "48", "49")}
    extrinsic_codes = {slot: _must_answer(model, extrinsic_texts, slot) for slot in ("47", "48", "49")}
    if active_codes == {slot: str(gold[slot]).strip() for slot in active_codes}:
        raise RuntimeError("主动内旋没有改写 R1、R2、R3。")
    if extrinsic_codes == {slot: str(gold[slot]).strip() for slot in extrinsic_codes}:
        raise RuntimeError("被动外旋没有改写 R1、R2、R3。")
    if active_codes != extrinsic_codes:
        raise RuntimeError("主动内旋与被动外旋的连乘码顺序不一致。")
    active_total = _must_answer(model, {**active_texts, **active_codes}, "55")
    extrinsic_total = _must_answer(model, {**extrinsic_texts, **extrinsic_codes}, "55")
    if _agrees(active_total, gold["55"]) or _agrees(extrinsic_total, gold["55"]) or _agrees(active_total, extrinsic_total):
        raise RuntimeError("主动内旋与被动外旋没有给出两张不同的非金标总矩阵。")
    transposed = _matrix_text(sp.simplify(_parse(gold["55"]).T))
    if not _agrees(active_total, transposed):
        raise RuntimeError("主动内旋的总矩阵不是金标总矩阵的转置。")


def _product(texts: dict[str, str], slots: tuple[str, ...]) -> Execution:
    pool = _ATTITUDE_POOL if slots == ("47", "48", "49") else _LOAD_POOL
    active = False
    if slots == ("47", "48", "49"):
        try:
            sense, _extrinsic = attitude_convention(texts)
        except RuntimeError as exc:
            return Execution("contradictory", reason=str(exc))
        active = sense == "主动"
    factors = []
    seen = []
    for slot in slots:
        name = str(texts.get(slot, "")).strip()
        if name not in pool:
            return Execution("contradictory", reason=f"R 槽中的“{name}”不在这组连乘码里。")
        if name in seen:
            return Execution("contradictory", reason="连乘码重复，不是一个排列。")
        seen.append(name)
        matrix_id = _NAME_OF_MATRIX[name]
        try:
            factor = _parse(texts[matrix_id])
        except Exception:
            return Execution("contradictory", reason=f"ID{matrix_id} 的矩阵无法解析。")
        if getattr(factor, "shape", None) != (3, 3):
            return Execution("contradictory", reason=f"ID{matrix_id} 的矩阵不是 3×3。")
        factors.append(sp.transpose(factor) if active else factor)
    if slots == ("98", "99", "100") and set(seen) != _LOAD_POOL:
        return Execution("contradictory", reason="连乘码不是天平系三个矩阵的一个排列。")
    product = factors[0]
    for factor in factors[1:]:
        product = sp.simplify(product * factor)
    if getattr(product, "shape", None) != (3, 3):
        return Execution("contradictory", reason="连乘结果不是 3×3 矩阵。")
    return Execution("derived", _matrix_text(product))


def _angle(texts: dict[str, str], parts: tuple[str, ...]) -> Execution:
    outer, inverse, num_sign, num_name, den_sign, den_name = (str(texts.get(item, "")).strip() for item in parts)
    if inverse not in {"arctan", "arcsin"} or outer not in {"正", "负"}:
        return Execution("contradictory", reason="反三角函数或外层符号无法确定。")
    try:
        numerator = _signed(num_sign, _component(texts, num_name))
        denominator = _signed(den_sign, _component(texts, den_name))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    if numerator is None or denominator is None:
        return Execution("contradictory", reason="分子或分母的符号不是正或负。")
    sympy, _locals = _sympy_modules()
    if _is_zero(denominator):
        return Execution("contradictory", reason="分母化简后为零。")
    interior = numerator if den_name == "1" and den_sign == "正" else numerator / denominator
    if inverse == "arcsin":
        value = sp.simplify(interior)
        if getattr(value, "is_number", False) and getattr(value, "is_real", False) and abs(value) > 1:
            return Execution("contradictory", reason="反正弦的自变量超出定义域。")
    body = sympy.atan(interior) if inverse == "arctan" else sympy.asin(interior)
    if outer == "负":
        body = -body
    return Execution("derived", _function_text(body))


def _moment_arm(texts: dict[str, str], key: str) -> Execution:
    try:
        coords = _vector(texts["184"])
        formula = _apply_sign("正", moment_term(texts, coords, key))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    arm = _canonical_arm(formula.split("·", 1)[1].rsplit("/", 1)[0])
    if arm not in ARM_OPTIONS:
        return Execution("contradictory", reason=f"坐标分量 {arm} 不在选项里。")
    return Execution("derived", arm)


def _moment_denom(texts: dict[str, str]) -> Execution:
    raw = str(texts.get("182", "")).replace(" ", "")
    if "L" not in raw:
        return Execution("contradictory", reason="ID182 没有给出由 L、L_t、L_r 组成的长度。")
    return Execution("derived", "L_RT")


def _compose_moment(texts: dict[str, str], sign_id: str, force_id: str, arm_id: str, denom_id: str) -> Execution:
    sign = str(texts.get(sign_id, "")).strip()
    force = str(texts.get(force_id, "")).strip()
    arm = str(texts.get(arm_id, "")).strip()
    denom = str(texts.get(denom_id, "")).strip()
    if sign not in {"正", "负"}:
        return Execution("contradictory", reason=f"ID{sign_id} 的符号不是正或负。")
    if force not in FORCE_OPTIONS:
        return Execution("contradictory", reason=f"ID{force_id} 的力系数不在 CN、CA、CZ 里。")
    if arm not in ARM_OPTIONS:
        return Execution("contradictory", reason=f"ID{arm_id} 的坐标分量不在选项里。")
    if denom not in DENOM_OPTIONS:
        return Execution("contradictory", reason=f"ID{denom_id} 的分母不在 L、L_t、L_r、L_RT 里。")
    outer = "-" if sign == "负" else ""
    return Execution("derived", f"{outer}{force}·{arm}/{denom}")


def _canonical_arm(arm_text: str) -> str:
    """Map a sympy rendering such as (-Xg + Xr) back onto the choice text."""
    if arm_text in ARM_OPTIONS:
        return arm_text
    symbols = {"Xr": _XR, "Xg": XG_SYM, "Yg": YG_SYM, "Zg": ZG_SYM}
    try:
        expr = sp.sympify(arm_text, locals=symbols)
    except Exception:
        return arm_text
    for option in ARM_OPTIONS:
        if sp.simplify(expr - sp.sympify(option, locals=symbols)) == 0:
            return option
    return arm_text


def _apply_sign(sign: str, expr) -> str:
    numer, denom = sp.fraction(sp.together(sp.simplify(expr)))
    if sp.simplify(denom - L_RT_SYM) != 0:
        raise RuntimeError("力矩项没有除以 L_RT。")
    force = next((name for name, symbol in FORCE_SYMBOL.items() if symbol in numer.free_symbols), None)
    if force is None:
        raise RuntimeError("力矩项里没有 CA、CZ 或 CN。")
    arm = sp.expand(sp.simplify(numer / FORCE_SYMBOL[force]))
    positive = _positive_arm(arm)
    arm_text = str(positive)
    if any(operator in arm_text for operator in ("+", "-")):
        arm_text = f"({arm_text})"
    outer = "-" if sign == "负" else ""
    return f"{outer}{force}·{arm_text}/L_RT"


def _positive_arm(arm):
    if sp.simplify(arm) == 0:
        raise RuntimeError("力臂长度为零。")
    polynomial = sp.Poly(sp.expand(arm), _XR, XG_SYM, YG_SYM, ZG_SYM)
    leading = sp.simplify(polynomial.LC())
    if leading not in (1, -1):
        raise RuntimeError(f"力臂 {arm} 的主系数不是 1 或 -1。")
    if leading == -1:
        return sp.expand(-arm)
    return sp.expand(arm)


def _adopt(expr):
    return expr.xreplace({
        symbol: _CANON[symbol.name]
        for symbol in getattr(expr, "free_symbols", ())
        if symbol.name in _CANON
    })


def _vector(text: str) -> list:
    parsed = _parse(text)
    sympy, _locals = _sympy_modules()
    if not isinstance(parsed, sympy.MatrixBase):
        raise RuntimeError("不是向量。")
    values = [_adopt(item) for item in list(parsed)]
    if len(values) != 3:
        raise RuntimeError("参考点坐标不是三个分量。")
    return values


def _component(texts: dict[str, str], name: str):
    if name == "1":
        return sp.Integer(1)
    node_id = _COMPONENT.get(name)
    if node_id is None:
        raise RuntimeError(f"分量“{name}”不是 XTZ、YTZ、ZTZ 或 1。")
    return _parse(texts[node_id])


def _signed(sign: str, expr):
    if sign == "正":
        return expr
    if sign == "负":
        return -expr
    return None


def _parse(text: str):
    sympy, locals_map = _sympy_modules()
    return _parse_expr(text, sympy, locals_map)


def _is_zero(expr) -> bool:
    return sp.simplify(expr) == 0


def _function_text(expr) -> str:
    rendered = str(sp.simplify(expr))
    rendered = rendered.replace("atan", "arctan").replace("asin", "arcsin")
    for source, target in DISPLAY.items():
        rendered = rendered.replace(source, target)
    rendered = rendered.replace("gamma_tp", "γ_tp").replace("gamma_M", "γ_M")
    return rendered
