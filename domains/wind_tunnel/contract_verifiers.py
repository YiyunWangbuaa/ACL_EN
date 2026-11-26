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
        raise RuntimeError("These A blanks still have no calculator: " + ", ".join(f"ID{item}" for item in sorted(missing, key=int)))
    for node_id in sorted((item for item in TARGETS if validator_of(item)), key=int):
        derived = derive(model, node_id, gold)
        if derived is None or not derived.ok or not _agrees(derived.matrix, gold[node_id]):
            reason = "" if derived is None else derived.reason or derived.matrix
            raise RuntimeError(f"The verifier of ID{node_id} did not derive the gold answer from the gold contract: {reason}")
    _require_change(model, gold, "80", {"68": "negative" if gold["68"] == "positive" else "positive"}, "After rewriting the sign of ID68, the formula of ID80 did not change.")
    _require_change(model, gold, "55", {"47": "RP1" if gold["47"] != "RP1" else "RP2"}, "After rewriting the matrix name of ID47, the total matrix of ID55 did not change.")
    _require_change(
        model, gold, "59",
        {"58": "[0, 0, 1]" if gold["58"].replace(" ", "") == "[1,0,0]" else "[1, 0, 0]"},
        "After rewriting the airspeed vector of ID58, ID59 did not change.",
    )
    swapped = dict(gold)
    swapped["105"], swapped["106"] = gold["106"], gold["105"]
    _require_change(model, gold, "111", {"105": swapped["105"], "106": swapped["106"]}, "After swapping ID105 and ID106, ID111 did not change.")
    _require_change(model, gold, "13", {"12": "opposite" if gold["12"] == "same" else "same"}, "After rewriting ID12, ID13 did not change.")
    _require_change(model, gold, "8", {"2": gold["4"], "4": gold["2"]}, "After swapping ID2 and ID4, ID8 did not change.")
    _require_change(model, gold, "36", {"35": "opposite" if gold["35"] == "same" else "same"}, "After rewriting ID35, ID36 did not change.")
    if gold["62"] in {"positive", "negative"}:
        component = gold["59"].strip()
        negated = component[1:] if component.startswith("-") else f"-({component})"
        _require_change(model, gold, "62", {"59": negated}, "After rewriting the sign of ID59, ID62 did not change.")
    _require_change(model, gold, "178", {"160": "negative" if gold["160"] == "positive" else "positive"}, "After rewriting ID160, ID178 did not change.")
    _require_change(model, gold, "184", {"183": "Zg" if gold["183"] == "Xg" else "Xg"}, "After rewriting ID183, ID184 did not change.")
    other_force = "CA" if gold["202"] != "CA" else "CN"
    other_arm = "Zg" if gold["208"] != "Zg" else "Yg"
    edited = dict(gold)
    edited["202"] = other_force
    composed = derive(model, "196", edited)
    expected = f"{'-' if gold['190'] == 'negative' else ''}{other_force}·{gold['208']}/L_RT"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"After rewriting ID202, ID196 should be {expected}.")
    edited = dict(gold)
    edited["208"] = other_arm
    composed = derive(model, "196", edited)
    expected = f"{'-' if gold['190'] == 'negative' else ''}{gold['202']}·{other_arm}/L_RT"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"After rewriting ID208, ID196 should be {expected}.")
    composed = derive(model, "196", {**gold, "213": "L"})
    expected = f"{'-' if gold['190'] == 'negative' else ''}{gold['202']}·{gold['208']}/L"
    if composed is None or not composed.ok or composed.matrix != expected:
        raise RuntimeError(f"After rewriting ID213, ID196 should be {expected}.")
    _require_change(
        model, gold, "56",
        {"3": "YFD" if gold["3"].lstrip("-") != "YFD" else "ZFD"},
        "After rewriting the wind-frame axial axis of ID3, ID56 did not change.",
    )
    _require_change(
        model, gold, "58",
        {"56": "[0, 1, 0]" if gold["56"].replace(" ", "") == "[1,0,0]" else "[1, 0, 0]"},
        "After rewriting ID56, ID58 did not change.",
    )
    _require_change(
        model, gold, "85",
        {"84": "negative" if gold["84"] == "positive" else "positive"},
        "After rewriting ID84, ID85 did not change.",
    )
    balance_axis = str(gold["90"]).strip()
    _require_change(
        model, gold, "85",
        {"90": balance_axis[1:] if balance_axis.startswith("-") else f"-{balance_axis}"},
        "After rewriting the pointing of ID90, ID85 did not change.",
    )
    if input_ids("16") != () or input_ids("47") != ("16", "17", "18", "44", "45", "46"):
        raise RuntimeError("The transform order and the leading place of R1 do not agree.")
    names = {"alpha": "angle of attack", "beta": "sideslip", "gamma": "roll"}
    order = [str(item) for item in model.euler_order]
    for node_id, angle in zip(("16", "17", "18"), order):
        derived = derive(model, node_id, {})
        if derived is None or not derived.ok or derived.matrix != names[angle]:
            raise RuntimeError(f"ID{node_id} did not derive the transform order from the declaration.")
    _require_change(
        model, gold, "47",
        {"16": gold["18"], "18": gold["16"]},
        "After swapping the first and last of the transform order, ID47 did not change.",
    )
    _require_change(
        model, gold, "49",
        {"16": gold["18"], "18": gold["16"]},
        "After swapping the first and last of the transform order, ID49 did not change.",
    )
    extrinsic = derive(model, "47", {**gold, "46": "extrinsic"})
    if extrinsic is not None and extrinsic.ok:
        raise RuntimeError("R1 was still derived for moving axes with an extrinsic rotation.")
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
        raise RuntimeError(f"ID{node_id} should give an answer, but it is contradictory: {reason}")
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
            raise RuntimeError(f"After cycling the axes, ID{node_id} should be {want}, but it is {got}.")
        if got == str(gold[node_id]).strip():
            raise RuntimeError(f"After cycling the axes, ID{node_id} is still the gold.")
    _must_reject(model, {**gold, "5": gold["6"]}, "24", "A repeated axis was not judged contradictory.")
    wind_axis = _must_answer(model, {**gold, "23": "wind frame"}, "24")
    if wind_axis != _unsigned(gold["4"]):
        raise RuntimeError(f"Under the wind frame the angle-of-attack axis should read the wind-frame lateral {_unsigned(gold['4'])}, but it is {wind_axis}.")
    if wind_axis == str(gold["24"]).strip():
        raise RuntimeError("Under the wind frame the angle-of-attack axis is still the gold.")
    balance_axis = _must_answer(model, {**gold, "31": "balance frame"}, "32")
    if balance_axis != _unsigned(gold["89"]):
        raise RuntimeError(f"Under the balance frame the sideslip axis should read the balance normal {_unsigned(gold['89'])}, but it is {balance_axis}.")
    _must_reject(model, {**gold, "24": "XFD"}, "25", "A wind-frame axis written on the intermediate body-axis frame was not judged contradictory.")
    opposite = {"upward": "downward", "downward": "upward", "left": "right", "right": "left", "front": "rear", "rear": "front"}
    for node_id, binding, direction in (("25", "7", "20"), ("33", "5", "28")):
        flipped = {**gold, binding: _flip_sign(gold[binding])}
        got = _must_answer(model, flipped, node_id)
        if got == str(gold[node_id]).strip() or got not in {"positive", "negative"}:
            raise RuntimeError(f"After swapping the pointing of ID{binding}, ID{node_id} did not become another legal sense.")
        turned = {**gold, direction: opposite[str(gold[direction]).strip()]}
        got = _must_answer(model, turned, node_id)
        if got == str(gold[node_id]).strip() or got not in {"positive", "negative"}:
            raise RuntimeError(f"After swapping ID{direction}, ID{node_id} did not become another legal sense.")
    hand = "opposite" if str(gold["35"]).strip() == "same" else "same"
    got = _must_answer(model, {**gold, "35": hand}, "42")
    if got == str(gold["42"]).strip() or got not in {"positive", "negative"}:
        raise RuntimeError("After swapping the right-hand rule, ID42 did not become another legal sense.")
    other = next(name for name in ("XTZ", "YTZ", "ZTZ") if name != str(gold["24"]).strip())
    _must_reject(model, {**gold, "24": other}, "25", "The sense was not judged contradictory after the axis was changed.")
    expr = str(gold["85"]).strip()
    negated = expr[1:].strip() if expr.startswith("-") else f"-({expr})"
    got = _must_answer(model, {**gold, "85": negated}, "87")
    if got == str(gold["87"]).strip() or got not in {"positive", "reverse"}:
        raise RuntimeError("After swapping the sign of the roll expression, ID87 did not become another legal sense.")
    other_balance = next(name for name in ("XTP", "YTP", "ZYP") if name != str(gold["86"]).strip())
    _must_reject(model, {**gold, "86": other_balance}, "87", "The sense was not judged contradictory after moving to another balance axis.")
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
                    raise RuntimeError(f"After binding {named[slot]}, ID{node_id} should be {want}, but it is {got}.")
                chosen[node_id] = got
            paired = {**texts, "24": chosen["24"], "32": chosen["32"], "41": chosen["41"]}
            for turn_id, axis_id in turn_of.items():
                got = _must_answer(model, paired, turn_id)
                if got not in {"positive", "negative"}:
                    raise RuntimeError(f"Under a legal binding the sense of ID{turn_id} is not positive or negative: {got}.")
                slot = role_slot[axis_id]
                if named[slot] == str(gold[slot]).strip():
                    continue
                axis_differs = chosen[axis_id] != str(gold[axis_id]).strip()
                turn_differs = got != str(gold[turn_id]).strip()
                if not axis_differs and not turn_differs:
                    raise RuntimeError(
                        f"After changing ID{slot} from {gold[slot]} to {named[slot]}, ID{axis_id} and ID{turn_id} are still the gold."
                    )


def _check_composed_rules(model, gold: dict[str, str]) -> None:
    """Batches 2–7: matrices, products, components, and moment arms follow the stated texts."""
    from domains.wind_tunnel.contract_executor import attitude_target, roll_target
    from domains.wind_tunnel.tz_fd_generate import ALPHA_SYM

    other = "YTZ" if str(gold["24"]).strip() != "YTZ" else "XTZ"
    changed = attitude_target(model, ALPHA_SYM, gold["23"], other, gold["25"], gold)
    if not changed.ok or _agrees(changed.matrix, gold["26"]):
        raise RuntimeError("After changing to another body axis, ID26 did not become another legal matrix.")
    clash = attitude_target(model, ALPHA_SYM, gold["23"], "XFD", gold["25"], gold)
    if clash.ok or not clash.reason:
        raise RuntimeError("ID26 was not judged contradictory when the axis and the frame are different families.")
    cycled = {**gold, "89": gold["90"], "90": gold["91"], "91": gold["89"]}
    new_axis = _unsigned(cycled["90"])
    derived = None
    for turn in (gold["87"], "reverse" if str(gold["87"]).strip() == "positive" else "positive"):
        derived = roll_target(
            model, gold["83"], gold["84"], new_axis, turn,
            cycled["89"], cycled["90"], cycled["91"],
        )
        if derived.ok:
            break
    if derived is None or not derived.ok or _agrees(derived.matrix, gold["88"]):
        reason = "" if derived is None else derived.reason or derived.matrix
        raise RuntimeError(f"After cycling the balance axes, ID88 did not become another legal matrix: {reason}")
    foreign = roll_target(
        model, gold["83"], gold["84"], "XTZ", gold["87"], gold["89"], gold["90"], gold["91"],
    )
    if foreign.ok or not foreign.reason:
        raise RuntimeError("ID88 was not judged contradictory when the roll axis is not a balance axis.")
    flipped_hand = "negative" if str(gold["84"]).strip() == "positive" else "positive"
    opposed = roll_target(
        model, gold["83"], flipped_hand, gold["86"], gold["87"], gold["89"], gold["90"], gold["91"],
    )
    if opposed.ok:
        raise RuntimeError("ID88 was not judged contradictory when the chirality and the sense disagree.")
    flipped_axis = {**gold, "6": _flip_sign(gold["6"])}
    got = _must_answer(model, flipped_axis, "184")
    if _agrees(got, gold["184"]):
        raise RuntimeError("After swapping the axial pointing, ID184 did not change.")
    moment = derive(model, "189", flipped_axis)
    if moment is None or not moment.ok or str(moment.matrix).strip() == str(gold["189"]).strip():
        raise RuntimeError("After swapping the axial pointing, ID189 did not become another legal sign.")
    _must_reject(model, {**gold, "48": gold["47"]}, "55", "A repeated product code was not judged contradictory.")
    _must_reject(model, {**gold, "26": "[1, 0, 0]"}, "55", "When an upstream matrix is not 3×3, the total matrix was not judged contradictory.")
    tilted = {**gold, "126": "[[0, 1, 0], [0, 0, 1], [1, 0, 0]]"}
    base = derive(model, "146", tilted)
    swapped = {**tilted, "133": gold["134"], "134": gold["133"]}
    wind = derive(model, "146", swapped)
    if base is None or wind is None or not base.ok or not wind.ok or _agrees(wind.matrix, base.matrix):
        reason = "" if wind is None else wind.reason
        raise RuntimeError(f"After swapping a product code, ID146 did not become another legal component: {reason}")
    _must_reject(model, {**gold, "134": gold["133"]}, "146", "A repeated product code was not judged contradictory.")
    direction = str(gold["136"]).strip()
    slot = {"upward": "123", "downward": "123", "left": "124", "right": "124", "front": "125", "rear": "125"}.get(direction)
    if slot is None:
        raise RuntimeError(f"The bend direction of ID136 does not correspond to a balance axis: {direction}")
    bent = {**gold, slot: _flip_sign(gold[slot])}
    deformed = derive(model, "145", bent)
    if deformed is None or not deformed.ok or _agrees(deformed.matrix, gold["145"]):
        reason = "" if deformed is None else deformed.reason
        raise RuntimeError(f"After swapping the pointing of the axis of the bend direction, ID145 did not change: {reason}")
    component = _COMPONENT.get(str(gold["77"]).strip())
    if str(gold["75"]).strip() == "arcsin" and component:
        _must_reject(model, {**gold, component: "2"}, "81", "An arcsin outside its domain was not judged contradictory.")


def _airspeed(texts: dict[str, str], index: int) -> Execution:
    try:
        matrix = _parse(texts["55"])
        vector = _column(_parse(texts["58"]))
    except Exception:
        return Execution("contradictory", reason="The total matrix or the airspeed vector cannot parse.")
    if getattr(matrix, "shape", None) != (3, 3) or vector is None:
        return Execution("contradictory", reason="R_t is not 3×3, or the airspeed is not three-dimensional.")
    return Execution("derived", _scalar_text((matrix * vector)[index]))


def _load(texts: dict[str, str], symbol: str) -> Execution:
    try:
        matrix = _parse(texts["104"])
    except Exception:
        return Execution("contradictory", reason="The total load matrix cannot parse.")
    if getattr(matrix, "shape", None) != (3, 3):
        return Execution("contradictory", reason="The total load matrix is not 3×3.")
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
            raise RuntimeError(f"The component correspondence of ID{node_id} is not a six-component pairing.")
        if symbol in pair:
            found.append(index)
    if len(found) != 1:
        raise RuntimeError(f"{symbol} does not have a unique body-axis row.")
    return found[0]


def _bound_vector(texts: dict[str, str], force: bool):
    values = [None, None, None]
    for node_id, index in _BALANCE_BIND:
        pair = _BALANCE_PAIR.get(str(texts.get(node_id, "")).strip())
        if pair is None:
            raise RuntimeError(f"The balance-component correspondence of ID{node_id} is not a six-component pairing.")
        values[index] = sp.Symbol(pair[0] if force else pair[1])
    if any(item is None for item in values) or len(set(values)) != 3:
        raise RuntimeError("The six balance components do not each land on one coordinate axis.")
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
    active_texts = {**gold, "44": "active", "45": "moving axes", "46": "intrinsic"}
    extrinsic_texts = {**gold, "44": "passive", "45": "fixed axes", "46": "extrinsic"}
    active_codes = {slot: _must_answer(model, active_texts, slot) for slot in ("47", "48", "49")}
    extrinsic_codes = {slot: _must_answer(model, extrinsic_texts, slot) for slot in ("47", "48", "49")}
    if active_codes == {slot: str(gold[slot]).strip() for slot in active_codes}:
        raise RuntimeError("Active intrinsic rotation did not rewrite R1, R2, and R3.")
    if extrinsic_codes == {slot: str(gold[slot]).strip() for slot in extrinsic_codes}:
        raise RuntimeError("Passive extrinsic rotation did not rewrite R1, R2, and R3.")
    if active_codes != extrinsic_codes:
        raise RuntimeError("The product-code order of active intrinsic rotation and passive extrinsic rotation does not agree.")
    active_total = _must_answer(model, {**active_texts, **active_codes}, "55")
    extrinsic_total = _must_answer(model, {**extrinsic_texts, **extrinsic_codes}, "55")
    if _agrees(active_total, gold["55"]) or _agrees(extrinsic_total, gold["55"]) or _agrees(active_total, extrinsic_total):
        raise RuntimeError("Active intrinsic rotation and passive extrinsic rotation did not give two different non-gold total matrices.")
    transposed = _matrix_text(sp.simplify(_parse(gold["55"]).T))
    if not _agrees(active_total, transposed):
        raise RuntimeError("The total matrix of active intrinsic rotation is not the transpose of the gold total matrix.")


def _product(texts: dict[str, str], slots: tuple[str, ...]) -> Execution:
    pool = _ATTITUDE_POOL if slots == ("47", "48", "49") else _LOAD_POOL
    active = False
    if slots == ("47", "48", "49"):
        try:
            sense, _extrinsic = attitude_convention(texts)
        except RuntimeError as exc:
            return Execution("contradictory", reason=str(exc))
        active = sense == "active"
    factors = []
    seen = []
    for slot in slots:
        name = str(texts.get(slot, "")).strip()
        if name not in pool:
            return Execution("contradictory", reason=f"The name {name!r} in an R slot is not in this set of product codes.")
        if name in seen:
            return Execution("contradictory", reason="A product code is repeated, so this is not a permutation.")
        seen.append(name)
        matrix_id = _NAME_OF_MATRIX[name]
        try:
            factor = _parse(texts[matrix_id])
        except Exception:
            return Execution("contradictory", reason=f"The matrix of ID{matrix_id} cannot parse.")
        if getattr(factor, "shape", None) != (3, 3):
            return Execution("contradictory", reason=f"The matrix of ID{matrix_id} is not 3×3.")
        factors.append(sp.transpose(factor) if active else factor)
    if slots == ("98", "99", "100") and set(seen) != _LOAD_POOL:
        return Execution("contradictory", reason="The product codes are not a permutation of the three balance-frame matrices.")
    product = factors[0]
    for factor in factors[1:]:
        product = sp.simplify(product * factor)
    if getattr(product, "shape", None) != (3, 3):
        return Execution("contradictory", reason="The product is not 3×3.")
    return Execution("derived", _matrix_text(product))


def _angle(texts: dict[str, str], parts: tuple[str, ...]) -> Execution:
    outer, inverse, num_sign, num_name, den_sign, den_name = (str(texts.get(item, "")).strip() for item in parts)
    if inverse not in {"arctan", "arcsin"} or outer not in {"positive", "negative"}:
        return Execution("contradictory", reason="The inverse trigonometric function or the outer sign cannot be determined.")
    try:
        numerator = _signed(num_sign, _component(texts, num_name))
        denominator = _signed(den_sign, _component(texts, den_name))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    if numerator is None or denominator is None:
        return Execution("contradictory", reason="The numerator sign or the denominator sign is not positive or negative.")
    sympy, _locals = _sympy_modules()
    if _is_zero(denominator):
        return Execution("contradictory", reason="The denominator simplifies to zero.")
    interior = numerator if den_name == "1" and den_sign == "positive" else numerator / denominator
    if inverse == "arcsin":
        value = sp.simplify(interior)
        if getattr(value, "is_number", False) and getattr(value, "is_real", False) and abs(value) > 1:
            return Execution("contradictory", reason="The argument of arcsin is outside its domain.")
    body = sympy.atan(interior) if inverse == "arctan" else sympy.asin(interior)
    if outer == "negative":
        body = -body
    return Execution("derived", _function_text(body))


def _moment_arm(texts: dict[str, str], key: str) -> Execution:
    try:
        coords = _vector(texts["184"])
        formula = _apply_sign("positive", moment_term(texts, coords, key))
    except Exception as exc:
        return Execution("contradictory", reason=str(exc))
    arm = _canonical_arm(formula.split("·", 1)[1].rsplit("/", 1)[0])
    if arm not in ARM_OPTIONS:
        return Execution("contradictory", reason=f"The coordinate component {arm} is not among the options.")
    return Execution("derived", arm)


def _moment_denom(texts: dict[str, str]) -> Execution:
    raw = str(texts.get("182", "")).replace(" ", "")
    if "L" not in raw:
        return Execution("contradictory", reason="ID182 did not give a length made of L, L_t, and L_r.")
    return Execution("derived", "L_RT")


def _compose_moment(texts: dict[str, str], sign_id: str, force_id: str, arm_id: str, denom_id: str) -> Execution:
    sign = str(texts.get(sign_id, "")).strip()
    force = str(texts.get(force_id, "")).strip()
    arm = str(texts.get(arm_id, "")).strip()
    denom = str(texts.get(denom_id, "")).strip()
    if sign not in {"positive", "negative"}:
        return Execution("contradictory", reason=f"The sign of ID{sign_id} is not positive or negative.")
    if force not in FORCE_OPTIONS:
        return Execution("contradictory", reason=f"The force coefficient of ID{force_id} is not one of CN, CA, and CZ.")
    if arm not in ARM_OPTIONS:
        return Execution("contradictory", reason=f"The coordinate component of ID{arm_id} is not among the options.")
    if denom not in DENOM_OPTIONS:
        return Execution("contradictory", reason=f"The denominator of ID{denom_id} is not one of L, L_t, L_r, and L_RT.")
    outer = "-" if sign == "negative" else ""
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
        raise RuntimeError("The moment term is not divided by L_RT.")
    force = next((name for name, symbol in FORCE_SYMBOL.items() if symbol in numer.free_symbols), None)
    if force is None:
        raise RuntimeError("The moment term does not contain CA, CZ, or CN.")
    arm = sp.expand(sp.simplify(numer / FORCE_SYMBOL[force]))
    positive = _positive_arm(arm)
    arm_text = str(positive)
    if any(operator in arm_text for operator in ("+", "-")):
        arm_text = f"({arm_text})"
    outer = "-" if sign == "negative" else ""
    return f"{outer}{force}·{arm_text}/L_RT"


def _positive_arm(arm):
    if sp.simplify(arm) == 0:
        raise RuntimeError("The moment-arm length is zero.")
    polynomial = sp.Poly(sp.expand(arm), _XR, XG_SYM, YG_SYM, ZG_SYM)
    leading = sp.simplify(polynomial.LC())
    if leading not in (1, -1):
        raise RuntimeError(f"The leading coefficient of the moment arm {arm} is not 1 or -1.")
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
        raise RuntimeError("It is not a vector.")
    values = [_adopt(item) for item in list(parsed)]
    if len(values) != 3:
        raise RuntimeError("The reference-point coordinate does not have three components.")
    return values


def _component(texts: dict[str, str], name: str):
    if name == "1":
        return sp.Integer(1)
    node_id = _COMPONENT.get(name)
    if node_id is None:
        raise RuntimeError(f"The component {name!r} is not XTZ, YTZ, ZTZ, or 1.")
    return _parse(texts[node_id])


def _signed(sign: str, expr):
    if sign == "positive":
        return expr
    if sign == "negative":
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
