# -*- coding: utf-8 -*-
"""Generate cloze answers for balance elastic angles in the wind frame.

The questions follow doc/TP_FD.md. Axis directions, installation sense, elastic
bends, and increment signs come from BodyFrameModel. Passive rotation is fixed.

The level balance frame is aligned with the wind frame by RP_TP_FD. The
installation roll is applied first, so

    [XFD, YFD, ZFD]^T = R1*R2*[XTP, YTP, ZYP]^T

with R2 the passive installation and R1 = RP_TP_FD. A positive elastic bend
moves the balance axial direction; the small-angle axial vector is then taken
through that product. Delta-alpha and delta-beta are positive when the resulting
nose motion increases the model's own angles.

Run::

    python -m domains.wind_tunnel.tp_fd_generate
    python -m domains.wind_tunnel.tp_fd_generate --spec declaration.json --out tp_fd_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from body_frame_model import (
    ALPHA,
    ALPHA_E,
    ALPHA_V,
    BALANCE_FRONT,
    BALANCE_VOCAB,
    BETA,
    BETA_E,
    BETA_V,
    DefinitionError,
    DELTA_ALPHA,
    DELTA_BETA,
    DELTA_GAMMA,
    FRONT,
    GAMMA,
    GAMMA_E,
    GAMMA_TP,
    NEGATIVE,
    NEGATIVE_ROTATION,
    OPPOSITE,
    PLACE_INVERTED,
    PLACE_UPRIGHT,
    POSITIVE,
    POSITIVE_ROTATION,
    ROTATION_ANGLE,
    SENSE_OPPOSITE,
    SENSE_SAME,
    UNKNOWN,
    WIND_VOCAB,
    BodyFrameModel,
    roll_about_nose_sense,
    _VECTOR,
    _body_direction_in_tunnel,
    _dot,
)


ROLE_ZH = {
    "forward": "front",
    "rearward": "rear",
    "up": "upward",
    "down": "downward",
    "left": "left",
    "right": "right",
}
AXIS_ROLE_ZH = {"axial": "axial", "normal": "normal", "lateral": "lateral"}
SIGN_ZH = {POSITIVE: "positive", NEGATIVE: "negative", UNKNOWN: "indeterminate"}
DIRECTED = [
    "XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP",
    "-XFD", "-YFD", "-ZFD", "-XTZ", "-YTZ", "-ZTZ", "-XTP", "-YTP", "-ZYP",
]
PLAIN_AXES = ["XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP"]
WIND_COMPONENTS = ["XFD", "YFD", "ZFD"]
FRAMES = ["airframe", "balance frame", "wind frame"]
CONVERSION_PAIRS = [f"{source} to {target}" for source in FRAMES for target in FRAMES]
DIRECTIONS = ["upward", "downward", "left", "right", "front", "rear"]
SIGNS = ["positive", "negative", "indeterminate"]
ROLES = ["axial", "normal", "lateral"]
TURNS = ["positive", "reverse"]
CLOCKS = ["counterclockwise", "clockwise"]
MOUNTS = ["upright", "inverted"]
VERTICAL = ["upward", "downward"]
SIDES = ["left", "right"]
HAND = ["positive", "negative"]
RELATIONS = ["same", "opposite"]

ALPHA_E_SYM, BETA_E_SYM, GAMMA_TP_SYM = sp.symbols("alpha_e beta_e gamma_tp", real=True)
DISPLAY = {
    "alpha_e": "α_e",
    "beta_e": "β_e",
    "gamma_e": "γ_e",
    "gamma_tp": "γ_tp",
}
ROLE_VECTOR = {"normal": _VECTOR["body_up"], "lateral": _VECTOR["body_right"], "axial": _VECTOR["body_forward"]}

STANDARD_SPEC = {
    "alpha": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma": {"right_wing_toward": "body_down", "sign": "positive"},
    "alpha_m": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta_m": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma_m": {"right_wing_toward": "body_down", "sign": "positive"},
    "placement": "upright",
    "body_frame": {
        "chirality": "right",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    },
    "wind_frame": {
        "chirality": "right",
        "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
        "YFD": {"along": "tunnel_normal", "points": "tunnel_up"},
    },
    "alpha_e": {"axial_bends": "balance_up", "sign": "positive"},
    "beta_e": {"axial_bends": "balance_left", "sign": "positive"},
    "gamma_e": {"lateral_moves": "balance_down", "sign": "positive"},
    "alpha_v": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta_v": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma_tp": {"sense": "same"},
    "balance_frame": {
        "chirality": "right",
        "XTP": {"along": "balance_axial", "points": "balance_forward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    },
}


def _text(expr) -> str:
    rendered = str(sp.simplify(expr))
    for source, target in DISPLAY.items():
        rendered = rendered.replace(source, target)
    return rendered


def _matrix_text(matrix: sp.Matrix) -> str:
    rows = []
    for row in range(matrix.rows):
        cells = ", ".join(_text(matrix[row, column]) for column in range(matrix.cols))
        rows.append(f"[{cells}]")
    return "[" + ", ".join(rows) + "]"


def _vector_text(values) -> str:
    return "[" + ", ".join(_text(value) for value in values) + "]"


def _axial_hand_sign(right_wing_toward: str) -> str:
    return "positive" if roll_about_nose_sense(right_wing_toward) == SENSE_SAME else "negative"


def _balance_roll_relation(model: BodyFrameModel) -> str:
    """Whether positive elastic roll matches the right-hand rule about balance forward."""
    toward = model.elastic_angles[GAMMA_E].positive_toward
    sense = model._roll_sense(_VECTOR[BALANCE_FRONT], toward)
    if sense == POSITIVE_ROTATION:
        return "same"
    if sense == NEGATIVE_ROTATION:
        return "opposite"
    raise DefinitionError("Elastic roll has no sense about the balance axial positive direction.")


def _direction_zh(name: str) -> str:
    return ROLE_ZH[name.split("_", 1)[1]]


def _role(absolute_axis: str) -> str:
    return absolute_axis.split("_", 1)[1]


def _directed(frame: dict, target) -> str:
    for name, axis in frame.items():
        alignment = _dot(axis.vector, target)
        if alignment == 1:
            return name
        if alignment == -1:
            return f"-{name}"
    raise DefinitionError("No coordinate axis lies on the requested absolute direction.")


def _coords(vector, frame: dict, names: tuple[str, ...]) -> list[int]:
    return [_dot(vector, frame[name].vector) for name in names]


def _sense_number(rotation: str) -> int:
    if rotation == POSITIVE_ROTATION:
        return 1
    if rotation == NEGATIVE_ROTATION:
        return -1
    raise DefinitionError("The installation or elastic roll has no right-hand sense.")


def _small_sign(expr, positives: tuple) -> str:
    expanded = sp.expand(expr)
    constant = expanded
    coefficients = []
    for symbol in positives:
        coefficients.append(sp.simplify(expanded.coeff(symbol)))
        constant = sp.simplify(constant.subs(symbol, 0))
    if constant != 0:
        if constant > 0:
            return "positive"
        if constant < 0:
            return "negative"
        return "indeterminate"
    usable = []
    for coefficient in coefficients:
        if coefficient == 0:
            continue
        if coefficient > 0:
            usable.append(1)
        elif coefficient < 0:
            usable.append(-1)
        else:
            return "indeterminate"
    if len(set(usable)) != 1:
        return "indeterminate"
    return "positive" if usable[0] > 0 else "negative"


def _projection(components, frame: dict, names: tuple[str, ...], target) -> sp.Expr:
    total = 0
    for name, value in zip(names, components):
        total += value * _dot(frame[name].vector, target)
    return sp.simplify(total)


def generate(spec: dict) -> dict:
    """Build the cloze key for one declaration."""
    model = BodyFrameModel().declare(spec)
    if model.placement not in (PLACE_UPRIGHT, PLACE_INVERTED):
        raise DefinitionError("placement must be upright or inverted.")
    for frame, names in (
        (model.wind_frame, WIND_VOCAB.coordinate_axes),
        (model.balance_frame, BALANCE_VOCAB.coordinate_axes),
        (model.body_frame, ("XTZ", "YTZ", "ZTZ")),
    ):
        if any(not frame[name].points_to for name in names):
            raise DefinitionError("Wind, balance, and body frames must be complete.")
    install = model.installation
    if install.rotation_when_positive not in (POSITIVE_ROTATION, NEGATIVE_ROTATION):
        raise DefinitionError("gamma_tp must declare its sense.")
    elastic = [model.elastic_angles[name] for name in (ALPHA_E, BETA_E, GAMMA_E)]
    if any(not record.positive_toward or not record.associated_axis for record in elastic):
        raise DefinitionError("alpha_e, beta_e, and gamma_e must be declared.")
    alpha_e, beta_e, gamma_e = elastic
    if _sense_number(gamma_e.rotation_when_positive) == 0:
        raise DefinitionError("gamma_e has no torsion sense.")

    balance_names = BALANCE_VOCAB.coordinate_axes
    wind_names = WIND_VOCAB.coordinate_axes
    alignment = sp.Matrix([
        [
            _dot(model.wind_frame[wind_name].vector, model.balance_frame[balance_name].vector)
            for balance_name in balance_names
        ]
        for wind_name in wind_names
    ])
    axial = model.balance_frame[install.associated_axis]
    coordinate_sign = _sense_number(axial.installation_rotation_when_positive)
    theta = coordinate_sign * GAMMA_TP_SYM
    roll = sp.simplify(axial.rotation_matrix.subs(ROTATION_ANGLE, theta))

    base = _coords(_VECTOR[BALANCE_FRONT], model.balance_frame, balance_names)
    alpha_shift = _coords(_VECTOR[alpha_e.positive_toward], model.balance_frame, balance_names)
    beta_shift = _coords(_VECTOR[beta_e.positive_toward], model.balance_frame, balance_names)
    deformed = [
        base[index] + alpha_shift[index] * ALPHA_E_SYM + beta_shift[index] * BETA_E_SYM
        for index in range(3)
    ]
    wind_vector = sp.simplify(alignment * roll * sp.Matrix(deformed))
    level_wind = [component.subs(GAMMA_TP_SYM, 0) for component in wind_vector]

    def signed_angle(theory_name: str, airspeed: bool) -> sp.Expr:
        if airspeed:
            toward = model.angles[theory_name].positive_airspeed_toward
        else:
            toward = model.angles[theory_name].right_wing_toward
        if not toward:
            raise DefinitionError(f"{theory_name} must declare its positive direction.")
        nose = _body_direction_in_tunnel(OPPOSITE[toward], model.placement)
        forward = _projection(wind_vector, model.wind_frame, wind_names, _VECTOR[FRONT])
        if forward == 0:
            raise DefinitionError("The deformed axial vector has no tunnel-forward component.")
        return sp.simplify(
            _projection(wind_vector, model.wind_frame, wind_names, nose) / forward
        )

    def component_formula(theory_name: str) -> sp.Expr:
        numerator, denominator = _wind_parts(theory_name)
        return sp.simplify(numerator / denominator)

    def _wind_parts(theory_name: str) -> tuple[sp.Expr, sp.Expr]:
        toward = model.angles[theory_name].positive_airspeed_toward
        nose = _body_direction_in_tunnel(OPPOSITE[toward], model.placement)
        symbols = [sp.Symbol(name) for name in wind_names]
        forward = _projection(symbols, model.wind_frame, wind_names, _VECTOR[FRONT])
        return _projection(symbols, model.wind_frame, wind_names, nose), forward

    value_signs = {
        name: _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM))
        for name, value in zip(WIND_COMPONENTS, level_wind)
    }

    def quotient_parts(theory_name: str) -> dict:
        delta_name = DELTA_ALPHA if theory_name == ALPHA else DELTA_BETA
        outer = SIGN_ZH.get(model.increments[delta_name].increment_sign, "")
        return _positive_quotient(*_wind_parts(theory_name), value_signs, outer)

    delta_alpha = signed_angle(ALPHA, True)
    delta_beta = signed_angle(BETA, True)

    def flow_sign(flow_name: str, theory_name: str) -> str:
        flow = model.flow_angles[flow_name].positive_airspeed_toward
        theory = model.angles[theory_name].positive_airspeed_toward
        if not flow or not theory:
            raise DefinitionError(f"{flow_name} and {theory_name} must both be declared.")
        return "positive" if flow == theory else "negative"

    items: list[dict] = []

    def add(step: int, label: str, prompt: str, answer, options=None) -> None:
        items.append({
            "id": len(items) + 1,
            "manuscript_id": str(len(items) + 1),
            "step": step,
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer if isinstance(answer, str) else _text(answer),
        })

    add(1, "C2", "To compute the angle-of-attack and sideslip increments that the balance elastic angles produce in the wind frame, the required transform", "balance frame to wind frame", CONVERSION_PAIRS)
    add(1, "C1", "The positive absolute tunnel axis (forward)", _directed(model.wind_frame, ROLE_VECTOR["axial"]), DIRECTED)
    add(1, "C1", "The positive absolute balance axis (forward)", _directed(model.balance_frame, ROLE_VECTOR["axial"]), DIRECTED)
    for role_name in ("normal", "lateral", "axial"):
        add(1, "C1", f"the coordinate axis for the positive absolute {role_name} of the wind frame", _directed(model.wind_frame, ROLE_VECTOR[role_name]), DIRECTED)
    for role_name in ("normal", "lateral", "axial"):
        add(
            1, "C1", f"the coordinate axis for the positive absolute {role_name} of the balance frame",
            _directed(model.balance_frame, ROLE_VECTOR[role_name]), DIRECTED,
        )
    add(1, "A2", "the transform matrix is RP_TP_FD", _matrix_text(alignment))
    add(1, "C2", "the frame whose axis directions RP_TP_FD aligns", "balance frame", FRAMES)
    add(1, "C1", "the frame whose coordinate directions are made the same", "wind frame", FRAMES)

    add(2, "A1", "When γ_tp > 0, the rotation is about this absolute direction of the balance", AXIS_ROLE_ZH[_role(axial.absolute_axis)], ROLES)
    add(2, "C1", "On the coordinate axes, the rotation is about", axial.name, PLAIN_AXES)
    add(2, "A1", "The sense about that axis", "positive" if coordinate_sign == 1 else "reverse", TURNS)
    add(2, "A2", "the rotation matrix is R_γ_tp", _matrix_text(roll))

    factor_options = ["RP_TP_FD", "R_γ_tp"]
    add(3, "A2", "R1 in the product", "RP_TP_FD", factor_options)
    add(3, "A2", "R2 in the product", "R_γ_tp", factor_options)

    add(4, "A1", "In the balance frame, the unit vector along the positive absolute balance axis (forward)", _vector_text(base))
    add(4, "C1", "When the balance produces α_e, the balance axis bends toward", _direction_zh(alpha_e.positive_toward), DIRECTIONS)
    add(4, "C1", "When the balance produces β_e, the balance axis bends toward", _direction_zh(beta_e.positive_toward), DIRECTIONS)
    add(
        4, "A1", "Looking along the positive absolute balance axis (forward), a positive elastic roll twists",
        "clockwise" if _balance_roll_relation(model) == "same" else "counterclockwise",
        CLOCKS,
    )
    for record, angle_name in ((alpha_e, "elastic angle of attack"), (beta_e, "elastic sideslip"), (gamma_e, "elastic roll")):
        add(4, "C1", f"the rotation axis of the {angle_name}", record.associated_axis, PLAIN_AXES)
    for name, value in zip(("XTP_e", "YTP_e", "ZTP_e"), deformed):
        add(4, "A1", f"the sign of {name} under a positive elastic angle", _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM)), SIGNS)
    add(4, "A1", "The axial unit vector in the original frame after the small-angle approximation", _vector_text(deformed))

    for name, value in zip(WIND_COMPONENTS, wind_vector):
        add(5, "A2", f"{name} after substituting the deformed axial unit vector", _text(value))

    add(6, "A1", "For a positive theoretical angle of attack, the wind-frame component used by Δα_e", _component_name(component_formula(ALPHA)), WIND_COMPONENTS)
    add(6, "A1", "For a positive theoretical sideslip, the wind-frame component used by Δβ_e", _component_name(component_formula(BETA)), WIND_COMPONENTS)

    add(7, "C1", "When the deformations occur, the model is", "upright" if model.placement == PLACE_UPRIGHT else "inverted", MOUNTS)
    add(7, "A1", "After the elastic angle of attack the airspeed lies this way of the model", _direction_zh(model.increments[DELTA_ALPHA].effect_toward), VERTICAL)
    add(7, "A1", "After the elastic sideslip the airspeed lies this way of the model", _direction_zh(model.increments[DELTA_BETA].effect_toward), SIDES)
    add(7, "A1", "The rotation about the positive absolute balance axis (forward) relative to the right-hand rule", _balance_roll_relation(model), RELATIONS)
    add(7, "C1", "For a positive theoretical angle of attack the airspeed points this way on the airframe", _direction_zh(model.angles[ALPHA].positive_airspeed_toward), DIRECTIONS)
    add(7, "C1", "For a positive theoretical sideslip the airspeed points this way on the airframe", _direction_zh(model.angles[BETA].positive_airspeed_toward), DIRECTIONS)
    add(7, "C1", "Roll is positive when it is this relation to the right-hand rule about the positive absolute airframe axis (forward)", _axial_hand_sign(model.angles[GAMMA].right_wing_toward), HAND)
    for delta_name, prompt in (
        (DELTA_ALPHA, "Δα_e produced by the balance elastic angles"),
        (DELTA_BETA, "Δβ_e produced by the balance elastic angles"),
        (DELTA_GAMMA, "Δγ_e produced by the balance elastic angles"),
    ):
        sign = model.increments[delta_name].increment_sign
        add(7, "A1", prompt, SIGN_ZH.get(sign, "indeterminate"), SIGNS)
    for name, value in zip(WIND_COMPONENTS, level_wind):
        add(7, "A1", f"When γ_tp = 0 and the elastic angles are positive, the sign of {name}", _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM)), SIGNS)

    inside = "The numerator and the denominator must both be positive. The sign may be written only outside the quotient"
    for symbol, theory_name in (("Δα_e", ALPHA), ("Δβ_e", BETA)):
        parts = quotient_parts(theory_name)
        add(8, "A1", f"The angle sign of {symbol}, written outside the quotient. {inside}", parts["outer"], SIGNS)
        add(8, "A1", f"From the sign of the chosen wind-frame component, the numerator sign of {symbol}. {inside}", parts["num_sign"], SIGNS)
        add(8, "A1", f"The wind-frame component in the numerator of {symbol}. {inside}", parts["num_axis"], WIND_COMPONENTS)
        add(8, "A1", f"From the sign of the chosen wind-frame component, the denominator sign of {symbol}. {inside}", parts["den_sign"], SIGNS)
        add(8, "A1", f"The wind-frame component in the denominator of {symbol}. {inside}", parts["den_axis"], WIND_COMPONENTS)
    for symbol, theory_name in (("Δα_e", ALPHA), ("Δβ_e", BETA)):
        parts = quotient_parts(theory_name)
        prompt = f"{symbol} after substitution. The numerator and the denominator are both positive, and a minus sign is written only before the whole quotient"
        if symbol == "Δα_e":
            prompt += ". This blank must be a quotient of the wind-frame components XFD, YFD, and ZFD. Do not write an elastic-angle expression that contains α_e, β_e, or γ_e"
        add(8, "A2", prompt, parts["formula"])
    add(8, "A2", "In α_e and β_e, Δα_e", _text(delta_alpha))
    add(8, "A2", "In α_e and β_e, Δβ_e", _text(delta_beta))
    roll_sign = model.increments[DELTA_GAMMA].increment_sign
    if roll_sign == POSITIVE:
        delta_gamma = "γ_e"
    elif roll_sign == NEGATIVE:
        delta_gamma = "-γ_e"
    else:
        raise DefinitionError("Δγ_e has no sign.")
    add(8, "A1", "The roll increment produced by the balance elastic angle does not change with γ_tp. Its sign is whether the elastic roll and the theoretical roll have the same sense about the positive absolute axis (forward)", delta_gamma)

    add(9, "A1", "The sign of the compensation Δα_V, positive when it increases the actual angle of attack", flow_sign(ALPHA_V, ALPHA), SIGNS)
    add(9, "A1", "The sign of the compensation Δβ_V, positive when it increases the actual sideslip", flow_sign(BETA_V, BETA), SIGNS)

    return {
        "task": "balance_elastic_angle_to_wind_frame",
        "declaration": spec,
        "fixed": {
            "transform": "passive",
            "product": "[XFD, YFD, ZFD]^T = R1*R2*[XTP, YTP, ZYP]^T",
            "small_angle": "sin(theta) = theta, cos(theta) = 1",
        },
        "items": items,
    }


def _single_axis_term(expr) -> tuple[int, str]:
    expanded = sp.expand(expr)
    found = None
    for name in WIND_COMPONENTS:
        coefficient = sp.simplify(expanded.coeff(sp.Symbol(name)))
        if coefficient == 0:
            continue
        if found is not None or coefficient not in (1, -1):
            raise DefinitionError(f"A wind-angle part must be one signed axis, got {expr}.")
        found = (int(coefficient), name)
    remainder = sp.simplify(expanded.subs({sp.Symbol(name): 0 for name in WIND_COMPONENTS}))
    if found is None or remainder != 0:
        raise DefinitionError(f"A wind-angle part must be one signed axis, got {expr}.")
    return found


def _sign_number(sign: str) -> int:
    if sign == "positive":
        return 1
    if sign == "negative":
        return -1
    raise DefinitionError(f"A wind component sign must be positive or negative, got {sign}.")


def _signed_factor(sign: str, axis: str) -> str:
    if sign == "positive":
        return f"({axis})"
    return f"(-{axis})"


def _positive_quotient(numerator, denominator, value_signs: dict, outer: str) -> dict:
    """Keep each written part positive when the component itself may be negative.

    A negative component is written with a minus inside the numerator or denominator.
    The angle sign stays outside and follows the sign of the angle, not that minus.
    """
    num_coefficient, num_axis = _single_axis_term(numerator)
    den_coefficient, den_axis = _single_axis_term(denominator)
    num_sign = value_signs[num_axis]
    den_sign = value_signs[den_axis]
    if outer not in ("positive", "negative"):
        raise DefinitionError(f"The wind-angle sign must be positive or negative, got {outer}.")
    geometric = num_coefficient * _sign_number(num_sign) * den_coefficient * _sign_number(den_sign)
    if geometric != _sign_number(outer):
        raise DefinitionError("The outside angle sign does not match the positive wind-component quotient.")
    formula = f"{_signed_factor(num_sign, num_axis)}/{_signed_factor(den_sign, den_axis)}"
    if outer == "negative":
        formula = f"-{formula}"
    return {
        "outer": outer,
        "num_sign": num_sign,
        "num_axis": num_axis,
        "den_sign": den_sign,
        "den_axis": den_axis,
        "formula": formula,
    }


def _component_name(formula) -> str:
    """The wind component in the numerator of the small-angle formula."""
    numerator, _denominator = sp.fraction(sp.together(formula))
    measuring = [name for name in WIND_COMPONENTS if sp.Symbol(name) in numerator.free_symbols]
    if len(measuring) != 1:
        raise DefinitionError(f"Cannot tell which wind component represents {formula}.")
    return measuring[0]


def _render(payload: dict) -> str:
    lines = ["# balance frame to wind frame answers", ""]
    step = None
    for item in payload["items"]:
        if item["step"] != step:
            step = item["step"]
            lines.append(f"## Step {step}")
            lines.append("")
        options = ""
        if item["options"]:
            options = "（" + "/".join(item["options"]) + "）"
        lines.append(
            f"- ID {item['manuscript_id']} [{item['label']}] {item['prompt']}{options}: {item['answer']}"
        )
    lines.append("")
    return "\n".join(lines)


def _check_standard() -> None:
    payload = generate(STANDARD_SPEC)
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    assert len(payload["items"]) == 64
    assert by_id["1"] == "balance frame to wind frame"
    assert by_id["2"] == "XFD"
    assert by_id["3"] == "XTP"
    assert [by_id[str(index)] for index in range(4, 10)] == ["YFD", "ZFD", "XFD", "YTP", "ZYP", "XTP"]
    assert by_id["10"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert by_id["11"] == "balance frame"
    assert by_id["12"] == "wind frame"
    assert by_id["13"] == "axial"
    assert by_id["14"] == "XTP"
    assert by_id["15"] == "positive"
    assert [by_id["17"], by_id["18"]] == ["RP_TP_FD", "R_γ_tp"]
    assert by_id["19"] == "[1, 0, 0]"
    assert by_id["20"] == "upward"
    assert by_id["21"] == "left"
    assert by_id["22"] == "clockwise"
    assert by_id["23"] == "ZYP"
    assert by_id["24"] == "YTP"
    assert by_id["25"] == "XTP"
    assert by_id["26"] == "positive"
    assert by_id["27"] == "positive"
    assert by_id["28"] == "negative"
    assert by_id["29"] == "[1, α_e, -β_e]"
    assert by_id["30"] == "1"
    assert by_id["33"] == "YFD"
    assert by_id["34"] == "ZFD"
    assert by_id["35"] == "upright"
    assert by_id["36"] == "downward"
    assert by_id["37"] == "right"
    assert by_id["38"] == "same"
    assert by_id["39"] == "downward"
    assert by_id["40"] == "right"
    assert by_id["41"] == "positive"
    assert by_id["42"] == "positive"
    assert by_id["43"] == "positive"
    assert by_id["44"] == "positive"
    assert by_id["45"] == "positive"
    assert by_id["46"] == "positive"
    assert by_id["47"] == "negative"
    assert [by_id[str(index)] for index in range(48, 53)] == ["positive", "positive", "YFD", "positive", "XFD"]
    assert [by_id[str(index)] for index in range(53, 58)] == ["positive", "negative", "ZFD", "positive", "XFD"]
    assert by_id["58"] == "(YFD)/(XFD)"
    assert by_id["59"] == "(-ZFD)/(XFD)"
    assert by_id["60"] == "α_e*cos(γ_tp) - β_e*sin(γ_tp)"
    assert by_id["61"] == "α_e*sin(γ_tp) + β_e*cos(γ_tp)"
    assert by_id["62"] == "γ_e"
    assert by_id["63"] == "positive"
    assert by_id["64"] == "positive"

    inverted = json.loads(json.dumps(STANDARD_SPEC))
    inverted["placement"] = PLACE_INVERTED
    flipped = {item["manuscript_id"]: item["answer"] for item in generate(inverted)["items"]}
    assert flipped["10"] == by_id["10"]
    assert flipped["35"] == "inverted"
    assert flipped["36"] == "upward"
    assert flipped["37"] == "left"
    assert flipped["38"] == "same"
    assert flipped["41"] == "positive"
    assert flipped["42"] == "negative"
    assert flipped["43"] == "negative"
    assert flipped["44"] == "positive"
    assert flipped["48"] == "negative"
    assert flipped["50"] == "YFD"
    assert flipped["58"] == "-(YFD)/(XFD)"
    assert flipped["60"] == "-α_e*cos(γ_tp) + β_e*sin(γ_tp)"
    assert flipped["62"] == "γ_e"

    opposite = json.loads(json.dumps(STANDARD_SPEC))
    opposite["gamma_tp"] = {"sense": SENSE_OPPOSITE}
    turned = {item["manuscript_id"]: item["answer"] for item in generate(opposite)["items"]}
    assert turned["15"] == "reverse"
    assert turned["60"] == "α_e*cos(γ_tp) + β_e*sin(γ_tp)"

    opposed = json.loads(json.dumps(STANDARD_SPEC))
    opposed["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    opposed_flow = {item["manuscript_id"]: item["answer"] for item in generate(opposed)["items"]}
    assert opposed_flow["63"] == "negative"

    rearward_balance = json.loads(json.dumps(STANDARD_SPEC))
    rearward_balance["balance_frame"] = {
        "chirality": "left",
        "XTP": {"along": "balance_axial", "points": "balance_rearward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    }
    rearward_balance_answers = {
        item["manuscript_id"]: item["answer"] for item in generate(rearward_balance)["items"]
    }
    assert rearward_balance_answers["3"] == "-XTP"
    assert rearward_balance_answers["9"] == "-XTP"
    assert rearward_balance_answers["15"] == "reverse"
    assert rearward_balance_answers["22"] == "clockwise"
    assert rearward_balance_answers["38"] == "same"
    assert rearward_balance_answers["44"] == "positive"
    assert rearward_balance_answers["62"] == "γ_e"

    roll_up = json.loads(json.dumps(STANDARD_SPEC))
    roll_up["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    roll_up["gamma"] = {"right_wing_toward": "body_up", "sign": "positive"}
    roll_up_answers = {item["manuscript_id"]: item["answer"] for item in generate(roll_up)["items"]}
    assert roll_up_answers["38"] == "opposite"
    assert roll_up_answers["41"] == "negative"
    assert roll_up_answers["62"] == "γ_e"

    elastic_up = json.loads(json.dumps(STANDARD_SPEC))
    elastic_up["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    elastic_up_answers = {item["manuscript_id"]: item["answer"] for item in generate(elastic_up)["items"]}
    assert elastic_up_answers["44"] == "negative"
    assert elastic_up_answers["62"] == "-γ_e"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the balance-to-wind elastic-angle cloze answer key.")
    parser.add_argument("--spec", type=Path, help="JSON declaration. Defaults to the standard upright frame.")
    parser.add_argument("--out", type=Path, default=Path("tp_fd_eval.json"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else STANDARD_SPEC
    if args.spec is None:
        _check_standard()
    payload = generate(spec)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = args.out.with_suffix(".md")
    markdown.write_text(_render(payload), encoding="utf-8")
    counts = {}
    for item in payload["items"]:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    print(f"Wrote {len(payload['items'])} blanks to {args.out}")
    print("Labels: " + ", ".join(f"{label} {counts[label]}" for label in ("C1", "C2", "A1", "A2")))


if __name__ == "__main__":
    main()
