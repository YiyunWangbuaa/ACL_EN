# -*- coding: utf-8 -*-
"""Generate cloze answers for the wind-frame to body-frame attitude evaluation.

The questions follow doc/TZ_FD.md. Geometry, signs, and matrices come from
BodyFrameModel under the given declaration. Passive rotation and intrinsic
moving-axis rotation are fixed by the aircraft and are not declaration choices.

Question IDs follow doc/TZ_FD.md. RP1 matches declared axis directions with the
model treated as upright, and that upright embedding is the intermediate body
frame used for the attitude rotations. RP2 is the extra mounting matrix:
identity when the model is upright, and a normal/lateral sign change when it
is inverted. The placed map is RP2*RP1, and it is applied before the attitude.

Run::

    python -m domains.wind_tunnel.tz_fd_generate
    python -m domains.wind_tunnel.tz_fd_generate --spec declaration.json --out tz_fd_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from body_frame_model import (
    ALPHA,
    ALPHA_M,
    AXIAL,
    BETA,
    BETA_M,
    BODY_AXES,
    BODY_VOCAB,
    DefinitionError,
    FRONT,
    RIGHT,
    GAMMA,
    GAMMA_M,
    INTRINSIC,
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
    TUNNEL_FRONT,
    SYMBOL,
    UNKNOWN,
    WIND_VOCAB,
    BodyFrameModel,
    roll_about_nose_sense,
    _VECTOR,
    _body_direction_in_tunnel,
    _cross,
    _direction_name,
    _dot,
)


ANGLE_ZH = {ALPHA: "angle of attack", BETA: "sideslip", GAMMA: "roll"}
MECHANISM_OF = {ALPHA: ALPHA_M, BETA: BETA_M, GAMMA: GAMMA_M}
ROLE_ZH = {
    "forward": "front",
    "rearward": "rear",
    "up": "upward",
    "down": "downward",
    "left": "left",
    "right": "right",
}
SIGN_ZH = {POSITIVE: "positive", NEGATIVE: "negative", UNKNOWN: "indeterminate"}
ROTATION_ZH = {POSITIVE_ROTATION: "positive", NEGATIVE_ROTATION: "negative"}
FRAMES = ["intermediate body-axis frame", "balance frame", "wind frame"]
CONVERSION_FRAMES = ["body-axis frame", "balance frame", "wind frame"]
CONVERSION_PAIRS = [
    f"{source} to {target}" for source in CONVERSION_FRAMES for target in CONVERSION_FRAMES
]
DIRECTIONS = ["upward", "downward", "left", "right", "front", "rear"]
SIGNS = ["positive", "negative", "indeterminate"]
EULER = ["angle of attack", "sideslip", "roll"]
TRANSFORMS = ["active", "passive"]
ROTATIONS = ["intrinsic", "extrinsic"]
FRAME_KINDS = ["moving axes", "fixed axes"]
LEVEL_FRAMES = ["level airframe", "wind frame"]
MOUNTS = ["upright", "inverted"]
AXIS_RELATIONS = ["same", "opposite"]
INVERSES = ["arcsin", "arctan"]
COMPONENTS = ["XTZ", "YTZ", "ZTZ", "1"]
AXES = ["XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP"]
DIRECTED = [
    "XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP",
    "-XFD", "-YFD", "-ZFD", "-XTZ", "-YTZ", "-ZTZ", "-XTP", "-YTP", "-ZYP",
]

ALPHA_SYM, BETA_SYM, GAMMA_SYM = sp.symbols("alpha_M beta_M gamma_M", real=True)
SYMBOL_OF = {ALPHA_M: ALPHA_SYM, BETA_M: BETA_SYM, GAMMA_M: GAMMA_SYM}
DISPLAY = {
    "alpha_M": "α_M", "beta_M": "β_M", "gamma_M": "γ_M",
    "alpha_t": "α_t", "beta_t": "β_t",
    "gamma_tp": "γ_tp", "alpha_e": "α_e", "beta_e": "β_e", "gamma_e": "γ_e",
    "alpha_V": "α_V", "beta_V": "β_V",
}

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
    "gamma_model": {"sense": "same"},
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


def _role(direction: str) -> str:
    return ROLE_ZH[direction.split("_", 1)[1]]


def _tunnel_direction(body_direction: str, placement: str) -> str:
    return _direction_name(_body_direction_in_tunnel(body_direction, placement), WIND_VOCAB)


def _body_from_tunnel(tunnel_direction: str, placement: str) -> str:
    target = _VECTOR[tunnel_direction]
    for body_direction in BODY_VOCAB.directions:
        if _body_direction_in_tunnel(body_direction, placement) == target:
            return body_direction
    raise DefinitionError(f"No body direction matches {tunnel_direction} under {placement}.")


def _nose_and_airspeed(model: BodyFrameModel, airspeed_toward: str) -> tuple[str, str]:
    """Nose motion in the tunnel, then the opposite airflow read on the body."""
    nose_tunnel = _tunnel_direction(OPPOSITE[airspeed_toward], model.placement)
    inferred = _body_from_tunnel(WIND_VOCAB.opposite[nose_tunnel], model.placement)
    if inferred != airspeed_toward:
        raise DefinitionError("Airspeed inferred from the axial deflection disagrees with the declaration.")
    return nose_tunnel, inferred


def _declared_roll_hand(right_wing_toward: str) -> str:
    """Hand relation of γ_M about the model's nose. Installation does not enter."""
    sense = roll_about_nose_sense(right_wing_toward)
    if sense == SENSE_SAME:
        return "same"
    if sense == SENSE_OPPOSITE:
        return "opposite"
    raise DefinitionError("Mechanism roll has no right-hand relation about the model axis.")


def _wing_about_nose(relation: str) -> str:
    """Body direction of the right wing after a positive roll about the nose."""
    rh_wing = _direction_name(_cross(_VECTOR[FRONT], _VECTOR[RIGHT]), BODY_VOCAB)
    if relation == "same":
        return rh_wing
    if relation == "opposite":
        return OPPOSITE[rh_wing]
    raise DefinitionError("Mechanism roll has no right-hand relation about the model axis.")


def _coordinate_along(frame: dict, absolute_axis: str) -> str:
    matches = [name for name, axis in frame.items() if axis.absolute_axis == absolute_axis]
    if len(matches) != 1:
        raise DefinitionError(f"{absolute_axis} must belong to exactly one coordinate axis, got {matches}.")
    return matches[0]


def _directed(frame: dict, target) -> str:
    for name, axis in frame.items():
        alignment = _dot(axis.vector, target)
        if alignment == 1:
            return name
        if alignment == -1:
            return f"-{name}"
    raise DefinitionError("No coordinate axis lies on the requested absolute direction.")


def _same_sense(left: str, right: str) -> str:
    if not left or not right:
        return "indeterminate"
    if left == right:
        return "positive"
    if left == OPPOSITE.get(right) or right == OPPOSITE.get(left):
        return "negative"
    return "indeterminate"


def _signed_component(model: BodyFrameModel, direction: str) -> tuple[str, str]:
    for name in BODY_AXES:
        pointed = model.body_frame[name].points_to
        if pointed == direction:
            return name, "positive"
        if pointed == OPPOSITE[direction]:
            return name, "negative"
    raise DefinitionError(f"{direction} is not aligned with XTZ, YTZ, or ZTZ.")


def _apply_sign(sign: str, expr) -> sp.Expr:
    return expr if sign == "positive" else -expr


def _magnitude_sign(signs: dict, axis: str, what: str) -> str:
    """Sign that makes this component positive when the mechanism angle is positive."""
    sign = signs.get(axis, "")
    if sign not in (POSITIVE, NEGATIVE):
        raise DefinitionError(
            f"{what} needs a definite sign on {axis} so the inverse-function argument stays positive."
        )
    return SIGN_ZH[sign]


def _require_positive_part(expr, active, what: str) -> None:
    """A positive mechanism angle must make this numerator or denominator positive by itself."""
    value = sp.simplify(expr.subs({
        ALPHA_SYM: 0,
        BETA_SYM: 0,
        GAMMA_SYM: 0,
        active: sp.Rational(1, 10),
    }))
    if not value.is_real or value <= 0:
        raise DefinitionError(f"{what} must be positive on its own, got {value}.")


def _signed_formula(outer: str, inverse: str, interior: str) -> str:
    """Put the angle sign outside. Numerator and denominator are each a positive magnitude."""
    if outer not in ("positive", "negative"):
        raise DefinitionError(f"{inverse} needs an outer angle sign of positive or negative, got {outer}.")
    body = f"{inverse}({interior})"
    return body if outer == "positive" else f"-{body}"


def _mechanism_matrix(model: BodyFrameModel, name: str) -> sp.Matrix:
    record = model.mechanism_angles[name]
    if record.rotation_when_positive not in ROTATION_ZH:
        raise DefinitionError(f"{name} has no rotation sense on its associated axis.")
    axis = model.body_frame[record.associated_axis]
    angle = SYMBOL_OF[name]
    if record.rotation_when_positive == NEGATIVE_ROTATION:
        angle = -angle
    return sp.simplify(axis.rotation_matrix.subs(ROTATION_ANGLE, angle))


def _axis_alignment(model: BodyFrameModel) -> sp.Matrix:
    """Wind-to-body matrix at the upright embedding, before mounting."""
    return sp.Matrix([
        [
            _dot(_VECTOR[model.body_frame[body_name].points_to], model.wind_frame[wind_name].vector)
            for wind_name in WIND_VOCAB.coordinate_axes
        ]
        for body_name in BODY_AXES
    ])


def _mount_prematrix(model: BodyFrameModel, alignment: sp.Matrix) -> sp.Matrix:
    """RP2, with RP2*RP1 equal to the placed wind-to-body matrix."""
    placed = sp.Matrix(model.wind_to_body)
    if alignment.det() == 0:
        raise DefinitionError("RP1 is singular, so the mounting matrix cannot be split off.")
    mount = sp.simplify(placed * alignment.inv())
    if not (mount * alignment).equals(placed):
        raise DefinitionError("RP2*RP1 does not recover the placed wind-to-body matrix.")
    return mount


def _level_airspeed(model: BodyFrameModel) -> tuple[sp.Matrix, sp.Matrix]:
    """Express the oncoming-flow unit vector in wind coordinates, then in the level body frame.

    Wind-axis forward is the airspeed direction. That is the oncoming flow, opposite the airflow.
    """
    if model.wind_to_body is None:
        raise DefinitionError("placement, body_frame, and wind_frame are required before RP can be built.")
    wind_components = sp.Matrix([
        _dot(model.wind_frame[name].vector, _VECTOR[TUNNEL_FRONT])
        for name in WIND_VOCAB.coordinate_axes
    ])
    body_components = sp.simplify(model.wind_to_body * wind_components)
    return wind_components, body_components


def generate(spec: dict) -> dict:
    """Build the cloze key for one declaration."""
    model = BodyFrameModel().declare(spec)
    for name in (ALPHA, BETA, GAMMA, ALPHA_M, BETA_M, GAMMA_M):
        record = model.angles[name] if name in model.angles else model.mechanism_angles[name]
        if name in (GAMMA, GAMMA_M):
            if not record.right_wing_toward:
                raise DefinitionError(f"{name} must declare right_wing_toward.")
        elif not record.positive_airspeed_toward:
            raise DefinitionError(f"{name} must declare airspeed_toward.")
    if model.aircraft.rotation_type != INTRINSIC:
        raise DefinitionError("Aircraft rotation must stay intrinsic.")
    if any(not model.body_frame[name].points_to for name in BODY_AXES):
        raise DefinitionError("body_frame is incomplete.")
    if any(not model.wind_frame[name].points_to for name in WIND_VOCAB.coordinate_axes):
        raise DefinitionError("wind_frame is incomplete.")

    if model.placement not in (PLACE_UPRIGHT, PLACE_INVERTED):
        raise DefinitionError("placement must be upright or inverted before RP1 and RP2 can be split.")
    alignment = _axis_alignment(model)
    mount = _mount_prematrix(model, alignment)
    wind_components, level_body = _level_airspeed(model)
    matrices = {
        name: _mechanism_matrix(model, MECHANISM_OF[name])
        for name in model.euler_order
    }
    ordered = [matrices[name] for name in model.euler_order]
    first, second, third = ordered
    # Intrinsic passive product in the intermediate body frame. RP2*RP1 has
    # already mapped the wind coordinates, so the earliest rotation stays on the right.
    total = sp.simplify(third * second * first)
    posed = sp.simplify(total * level_body)
    component_expr = {name: posed[index] for index, name in enumerate(BODY_AXES)}

    alpha_num, _alpha_axis_sense = _signed_component(model, model.angles[ALPHA].positive_airspeed_toward)
    alpha_den, _forward_sense = _signed_component(model, FRONT)
    beta_num, _beta_axis_sense = _signed_component(model, model.angles[BETA].positive_airspeed_toward)
    alpha_signs = model.mechanism_angles[ALPHA_M].airspeed_signs
    beta_signs = model.mechanism_angles[BETA_M].airspeed_signs
    # Each side is signed from its own component. A positive ratio is not enough:
    # numerator and denominator must each be positive, so both-negative is rejected.
    alpha_num_sign = _magnitude_sign(alpha_signs, alpha_num, "α_t")
    alpha_den_sign = _magnitude_sign(alpha_signs, alpha_den, "α_t")
    beta_num_sign = _magnitude_sign(beta_signs, beta_num, "β_t")
    alpha_outer = _same_sense(
        model.mechanism_angles[ALPHA_M].positive_airspeed_toward,
        model.angles[ALPHA].positive_airspeed_toward,
    )
    beta_outer = _same_sense(
        model.mechanism_angles[BETA_M].positive_airspeed_toward,
        model.angles[BETA].positive_airspeed_toward,
    )
    expressed = {
        **component_expr,
        "1": sp.Integer(1),
    }
    alpha_num_expr = _apply_sign(alpha_num_sign, expressed[alpha_num])
    alpha_den_expr = _apply_sign(alpha_den_sign, expressed[alpha_den])
    beta_num_expr = _apply_sign(beta_num_sign, expressed[beta_num])
    alpha_answer = _signed_formula(
        alpha_outer, "arctan", f"({_text(alpha_num_expr)})/({_text(alpha_den_expr)})"
    )
    beta_answer = _signed_formula(beta_outer, "arcsin", _text(beta_num_expr))
    _require_positive_part(alpha_num_expr, ALPHA_SYM, "α_t numerator")
    _require_positive_part(alpha_den_expr, ALPHA_SYM, "α_t denominator")
    _require_positive_part(beta_num_expr, BETA_SYM, "β_t numerator")
    _require_positive_part(sp.Integer(1), BETA_SYM, "β_t denominator")

    items: list[dict] = []

    def add(manuscript_id: str, step: int, label: str, prompt: str, answer, options=None) -> None:
        number = int(manuscript_id)
        if number >= 2:
            number -= 1
        items.append({
            "id": len(items) + 1,
            "manuscript_id": str(number),
            "step": step,
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer if isinstance(answer, str) else _text(answer),
        })

    add(
        "1", 1, "C2",
        "To compute the effect of the mechanism angles α_M, β_M, and γ_M on α_t, β_t, and roll before compensation, the required transform",
        "wind frame to body-axis frame",
        CONVERSION_PAIRS,
    )

    role_targets = (("normal", "body_up"), ("axial", "body_forward"), ("lateral", "body_right"))
    for manuscript_id, (role_name, target_name) in zip(("3", "4", "5"), role_targets):
        add(
            manuscript_id, 2, "C1",
            f"the coordinate axis for the positive absolute {role_name} of the wind frame",
            _directed(model.wind_frame, _VECTOR[target_name]),
            DIRECTED,
        )
    for manuscript_id, (role_name, target_name) in zip(("6", "7", "8"), role_targets):
        add(
            manuscript_id, 2, "C1",
            f"the coordinate axis for the positive absolute {role_name} of the airframe",
            _directed(model.body_frame, _VECTOR[target_name]),
            DIRECTED,
        )
    add("9", 2, "A2", "The pre-transform from the wind frame to the body-axis frame at the level pose is RP1", _matrix_text(alignment))
    add("10", 2, "C2", "the frame whose axes RP1 aligns with their own absolute directions", "body-axis frame", CONVERSION_FRAMES)
    add("11", 2, "C2", "the frame whose axes RP1 compares with their own absolute directions", "wind frame", CONVERSION_FRAMES)
    add(
        "12", 2, "C1",
        "The model installation",
        "upright" if model.placement == PLACE_UPRIGHT else "inverted",
        MOUNTS,
    )
    add(
        "13", 2, "A1",
        "The positive absolute normal (upward) and positive absolute lateral (right) of the model relative to the positive absolute normal and positive absolute lateral of the tunnel",
        "same" if model.placement == PLACE_UPRIGHT else "opposite",
        AXIS_RELATIONS,
    )
    add("14", 2, "A2", "the corresponding pre-transform is RP2", _matrix_text(mount))
    add("15", 2, "C2", "the frame whose axis directions RP2 makes the same", "body-axis frame", CONVERSION_FRAMES)
    add("16", 2, "C2", "the frame whose coordinate directions RP2 matches", "wind frame", CONVERSION_FRAMES)

    def add_airspeed_block(step: int, angle: str, mechanism: str, ids: tuple[str, ...]) -> None:
        mechanism_record = model.mechanism_angles[mechanism]
        theory = model.angles[angle]
        _nose_tunnel, airspeed_body = _nose_and_airspeed(model, mechanism_record.positive_airspeed_toward)
        symbol = SYMBOL[mechanism]
        add(
            ids[0], step, "C1",
            f"The model installation used when {symbol} > 0",
            "upright" if model.placement == PLACE_UPRIGHT else "inverted",
            ["upright", "inverted"],
        )
        add(ids[1], step, "A1", f"When {symbol} > 0 the airspeed points this way on the airframe", _role(airspeed_body), DIRECTIONS)
        add(ids[2], step, "C1", f"When the theoretical {ANGLE_ZH[angle]} is positive the airspeed points this way on the airframe", _role(theory.positive_airspeed_toward), DIRECTIONS)
        add(
            ids[3], step, "A1",
            f"When {symbol} > 0 the sign of the theoretical {ANGLE_ZH[angle]}",
            _same_sense(airspeed_body, theory.positive_airspeed_toward),
            SIGNS,
        )
        add(ids[4], step, "C2", f"This is the level pose of the frame for {symbol}", "intermediate body-axis frame", FRAMES)
        add(ids[5], step, "C1", f"The rotation of {symbol} is about this body coordinate axis", mechanism_record.associated_axis, AXES)
        add(ids[6], step, "A1", f"When {symbol} > 0 the sense about that axis", ROTATION_ZH[mechanism_record.rotation_when_positive], SIGNS)
        add(ids[7], step, "A2", f"the rotation matrix is R_{symbol}", _matrix_text(matrices[angle]))

    def add_roll_block(step: int, ids: tuple[str, ...]) -> None:
        mechanism_record = model.mechanism_angles[GAMMA_M]
        theory = model.angles[GAMMA]
        mechanism_hand = _declared_roll_hand(mechanism_record.right_wing_toward)
        theory_hand = _declared_roll_hand(theory.right_wing_toward)
        mechanism_wing = _wing_about_nose(mechanism_hand)
        theory_wing = _wing_about_nose(theory_hand)
        if mechanism_wing != mechanism_record.right_wing_toward or theory_wing != theory.right_wing_toward:
            raise DefinitionError("Right-wing direction inferred from the axial right-hand rule disagrees with the declaration.")
        symbol = SYMBOL[GAMMA_M]
        add(ids[0], step, "C1", f"When {symbol} > 0, about the positive absolute model axis (forward), the relation to the right-hand rule", mechanism_hand, AXIS_RELATIONS)
        add(ids[1], step, "A1", f"When {symbol} > 0 the right wing points this way on the airframe", _role(mechanism_wing), DIRECTIONS)
        add(
            ids[2], step, "C1",
            f"When the theoretical {ANGLE_ZH[GAMMA]} is positive, about the positive absolute airframe axis (forward), the relation to the right-hand rule",
            theory_hand, AXIS_RELATIONS,
        )
        add(ids[3], step, "A1", f"When the theoretical {ANGLE_ZH[GAMMA]} is positive the right wing points this way on the airframe", _role(theory_wing), DIRECTIONS)
        add(
            ids[4], step, "A1",
            f"When {symbol} > 0 the sign of the theoretical {ANGLE_ZH[GAMMA]}",
            _same_sense(mechanism_wing, theory_wing),
            SIGNS,
        )
        add(ids[5], step, "C2", f"This is the level pose of the frame for {symbol}", "intermediate body-axis frame", FRAMES)
        add(ids[6], step, "C1", f"The rotation of {symbol} is about this body coordinate axis", mechanism_record.associated_axis, AXES)
        add(ids[7], step, "A1", f"When {symbol} > 0 the sense about the positive direction of that coordinate axis", ROTATION_ZH[mechanism_record.rotation_when_positive], SIGNS)
        add(ids[8], step, "A2", f"the rotation matrix is R_{symbol}", _matrix_text(matrices[GAMMA]))

    for index, (manuscript_id, angle) in enumerate(zip(("17", "18", "19"), model.euler_order), start=1):
        add(manuscript_id, 3, "A1", f"rotation angle {index} of the order", ANGLE_ZH[angle], EULER)
    add_airspeed_block(4, ALPHA, ALPHA_M, ("20", "21", "22", "23", "24", "25", "26", "27"))
    add_airspeed_block(5, BETA, BETA_M, ("28", "29", "30", "31", "32", "33", "34", "35"))
    add_roll_block(6, ("36", "37", "38", "39", "40", "41", "42", "43", "44"))
    add("45", 7, "C2", "This step changes coordinates, so the transform is", "passive", TRANSFORMS)
    add("46", 7, "C2", "The body-axis frame is", "moving axes", FRAME_KINDS)
    add("47", 7, "C2", "For that kind of frame the rotation uses", "intrinsic", ROTATIONS)
    matrix_code = {ALPHA: "R_α_M", BETA: "R_β_M", GAMMA: "R_γ_M"}
    factor_options = ["RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M"]
    first_code, second_code, third_code = [matrix_code[name] for name in model.euler_order]
    add("48", 7, "A2", "R1 in the product", third_code, factor_options)
    add("49", 7, "A2", "R2 in the product", second_code, factor_options)
    add("50", 7, "A2", "R3 in the product", first_code, factor_options)
    add("51", 7, "A2", "R4 in the product", "RP2", factor_options)
    add("52", 7, "A2", "R5 in the product", "RP1", factor_options)
    add("56", 8, "A2", "the symbolically simplified product R_t = R1*R2*R3", _matrix_text(total))
    add(
        "57", 9, "A1",
        "RP2*RP1 times the wind-frame airspeed unit vector, in the level airframe",
        _vector_text(level_body),
    )
    add("58", 9, "C2", "the frame whose airspeed coordinates are mapped after the attitude change", "level airframe", LEVEL_FRAMES)
    add("59", 10, "A1", "The airspeed unit vector [X_V, Y_V, Z_V] in the level frame", _vector_text(level_body))
    add("60", 10, "A2", "XTZ from R_t*[X_V, Y_V, Z_V]", _text(component_expr["XTZ"]))
    add("61", 10, "A2", "YTZ from R_t*[X_V, Y_V, Z_V]", _text(component_expr["YTZ"]))
    add("62", 10, "A2", "ZTZ from R_t*[X_V, Y_V, Z_V]", _text(component_expr["ZTZ"]))

    def add_component_signs(step: int, mechanism: str, start: int) -> None:
        signs = model.mechanism_angles[mechanism].airspeed_signs
        for offset, axis in enumerate(BODY_AXES):
            add(
                str(start + offset), step, "A1",
                f"When only {SYMBOL[mechanism]} > 0, the sign of {axis}",
                SIGN_ZH[signs[axis]],
                SIGNS,
            )

    add_component_signs(11, ALPHA_M, 63)
    add_component_signs(12, BETA_M, 66)

    inside = "The numerator and the denominator must both be positive. The sign may be written only outside the inverse function"
    add("69", 13, "A1", f"The angle sign of α_t, written outside arctan. {inside}", alpha_outer, SIGNS)
    add("70", 13, "A1", f"The inverse function of α_t. {inside}", "arctan", INVERSES)
    add("71", 13, "A1", f"From the sign of the chosen body-axis component when only α_M > 0, the numerator sign of α_t. {inside}", alpha_num_sign, SIGNS)
    add("72", 13, "A1", f"The body-axis component in the numerator of α_t. {inside}", alpha_num, COMPONENTS)
    add("73", 13, "A1", f"From the sign of the chosen body-axis component when only α_M > 0, the denominator sign of α_t. {inside}", alpha_den_sign, SIGNS)
    add("74", 13, "A1", f"The body-axis component in the denominator of α_t. {inside}", alpha_den, COMPONENTS)
    add("75", 14, "A1", f"The angle sign of β_t, written outside arcsin. {inside}", beta_outer, SIGNS)
    add("76", 14, "A1", f"The inverse function of β_t. {inside}", "arcsin", INVERSES)
    add("77", 14, "A1", f"From the sign of the chosen body-axis component when only β_M > 0, the numerator sign of β_t. {inside}", beta_num_sign, SIGNS)
    add("78", 14, "A1", f"The body-axis component in the numerator of β_t. {inside}", beta_num, COMPONENTS)
    add("79", 14, "A1", f"The denominator sign of β_t. {inside}", "positive", SIGNS)
    add("80", 14, "A1", f"The denominator component of β_t. {inside}", "1", COMPONENTS)
    add("81", 15, "A2", f"α_t after substituting the component expressions. {inside}. A minus sign is written only before the whole arctan", alpha_answer)
    add("82", 15, "A2", f"β_t after substituting the component expressions. {inside}. A minus sign is written only before the whole arcsin", beta_answer)

    _check_component_signs(model, component_expr)
    return {
        "task": "wind_frame_to_body_frame",
        "declaration": spec,
        "fixed": {
            "transform": "passive",
            "rotation_type": INTRINSIC,
            "body_frame_kind": "moving_axes",
            "euler_order": list(model.euler_order),
            "product": "[XTZ, YTZ, ZTZ]^T = R1*R2*R3*R4*R5*[XFD, YFD, ZFD]^T",
        },
        "wind_airspeed": [int(value) for value in wind_components],
        "items": items,
    }


def _check_component_signs(model: BodyFrameModel, component_expr: dict[str, sp.Expr]) -> None:
    """Single-angle finite rotation must carry the same signs as the model projection."""
    samples = {ALPHA_M: {ALPHA_SYM: sp.Rational(1, 10)}, BETA_M: {BETA_SYM: sp.Rational(1, 10)}}
    for mechanism, substitution in samples.items():
        signs = model.mechanism_angles[mechanism].airspeed_signs
        values = {
            axis: sp.simplify(expr.subs({ALPHA_SYM: 0, BETA_SYM: 0, GAMMA_SYM: 0, **substitution}))
            for axis, expr in component_expr.items()
        }
        for axis, sign in signs.items():
            value = values[axis]
            if sign == UNKNOWN and value != 0:
                raise DefinitionError(f"{mechanism} leaves a nonzero {axis} component {value}.")
            if sign == POSITIVE and value <= 0:
                raise DefinitionError(f"{mechanism} expected positive {axis}, got {value}.")
            if sign == NEGATIVE and value >= 0:
                raise DefinitionError(f"{mechanism} expected negative {axis}, got {value}.")


def _render(payload: dict) -> str:
    lines = ["# wind frame to body-axis frame answers", ""]
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
    assert by_id["1"] == "wind frame to body-axis frame"
    assert by_id["2"] == "YFD"
    assert by_id["3"] == "XFD"
    assert by_id["4"] == "ZFD"
    assert by_id["5"] == "YTZ"
    assert by_id["6"] == "XTZ"
    assert by_id["7"] == "ZTZ"
    assert by_id["8"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert by_id["9"] == "body-axis frame"
    assert by_id["10"] == "wind frame"
    assert by_id["11"] == "upright"
    assert by_id["12"] == "same"
    assert by_id["13"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert by_id["14"] == "body-axis frame"
    assert by_id["15"] == "wind frame"
    assert [by_id["16"], by_id["17"], by_id["18"]] == ["angle of attack", "sideslip", "roll"]
    assert by_id["19"] == "upright"
    assert by_id["20"] == "downward"
    assert by_id["21"] == "downward"
    assert by_id["22"] == "positive"
    assert by_id["23"] == "intermediate body-axis frame"
    assert by_id["24"] == "ZTZ"
    assert by_id["25"] == "positive"
    assert by_id["27"] == "upright"
    assert by_id["28"] == "right"
    assert by_id["30"] == "positive"
    assert by_id["31"] == "intermediate body-axis frame"
    assert by_id["32"] == "YTZ"
    assert by_id["33"] == "positive"
    assert by_id["35"] == "same"
    assert next(item["label"] for item in payload["items"] if item["manuscript_id"] == "35") == "C1"
    assert by_id["36"] == "downward"
    assert by_id["37"] == "same"
    assert by_id["38"] == "downward"
    assert by_id["39"] == "positive"
    assert by_id["40"] == "intermediate body-axis frame"
    assert by_id["41"] == "XTZ"
    assert by_id["42"] == "positive"
    assert by_id["44"] == "passive"
    assert by_id["46"] == "intrinsic"
    assert [by_id["47"], by_id["48"], by_id["49"], by_id["50"], by_id["51"]] == [
        "R_γ_M", "R_β_M", "R_α_M", "RP2", "RP1",
    ]
    assert by_id["57"] == "level airframe"
    assert "52" not in by_id and "53" not in by_id and "54" not in by_id
    assert by_id["56"] == "[1, 0, 0]"
    assert by_id["62"] == "positive"
    assert by_id["63"] == "negative"
    assert by_id["64"] == "indeterminate"
    assert by_id["65"] == "positive"
    assert by_id["66"] == "indeterminate"
    assert by_id["67"] == "positive"
    assert by_id["70"] == "negative"
    assert by_id["71"] == "YTZ"
    assert by_id["73"] == "XTZ"
    assert by_id["77"] == "ZTZ"
    assert by_id["79"] == "1"
    assert by_id["80"] == (
        "arctan((sin(α_M)*cos(γ_M) - sin(β_M)*sin(γ_M)*cos(α_M))/(cos(α_M)*cos(β_M)))"
    )
    assert by_id["81"] == "arcsin(sin(α_M)*sin(γ_M) + sin(β_M)*cos(α_M)*cos(γ_M))"
    assert len(payload["items"]) == 78
    labels = {item["label"] for item in payload["items"]}
    assert labels == {"C1", "C2", "A1", "A2"}

    flipped = json.loads(json.dumps(STANDARD_SPEC))
    flipped["beta"] = {"airspeed_toward": "body_left", "sign": "positive"}
    flipped_answers = {item["manuscript_id"]: item["answer"] for item in generate(flipped)["items"]}
    assert flipped_answers["30"] == "negative"
    assert flipped_answers["74"] == "negative"
    assert flipped_answers["76"] == "positive"
    assert flipped_answers["77"] == "ZTZ"
    assert flipped_answers["78"] == "positive"
    assert flipped_answers["81"].startswith("-arcsin(")
    assert "β_M" in flipped_answers["81"]

    inverted = json.loads(json.dumps(STANDARD_SPEC))
    inverted["placement"] = PLACE_INVERTED
    inverted_answers = {item["manuscript_id"]: item["answer"] for item in generate(inverted)["items"]}
    assert inverted_answers["8"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert inverted_answers["9"] == "body-axis frame"
    assert inverted_answers["10"] == "wind frame"
    assert inverted_answers["11"] == "inverted"
    assert inverted_answers["12"] == "opposite"
    assert inverted_answers["13"] == "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]"
    assert inverted_answers["14"] == "body-axis frame"
    assert inverted_answers["15"] == "wind frame"
    assert [inverted_answers["47"], inverted_answers["48"], inverted_answers["49"], inverted_answers["50"], inverted_answers["51"]] == [
        "R_γ_M", "R_β_M", "R_α_M", "RP2", "RP1",
    ]
    assert inverted_answers["23"] == "intermediate body-axis frame"
    assert inverted_answers["24"] == "ZTZ"
    assert inverted_answers["25"] == "positive"
    assert inverted_answers["35"] == "same"
    assert inverted_answers["36"] == "downward"
    assert inverted_answers["19"] == "inverted"
    assert inverted_answers["20"] == "downward"
    assert inverted_answers["27"] == "inverted"
    assert inverted_answers["28"] == "right"

    rearward = json.loads(json.dumps(STANDARD_SPEC))
    rearward["body_frame"] = {
        "chirality": "left",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_rearward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    }
    rearward_answers = {item["manuscript_id"]: item["answer"] for item in generate(rearward)["items"]}
    assert rearward_answers["6"] == "-XTZ"
    assert rearward_answers["35"] == "same"
    assert rearward_answers["36"] == "downward"
    assert rearward_answers["37"] == "same"
    assert rearward_answers["38"] == "downward"
    assert rearward_answers["40"] == "intermediate body-axis frame"
    assert rearward_answers["41"] == "XTZ"
    assert rearward_answers["42"] == "negative"

    wind_swap = json.loads(json.dumps(STANDARD_SPEC))
    wind_swap["wind_frame"] = {
        "chirality": "right",
        "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
        "YFD": {"along": "tunnel_lateral", "points": "tunnel_right"},
    }
    swapped = {item["manuscript_id"]: item["answer"] for item in generate(wind_swap)["items"]}
    assert swapped["2"] == "-ZFD"
    assert swapped["3"] == "XFD"
    assert swapped["4"] == "YFD"
    assert swapped["24"] == "ZTZ"
    assert swapped["32"] == "YTZ"

    opposite_roll = json.loads(json.dumps(STANDARD_SPEC))
    opposite_roll["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    opposite_answers = {item["manuscript_id"]: item["answer"] for item in generate(opposite_roll)["items"]}
    assert opposite_answers["35"] == "opposite"
    assert opposite_answers["36"] == "upward"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the wind-to-body cloze answer key.")
    parser.add_argument("--spec", type=Path, help="JSON declaration. Defaults to the standard upright frame.")
    parser.add_argument("--out", type=Path, default=Path("tz_fd_eval.json"))
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
