# -*- coding: utf-8 -*-
"""Flight-mechanics spatial-transform cloze.

The manuscript is the standard coordinate-conversion flow: ground to body,
ground to the unbanked track frame, airflow roll about the track, body through
stability to airflow, then the airspeed closure. Elementary matrices are the passive right-handed matrices
already used by FlightFrameModel. A parent-from-child attitude is the transpose
of the child-from-parent product, so the factors below are computed, not copied
from a hand formula.

Run from the repository root::

    python -m domains.aerodynamics.flow_generate
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import sympy as sp

from cloze_score import answer_letter, format_choices
from domain_layout import DOMAINS
from domains.aerodynamics.flow_loads import build_loads
from flight_frame_model import (
    ALPHA,
    BETA,
    GAMMA,
    MU,
    NEGATIVE,
    NEGATIVE_ROTATION,
    PHI,
    POSITIVE,
    POSITIVE_ROTATION,
    PSI,
    STANDARD_SPEC,
    THETA,
    VARPHI,
    DefinitionError,
    FlightFrameModel,
)


BLANK_COUNT = 131
OUT_DIR = DOMAINS["aerodynamics"]["flow"]["review"]
EYE = sp.eye(3)
E1 = sp.Matrix([1, 0, 0])

YAW, PITCH, ROLL, AZIMUTH, INCLINE, BANK, ATTACK, SIDESLIP = sp.symbols(
    "yaw pitch roll azimuth incline bank attack sideslip", real=True
)
VX, VY, VZ = sp.symbols("vx vy vz", real=True)
SYMBOL = {
    PSI: YAW,
    THETA: PITCH,
    PHI: ROLL,
    VARPHI: AZIMUTH,
    MU: INCLINE,
    GAMMA: BANK,
    ALPHA: ATTACK,
    BETA: SIDESLIP,
}
NAME = {
    YAW: "ψ",
    PITCH: "θ",
    ROLL: "Φ",
    AZIMUTH: "φ",
    INCLINE: "μ",
    BANK: "γ",
    ATTACK: "α",
    SIDESLIP: "β",
}
SLOT_LINE = re.compile(r"^(\d+)\. .+\{ID:(\d+), .+\}\.$")


def _passive(slot: str, angle) -> sp.Matrix:
    c = sp.cos(angle)
    s = sp.sin(angle)
    if slot == "X":
        return sp.Matrix([[1, 0, 0], [0, c, s], [0, -s, c]])
    if slot == "Y":
        return sp.Matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    if slot == "Z":
        return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    raise DefinitionError(slot)


def _binding(model: FlightFrameModel, origin: str, destination: str, angle: str):
    found = [item for item in model.conversion(origin, destination) if item.angle == angle]
    if len(found) != 1 or not found[0].axis:
        raise DefinitionError(f"{angle} is not bound from {origin} to {destination}.")
    return found[0]


def _argument(model: FlightFrameModel, origin: str, destination: str, angle: str):
    binding = _binding(model, origin, destination, angle)
    symbol = SYMBOL[angle]
    sense = binding.rotation_when_positive
    if sense == POSITIVE_ROTATION:
        return binding, symbol
    if sense == NEGATIVE_ROTATION:
        return binding, -symbol
    raise DefinitionError(f"{angle} has no rotation sense.")


def _product(model: FlightFrameModel, steps: tuple[tuple[str, str, str], ...]) -> tuple[sp.Matrix, str]:
    """Parent-from-child matrix and its factor text.

    ``steps`` run from the first intrinsic rotation to the last. The
    child-from-parent product therefore has the first factor on the right.
    """
    factors = []
    for origin, destination, angle in steps:
        binding, argument = _argument(model, origin, destination, angle)
        slot = binding.axis[0]
        shown = sp.simplify(-argument)
        factors.append(f"C{slot.lower()}({_argument_text(shown, SYMBOL[angle])})")
    text = "*".join(factors)
    matrix = EYE
    for part in text.split("*"):
        matrix *= _parse_factor(part)
    return sp.simplify(matrix), text


def _argument_text(expr, symbol) -> str:
    expr = sp.simplify(expr)
    if expr == symbol:
        return NAME[symbol]
    if expr == -symbol:
        return f"-{NAME[symbol]}"
    raise DefinitionError(f"Cannot show rotation argument {expr}.")


def _parse_factor(text: str) -> sp.Matrix:
    matched = re.fullmatch(r"C([xyz])\((.+)\)", text)
    if not matched:
        raise DefinitionError(f"Bad factor {text}.")
    slot = matched.group(1).upper()
    body = matched.group(2)
    sign = -1 if body.startswith("-") else 1
    name = body[1:] if body.startswith("-") else body
    symbol = next(key for key, value in NAME.items() if value == name)
    return _passive(slot, sign * symbol)


def _flip_factor(text: str) -> str:
    matched = re.fullmatch(r"C([xyz])\((.+)\)", text)
    if not matched:
        raise DefinitionError(f"Bad factor {text}.")
    body = matched.group(2)
    flipped = body[1:] if body.startswith("-") else f"-{body}"
    return f"C{matched.group(1)}({flipped})"


def _transpose_product(text: str) -> str:
    """Factor string of the transpose. Passive C(η)^T equals C(-η)."""
    return "*".join(_flip_factor(part) for part in reversed(text.split("*")))


def _track_wrongs(correct: str, bank_factor: str) -> list[str]:
    parts = correct.split("*")
    candidates = [
        "*".join(_flip_factor(part) for part in parts),
        "*".join(reversed(parts)),
        "*".join(_flip_factor(part) if index == len(parts) - 1 else part for index, part in enumerate(parts)),
        f"{correct}*{bank_factor}",
    ]
    wrongs = []
    for item in candidates:
        if item != correct and item not in wrongs:
            wrongs.append(item)
    if len(wrongs) < 2:
        raise DefinitionError(f"Not enough distinct distractors for {correct}.")
    return wrongs


def _show(expr) -> str:
    text = str(sp.simplify(expr))
    text = text.replace("**", "^")
    for symbol, name in NAME.items():
        text = text.replace(str(symbol), name)
    return text


def _vector(matrix_times_column) -> str:
    cells = ", ".join(_show(item) for item in matrix_times_column)
    return f"[{cells}]"


def _at_degrees(expr, values: dict) -> str:
    symbols = {key: sp.Symbol(f"deg{amount}") for key, amount in values.items()}
    shown = _show(sp.simplify(expr.subs(symbols)))
    for amount in values.values():
        shown = shown.replace(f"deg{amount}", f"{amount}°")
    return shown


def _component_word(expr, values: dict) -> str:
    number = sp.N(expr.subs({key: sp.rad(amount) for key, amount in values.items()}))
    if abs(number) < sp.Float("1e-10"):
        raise DefinitionError("The vertical component is zero, so its direction is undefined.")
    return "downward" if number > 0 else "upward"


def _recovers(component_expr, target, candidates: list) -> sp.Expr:
    for candidate in candidates:
        good = True
        for amount in (-0.6, -0.2, 0.15, 0.55):
            other = 0.25 if target == SIDESLIP else -0.2
            values = {ATTACK: other, SIDESLIP: amount} if target == SIDESLIP else {ATTACK: amount, SIDESLIP: other}
            predicted = candidate.subs({
                VX: component_expr[0].subs(values),
                VY: component_expr[1].subs(values),
                VZ: component_expr[2].subs(values),
            })
            if abs(sp.N(predicted - values[target])) > sp.Float("1e-8"):
                good = False
                break
        if good:
            return candidate
    raise DefinitionError(f"No recovery formula matches {target}.")


def _formula(expr) -> str:
    text = str(expr).replace("**", "^").replace(" ", "")
    return text.replace(",", ", ")


def _sign_word(model: FlightFrameModel, angle: str, words: dict[str, str]) -> str:
    toward = model.binding(angle).positive_toward
    if toward not in words:
        raise DefinitionError(f"{angle} points {toward}, which this question does not name.")
    return words[toward]


def _choose(correct: str, wrongs: list[str], salt: int) -> tuple[list[str], str]:
    options = []
    for item in (correct, *wrongs):
        if item not in options:
            options.append(item)
    if len(options) < 2 or correct not in options:
        raise DefinitionError(f"Choices for {correct} are not usable.")
    shift = salt % len(options)
    return options[shift:] + options[:shift], correct


def _choice(number: int, label: str, stem: str, correct: str, wrongs: list[str]) -> dict:
    options, answer = _choose(correct, wrongs, number)
    return {"id": number, "label": label, "stem": stem, "options": options, "answer": answer, "format": "choice"}


def _open(number: int, label: str, stem: str, answer: str, fmt: str = "vector") -> dict:
    return {"id": number, "label": label, "stem": stem, "options": None, "answer": answer, "format": fmt}


def _load_questions(loads: dict) -> list[dict]:
    added = loads["added"]
    rows = [
        _choice(60, "C2", "The known aerodynamic components are in the airflow frame and the output is required in the airframe, so the transform is", "airflow frame to airframe", ["airframe to airflow frame", "airflow frame to stability frame, then stop"]),
        _choice(61, "C2", "Across this subprocess, the moment reference point", "remains A", ["changes from A to G", "changes from G to A"]),
        _choice(62, "C1", "The positive absolute airframe axis corresponds to", "X_b", ["Y_b", "Z_b"]),
        _choice(63, "C1", "The positive absolute airframe lateral direction corresponds to", "Y_b", ["X_b", "Z_b"]),
        _choice(64, "C1", "The positive absolute airframe normal corresponds to", "Z_b", ["X_b", "Y_b"]),
        _choice(65, "C1", "Drag in the airflow frame is along", "-X_a", ["+X_a", "-Z_a"]),
        _choice(66, "C1", "Positive lift in the airflow frame is along", loads["lift_axis"], [item for item in ("-Z_a", "+Z_a", "+Y_a") if item != loads["lift_axis"]]),
        _choice(67, "C1", "Positive side force in the airflow frame is along", loads["side_axis"], [item for item in ("+Y_a", "-Y_a", "+Z_a") if item != loads["side_axis"]]),
        _open(68, "A2", "In the order X_a, Y_a, Z_a, the airflow-frame aerodynamic force is", loads["air_force"]),
        _choice(69, "C1", "Positive airflow roll moment l_a is about", "+X_a", ["+Y_a", "+Z_a"]),
        _choice(70, "C1", "Positive airflow pitch moment m_a is about", "+Y_a", ["+X_a", "+Z_a"]),
        _open(71, "A2", "In the order X_a, Y_a, Z_a, the airflow-frame moment is", loads["air_moment"]),
        _choice(72, "A1", "With only a positive angle of attack, the stability forward axis relative to the airframe forward axis points", loads["stable_side"], ["upward", "downward", "right"]),
        _choice(73, "A1", "The Z_b component of that forward axis in the airframe is therefore", loads["stable_z"], ["positive", "negative", "zero"]),
        _open(74, "A2", "The passive matrix T_s←b from the airframe to the stability frame is", loads["to_stable"], "matrix"),
        _choice(75, "A1", "With only a positive sideslip, the airflow forward axis relative to the stability forward axis points", loads["air_side"], ["left", "right", "downward"]),
        _choice(76, "A1", "The Y_s component of that forward axis in the stability frame is therefore", loads["air_y"], ["positive", "negative", "zero"]),
        _open(77, "A2", "The passive matrix T_a←s from the stability frame to the airflow frame is", loads["stable_to_air"], "matrix"),
        _choice(78, "C2", "The full transform T_a←b from the airframe to the airflow frame is", "T_a←s*T_s←b", ["T_s←b*T_a←s"]),
        _open(79, "A2", "The expanded T_a←s is", loads["stable_to_air"], "matrix"),
        _open(80, "A2", "The expanded T_s←b is", loads["to_stable"], "matrix"),
        _open(81, "A2", "Expanding that product gives T_a←b =", loads["to_air"], "matrix"),
        _choice(82, "C2", "Convert the airflow-frame force into the airframe force with", "the transpose of T_a←b", ["T_a←b"]),
        _open(83, "A2", "Call that reverse matrix B and expand it to", loads["body_from_air"], "matrix"),
        _open(84, "A2", "The airframe axial force X is", loads["body_x"], "expression"),
        _open(85, "A2", "The airframe lateral force Y_b is", loads["body_y"], "expression"),
        _open(86, "A2", "The airframe normal force Z is", loads["body_z"], "expression"),
        _choice(87, "A1", "The airframe axial component produced by lift is", loads["lift_x"], ["positive", "negative", "zero"]),
        _choice(88, "A1", "The airframe normal component produced by lift is", loads["lift_z"], ["positive", "negative", "zero"]),
        _choice(89, "A1", "When β=0, Y=0, D>0, and L>0, these conditions alone make the sign of the airframe axial force X", loads["axial_sign"], ["certainly positive", "certainly negative", "indeterminate"]),
        _open(90, "A2", "At α=30°, β=0°, D=2, Y=0, and L=2*sqrt(3), the airframe aerodynamic force is", loads["example_force"]),
        _choice(91, "A1", "After the coordinate transform, the Euclidean length of the aerodynamic force is", "same", ["generally increases", "generally decreases"]),
        _choice(92, "C2", "At the same reference point A, the aerodynamic moment is moved from the airflow frame to the airframe with", "the same matrix B", ["the negative of B", "an extra moment-arm translation"]),
        _open(93, "A2", "The airframe roll moment l_A is", loads["body_l"], "expression"),
        _open(94, "A2", "The airframe pitch moment m_A is", loads["body_m"], "expression"),
        _open(95, "A2", "The airframe yaw moment n_A is", loads["body_n"], "expression"),
        _choice(96, "A1", "At β=0, the axis of the positive airflow pitch moment and the airframe lateral axis are", loads["pitch_relation"], ["aligned", "perpendicular", "direction indeterminate"]),
        _choice(97, "A1", "When α=0, 0°<β<90°, l_a>0, and m_a=n_a=0, the airframe pitch moment m_A is", loads["roll_to_pitch"], ["positive", "negative", "zero"]),
        _open(98, "A2", "Dynamic pressure q_bar from ρ and V is", loads["dynamic"], "expression"),
        _open(99, "A2", "The dimensional airflow-frame force from the aerodynamic coefficients is", loads["coefficient_force"]),
        _choice(100, "C2", f"The three force components share one scale, so the signed coefficient vector {loads['coeff_symbol']} can", "convert directly with the same matrix B", ["must be multiplied by different reference lengths before conversion"]),
        _choice(101, "C2", "When the three moment coefficients use reference lengths b, c_bar, and b, and b is not equal to c_bar,", "recover the dimensional moment, rotate it, then normalize by the output-axis reference length", ["multiply B directly by [C_la, C_ma, C_na]"]),
        _open(102, "A2", "Under the current normalization, the airframe pitch-moment coefficient C_mA is", loads["pitch_coefficient"], "expression"),
        _choice(103, "C1", "Before the transfer, the moment reference point is", "A", ["G", "N"]),
        _choice(104, "C1", "After the transfer, the moment reference point is", "G", ["A", "N"]),
        _open(105, "A2", "From the collinear lengths, the distance X_N from A to the nose reference N is", loads["nose_distance"], "expression"),
        _choice(106, "C1", "The positive engineering axial position x_Ge points", "rearward", ["forward"]),
        _choice(107, "A1", "That direction and the airframe +X_b direction are", "opposite", ["same"]),
        _open(108, "A2", "The axial coordinate of the center of mass G in the airframe with origin A is therefore", loads["lever_x"], "expression"),
        _open(109, "A2", "In the order X_b, Y_b, Z_b, the vector r_AG from A to G is", loads["arm_from_a"]),
        _open(110, "A2", "The vector r_GA from G to A is", loads["arm_from_g"]),
        _choice(111, "C2", "After the reference point moves from A to G, the resultant force of the same aerodynamic system", "stays unchanged", ["must be reversed", "must gain a moment-arm term"]),
        _choice(112, "C2", "To transfer the moment about A to the moment about G, the added moment is", "r_GA×F_b", ["r_AG×F_b", "r_GA×M_A"]),
        _choice(113, "A2", "Using the vector r_AG from A to G, the transfer is", "M_G=M_A-r_AG×F_b", ["M_G=M_A+r_AG×F_b", "M_G=F_b-r_AG×M_A"]),
        _choice(114, "A1", "With only a positive axial force X, the added pitch moment at G is", added[("X", "pitch")], ["positive", "negative", "zero"]),
        _choice(115, "A1", "With only a positive axial force X, the added yaw moment at G is", added[("X", "yaw")], ["positive", "negative", "zero"]),
        _choice(116, "A1", "With only a positive lateral force Y_b, the added roll moment at G is", added[("Y_b", "roll")], ["positive", "negative", "zero"]),
        _choice(117, "A1", "With only a positive lateral force Y_b, the added yaw moment at G is", added[("Y_b", "yaw")], ["positive", "negative", "zero"]),
        _choice(118, "A1", "With only a positive normal force Z, the added roll moment at G is", added[("Z", "roll")], ["positive", "negative", "zero"]),
        _choice(119, "A1", "With only a positive normal force Z, the added pitch moment at G is", added[("Z", "pitch")], ["positive", "negative", "zero"]),
        _open(120, "A2", "The roll moment l_G transferred to the center of mass is", loads["roll_g"], "expression"),
        _open(121, "A2", "The pitch moment m_G transferred to the center of mass is", loads["pitch_g"], "expression"),
        _open(122, "A2", "The yaw moment n_G transferred to the center of mass is", loads["yaw_g"], "expression"),
        _open(123, "A2", "After substituting the engineering position, the pitch moment m_G at the center of mass is", loads["pitch_g_engineering"], "expression"),
        _choice(124, "A1", "If G is offset from A only forward on the airframe, so x>0 and y=z=0, and the force is only upward on the airframe, the added pitch moment is", loads["nose_up"], ["nose-up", "nose-down", "zero"]),
        _choice(125, "A1", "If r_AG is parallel to F_b, the moment added by the reference-point transfer is", "zero", ["certainly positive", "certainly negative"]),
        _choice(126, "A1", "If G moves from A only along the airframe axis, the roll moments l_G and l_A are", "same", ["necessarily opposite", "no relation can be formed"]),
        _open(127, "A2", "If every reference length, reference area, and dynamic pressure stays fixed, and [X, Y_b, Z]=q_bar*S*[C_X, C_Yb, C_Z], the roll-moment coefficient C_lG at the center of mass is", loads["roll_coefficient"], "expression"),
        _open(128, "A2", "Under the same convention, the pitch-moment coefficient C_mG at the center of mass is", loads["pitch_coefficient_g"], "expression"),
        _open(129, "A2", "Under the same convention, the yaw-moment coefficient C_nG at the center of mass is", loads["yaw_coefficient_g"], "expression"),
        _choice(130, "C2", "If one aerodynamic system is both rotated from the airflow frame into the airframe and transferred from A to G, the reference-point transfer may be done first in the airflow frame, but the same moment arm must also be expressed in", "the airflow frame", ["the airframe without converting it", "an arbitrary frame"]),
        _choice(131, "A2", "In this right-handed orthogonal frame, if the force, moment, and arm agree on both paths, the final airframe moment at the center of mass from rotating first and from transferring the point first is", "same", ["negatives of each other", "generally unrelated"]),
    ]
    return rows


def build(spec: dict, lift_z: int = -1, side_y: int = 1) -> dict:
    model = FlightFrameModel().declare(spec)
    for parent, child in (
        ("ground", "body"),
        ("ground", "track"),
        ("body", "stability"),
        ("stability", "airflow"),
    ):
        if model.alignment(parent, child) != EYE:
            raise DefinitionError(f"{parent} and {child} are not aligned at zero angles.")

    ground_body, ground_body_text = _product(model, (("ground", "body", PSI), ("ground", "body", THETA), ("ground", "body", PHI)))
    body_from_ground = _transpose_product(ground_body_text)
    if sp.simplify(_parse_product(body_from_ground) - ground_body.T) != sp.zeros(3):
        raise DefinitionError("bRg is not the transpose of gRb.")
    if "ψ" not in body_from_ground.split("*")[-1]:
        raise DefinitionError("The yaw factor is not the rightmost factor of bRg.")
    ground_track, ground_track_text = _product(model, (("ground", "track", VARPHI), ("ground", "track", MU)))
    track_air, track_air_text = _product(model, (("track", "airflow", GAMMA),))
    if sp.simplify(track_air * E1 - E1) != sp.zeros(3, 1):
        raise DefinitionError("Track roll changes the airspeed axis.")
    body_stability, body_stability_text = _product(model, (("body", "stability", ALPHA),))
    stability_air, stability_air_text = _product(model, (("stability", "airflow", BETA),))
    body_air = sp.simplify(body_stability * stability_air)
    body_air_text = f"{body_stability_text}*{stability_air_text}"
    if sp.simplify(body_air - _parse_product(body_air_text)) != sp.zeros(3):
        raise DefinitionError("The airflow product does not match its factors.")

    nose = sp.simplify(ground_body * E1)
    track_speed = sp.simplify(ground_track * E1)
    body_speed = sp.simplify(body_air * E1)
    beta_formula = _recovers(body_speed, SIDESLIP, [
        sp.atan2(VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(-VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(VZ, sp.sqrt(VX**2 + VY**2)),
        sp.atan2(-VZ, sp.sqrt(VX**2 + VY**2)),
    ])
    alpha_formula = _recovers(body_speed, ATTACK, [
        sp.atan2(VZ, VX),
        sp.atan2(-VZ, VX),
        sp.atan2(VY, VX),
        sp.atan2(-VY, VX),
    ])
    nose_example = _vector(nose.subs({YAW: sp.Symbol("deg30"), PITCH: sp.Symbol("deg20"), ROLL: sp.Symbol("deg15")})).replace("deg30", "30°").replace("deg20", "20°").replace("deg15", "15°")
    track_z = _at_degrees(track_speed[2], {AZIMUTH: 40, INCLINE: 10})
    nose_word = _component_word(nose[2], {YAW: 30, PITCH: 20, ROLL: 15})
    speed_word = _component_word(track_speed[2], {AZIMUTH: 40, INCLINE: 10, BANK: 0})
    zero_incline = sp.simplify(track_speed[2].subs({INCLINE: 0, BANK: 0}))
    if zero_incline != 0:
        raise DefinitionError(f"Zero inclination still leaves a normal component {zero_incline}.")
    positive_beta = sp.N(body_speed[1].subs({ATTACK: 0.2, SIDESLIP: 0.3}))
    positive_alpha = sp.N(body_speed[2].subs({ATTACK: 0.3, SIDESLIP: 0.1}))
    _check_closure(ground_body, body_air, ground_track, track_air)
    _check_rotation(ground_body)
    _check_rotation(ground_track)
    _check_rotation(body_air)
    loads = build_loads(body_speed, ATTACK, SIDESLIP, _show, lift_z=lift_z, side_y=side_y)

    yaw_side = _sign_word(model, PSI, {"ground_right": "right on the ground", "ground_left": "left on the ground"})
    pitch_side = _sign_word(model, THETA, {"ground_up": "upward", "ground_down": "downward"})
    incline_side = _sign_word(model, MU, {"ground_up": "upward", "ground_down": "downward"})
    attack_side = _sign_word(model, ALPHA, {"body_down": "downward", "body_up": "upward"})
    sideslip_side = _sign_word(model, BETA, {"body_right": "right", "body_left": "left"})
    angles = {name: model.binding(name).definition for name in SYMBOL}
    if model.binding(PHI).sign == POSITIVE:
        angles[PHI] += ". In forward-right-down coordinates, the right wing down is positive"

    questions = _questions(
        ground_body_text=ground_body_text,
        body_from_ground=body_from_ground,
        ground_track_text=ground_track_text,
        track_air_text=track_air_text,
        body_stability_text=body_stability_text,
        stability_air_text=stability_air_text,
        body_air_text=body_air_text,
        nose_example=nose_example,
        nose_word=nose_word,
        track_speed=_vector(track_speed),
        track_z=track_z,
        speed_word=speed_word,
        body_speed=_vector(body_speed),
        beta_text=_formula(beta_formula),
        alpha_text=_formula(alpha_formula),
        beta_sign="positive" if positive_beta > 0 else "negative",
        alpha_sign="positive" if positive_alpha > 0 else "negative",
        yaw_side=yaw_side,
        pitch_side=pitch_side,
        incline_side=incline_side,
        attack_side=attack_side,
        sideslip_side=sideslip_side,
    )
    questions.extend(_load_questions(loads))
    if [item["id"] for item in questions] != list(range(1, BLANK_COUNT + 1)):
        raise DefinitionError("Question numbers are not 1 through 131.")
    return {
        "model": model,
        "angles": angles,
        "ground_body": ground_body_text,
        "body_from_ground": body_from_ground,
        "ground_track": ground_track_text,
        "track_air": track_air_text,
        "body_stability": body_stability_text,
        "stability_air": stability_air_text,
        "body_air": body_air_text,
        "track_speed": _vector(track_speed),
        "body_speed": _vector(body_speed),
        "lift_axis": loads["lift_axis"],
        "side_axis": loads["side_axis"],
        "questions": questions,
    }


def _parse_product(text: str) -> sp.Matrix:
    matrix = EYE
    for part in text.split("*"):
        matrix *= _parse_factor(part)
    return matrix


def _check_rotation(matrix: sp.Matrix) -> None:
    if sp.simplify(matrix.T * matrix - EYE) != sp.zeros(3):
        raise DefinitionError("A pose matrix is not orthogonal.")
    if sp.simplify(matrix.det() - 1) != 0:
        raise DefinitionError("A pose matrix does not have determinant +1.")


def _check_closure(ground_body: sp.Matrix, body_air: sp.Matrix, ground_track: sp.Matrix, track_air: sp.Matrix) -> None:
    """The angles describe one state: airspeed axes meet, and one bank matches the frames."""
    values = {YAW: 0.4, PITCH: -0.25, ROLL: 0.15, ATTACK: 0.3, SIDESLIP: -0.2}
    target = sp.N((ground_body * body_air).subs(values))
    speed = sp.simplify(ground_track * E1)
    equations = [speed[index] - target[index, 0] for index in (0, 1)]
    for guess in ((0.2, -0.1), (0.2, 0.4), (-0.3, 0.5), (1.0, -0.6), (-1.0, 0.6)):
        try:
            solution = sp.nsolve(equations, [AZIMUTH, INCLINE], guess)
        except (ValueError, TypeError, AttributeError):
            continue
        pose = sp.N(ground_track.subs({AZIMUTH: solution[0], INCLINE: solution[1]}))
        predicted = pose * E1
        if max(abs(predicted[index] - target[index, 0]) for index in range(3)) > sp.Float("1e-8"):
            continue
        relative = pose.T * target
        bank = sp.atan2(relative[2, 1], relative[1, 1])
        for signed in (bank, -bank):
            combined = sp.N((ground_track * track_air).subs({
                AZIMUTH: solution[0], INCLINE: solution[1], BANK: signed,
            }))
            if max(abs(combined[index, column] - target[index, column]) for index in range(3) for column in range(3)) <= sp.Float("1e-8"):
                return
    raise DefinitionError("The track path and the body path do not describe the same airflow attitude.")


def _questions(**text) -> list[dict]:
    g = text["ground_body_text"]
    p = text["ground_track_text"]
    bs = text["body_stability_text"]
    sa = text["stability_air_text"]
    ba = text["body_air_text"]
    rows = [
        _choice(1, "C1", "The positive normal of the ground frame, the airframe, and the track frame points", "downward", ["upward"]),
        _choice(2, "C1", "The positive lateral direction of the ground frame, the airframe, and the track frame points", "right", ["left"]),
        _choice(3, "C1", "Forward, right, and down on the airframe form", "right-handed", ["left-handed"]),
        _choice(4, "C1", "When yaw is positive, the horizontal projection of the airframe forward axis turns", text["yaw_side"], ["right on the ground", "left on the ground"]),
        _choice(5, "C1", "When pitch is positive, the nose points", text["pitch_side"], ["upward", "downward"]),
        _choice(6, "C1", "Roll Φ is taken about the airframe", "axial", ["lateral", "normal"]),
        _choice(7, "C2", "Coordinate transforms in this item use", "passive", ["active"]),
        _choice(8, "C2", "Attitude angles use", "intrinsic moving axes", ["extrinsic fixed axes"]),
        _choice(9, "C2", f"After the yaw-pitch-roll attitude, bRg={text['body_from_ground']} maps ground-frame components into the airframe, and the yaw matrix is", "the rightmost factor", ["the leftmost factor"]),
        _choice(10, "C2", "gRb converts airframe coordinates into the ground frame, and its mapping is", "airframe to ground frame", ["ground frame to airframe"]),
        _choice(11, "C2", "A ground-frame vector is converted into the airframe by using gRb's", "transpose", ["itself"]),
        _choice(12, "C2", "By the declaration, the factored form of gRb is", g, ["Cx(Φ)*Cy(θ)*Cz(ψ)", "Cz(ψ)*Cy(θ)*Cx(Φ)", "Cy(θ)*Cz(ψ)*Cx(Φ)"]),
        _open(13, "C1", "The airframe coordinates of the airframe forward unit vector are", "[1, 0, 0]"),
        _choice(14, "C2", "Map the airframe forward axis into the ground frame by computing", "gRb*[1, 0, 0]", ["gRb^T*[1, 0, 0]"]),
        _open(15, "A2", "At ψ=30°, θ=20°, and Φ=15°, the ground-frame components of the airframe forward axis are", text["nose_example"]),
        _choice(16, "A1", "From the sign of that ground normal component, the nose points", text["nose_word"], ["upward", "downward"]),
        _choice(17, "C2", "In the full attitude matrix, the airframe right axis in the ground frame is column", "the second column", ["the first column", "the third column"]),
        _choice(18, "C2", "The airframe down axis in the ground frame is column", "the third column", ["the first column", "the second column"]),
        _choice(19, "C1", "The transpose of an orthogonal attitude matrix equals its", "inverse", ["negative of the matrix", "determinant"]),
        _choice(20, "C1", "The determinant of any valid three-dimensional rotation matrix is", "+1", ["-1", "0"]),
        _choice(21, "C1", "Track azimuth φ describes the horizontal airspeed projection relative to", "Xg", ["Yg", "Zg"]),
        _choice(22, "C1", "Positive track inclination means the aircraft moves", text["incline_side"], ["upward", "downward"]),
        _choice(23, "C1", "Track roll γ rotates the airflow frame about the shared airspeed axis of the track frame, which is", "Xp", ["Yp", "Zp"]),
        _choice(24, "C1", "The track attitude seen from the ground frame is", "gRp", ["pRg"]),
        _choice(25, "C2", "By the current declaration, the factored form of gRp without roll is", p, _track_wrongs(p, text["track_air_text"])),
        _open(26, "C1", "The track-frame unit airspeed vector is", "[1, 0, 0]"),
        _choice(27, "C2", "Convert the track-frame unit airspeed into the ground frame with", "gRp", ["gRp^T"]),
        _choice(28, "A2", "The ground normal component of the airspeed is", text["track_z"], [text["track_z"].replace("-", "") if text["track_z"].startswith("-") else f"-{text['track_z']}", "cos(10°)", "sin(10°)"]),
        _choice(29, "A1", "From the sign of that ground normal component, the airspeed points", text["speed_word"], ["upward", "downward"]),
        _choice(30, "A2", "When track inclination is zero, the ground normal component of the airspeed is", "0", ["+1", "-1"]),
        _choice(31, "A1", "At μ=0, changing track azimuth makes the horizontal airspeed projection", "turns", ["stays unchanged"]),
        _choice(32, "A1", "When track roll γ changes, the airspeed direction itself", "does not change", ["changes"]),
        _choice(33, "A1", "Track roll changes", "the directions of airflow Ya and Za relative to the track frame", ["the track axes Yp and Zp", "the airspeed magnitude"]),
        _open(34, "A2", "The ground-frame airspeed written from the track direction is", text["track_speed"]),
        _choice(35, "C1", "The stability axis Xs follows the airspeed projected onto the aircraft", "projection in the plane of symmetry", ["spanwise wing projection"]),
        _choice(36, "C1", "When angle of attack α is positive, the airspeed in the plane of symmetry points", text["attack_side"], ["upward", "downward"]),
        _choice(37, "C1", "When sideslip β is positive, the airspeed points", text["sideslip_side"], ["left", "right"]),
        _choice(38, "C2", "The attitude difference from the airframe to the stability frame is only", "angle of attack", ["sideslip", "track azimuth"]),
        _choice(39, "C2", "The attitude difference from the stability frame to the airflow frame is only", "sideslip", ["angle of attack", "yaw"]),
        _choice(40, "C2", "By the declaration, the matrix from the stability frame to the airframe is", bs, ["Cy(-α)", "Cy(α)", "Cz(-β)"]),
        _choice(41, "C2", "The matrix from the airflow frame to the stability frame is", sa, ["Cz(-β)", "Cz(β)", "Cy(-α)"]),
        _choice(42, "C2", "The composite matrix from the airflow frame to the airframe is", ba, ["Cy(-α)*Cz(-β)", "Cz(-β)*Cy(-α)", "Cy(α)*Cz(β)"]),
        _open(43, "C1", "The unit airspeed in the airflow frame is", "[1, 0, 0]"),
        _choice(44, "C2", "Transform the airspeed from the airflow frame into the airframe with", "bRa", ["bRa^T"]),
        _open(45, "A2", "The unit airspeed in the airframe is", text["body_speed"]),
        _choice(46, "A1", "When the airframe airspeed component yb is positive, the airspeed points", "right", ["left"]),
        _choice(47, "A1", "When the airframe airspeed component zb is positive, the airspeed points", "downward", ["upward"]),
        _choice(48, "A2", "Recover sideslip from the airframe unit airspeed [vx, vy, vz] with", text["beta_text"], [
            "atan2(vy, sqrt(vx^2+vz^2))",
            "atan2(-vy, sqrt(vx^2+vz^2))",
            "atan2(vz, sqrt(vx^2+vy^2))",
            "atan2(vx, vy)",
        ]),
        _choice(49, "A1", "When vx>0 and vy>0, the sign of sideslip is", text["beta_sign"], ["positive", "negative"]),
        _choice(50, "A1", "When vx>0 and vz>0, the sign of angle of attack is", text["alpha_sign"], ["positive", "negative"]),
        _choice(51, "A2", "A stable expression for angle of attack is", text["alpha_text"], ["atan2(vz, vx)", "atan2(-vz, vx)", "atan2(vy, vx)"]),
        _choice(52, "C1", "Both gRb and gRp map components of their child frames into", "the ground frame", ["the airflow frame"]),
        _choice(53, "C2", "After the track angles give the ground-frame airspeed, convert it into the airframe by", "gRb^T*vg", ["gRb*vg"]),
        _choice(54, "C2", "When angle of attack and sideslip give the airframe airspeed, γ and Φ", "do not enter the airframe airspeed formula directly", ["must be included"]),
        _open(55, "A2", "The ground-frame airspeed built from the track angles is", text["track_speed"]),
        _choice(56, "A1", "After converting that airspeed into the airframe, it should match the airframe airspeed built directly from α and β", "same", ["opposite", "generally unrelated"]),
        _choice(57, "A2", "When the angles describe one flight state and rounding is ignored, disagreement between the two paths means that the calculation or the use of the convention has", "at least one error", ["certainly no error"]),
        _choice(58, "C2", "Because γ does not change the ground coordinates of the unit airspeed, comparing airspeed vectors checks the shared airspeed axis and not the roll about it, so the airspeed direction can be checked by choosing to", "compare only the airspeed vectors", ["ignore the airspeed axis", "compare only the roll angle"]),
        _choice(59, "A2", "Closing the unit vectors from the two paths requires the norm of their difference to be", "0", ["1", "2"]),
    ]
    return rows


def _slot(item: dict) -> str:
    if item["options"] is None:
        choices = "none"
        answer = item["answer"]
    else:
        choices = format_choices(item["options"])
        answer = answer_letter(item["answer"], item["options"])
        if answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise DefinitionError(f"ID {item['id']} answer is not a letter.")
        chosen = item["options"][ord(answer) - ord("A")]
        if chosen != item["answer"]:
            raise DefinitionError(f"ID {item['id']} letter {answer} points at {chosen}, not {item['answer']}.")
    return f"{{ID:{item['id']}, options:{choices}, format:{item['format']}, label:{item['label']}, answer:{answer}}}"


def _line(item: dict) -> str:
    line = f"{item['id']}. {item['stem']}{_slot(item)}."
    matched = SLOT_LINE.fullmatch(line)
    if not matched or matched.group(1) != matched.group(2) or line.count(".") != 2 or line.count("{ID:") != 1:
        raise DefinitionError(f"ID {item['id']} is not one sentence with one slot.")
    return line


def _load_definition(angles: dict, lift_axis: str = "-Z_a", side_axis: str = "+Y_a") -> list[str]:
    return [
        "## Aerodynamic loads and the reference point",
        "",
        "This section is sent only to the aerodynamic-load subprocess and the reference-point subprocess.",
        "",
        "Fixed conventions",
        "- Vectors are column vectors, and a matrix multiplies a column vector on the left.",
        "- T_a←b maps the airframe components of one physical vector to airflow-frame components. The reverse transform is its transpose.",
        "- T_s←b goes from the airframe to the stability frame, and T_a←s goes from the stability frame to the airflow frame. The declaration does not give the entries of these two matrices.",
        "- All three frames are right-handed. Components are written in the order X, Y, Z.",
        "- A force is an aerodynamic force acting on the aircraft. Aerodynamic forces and moments are signed components.",
        "- Changing the coordinate expression of one point, and changing the moment reference point, are two different operations. Force and moment transforms both go from the airflow frame to the airframe, and the reference point stays at A.",
        "- An angle enters a trigonometric function in one angle unit. This manuscript uses -90°<α<90° and -90°<β<90°, so cos(α)>0 and cos(β)>0.",
        "",
        "Directions and axes",
        "- Forward, right, and down are the positive axial, lateral, and normal directions. Forward × right = down.",
        "- X_b points forward on the airframe, Y_b points right on the airframe, and Z_b points down on the airframe. The airframe origin is the aerodynamic reference point A.",
        "- X_s is the projection of the airspeed onto the airframe plane of symmetry, Y_s has the same sense as Y_b, and Z_s is fixed by X_s×Y_s. The origin coincides with A.",
        "- X_a follows the velocity of the aircraft relative to the air. Z_a lies in the airframe plane of symmetry, is perpendicular to X_a, and is the branch whose dot product with Z_b is positive. Y_a is fixed by Z_a×X_a. The origin coincides with A.",
        "- Airspeed is the velocity of the aircraft relative to the air. The direction of the air relative to the aircraft is the opposite direction.",
        "",
        "Airflow angles",
        f"- Angle of attack α: {angles[ALPHA]}.",
        f"- Sideslip β: {angles[BETA]}.",
        "- Set β=0 when only angle of attack is analyzed. Set α=0 when only sideslip is analyzed.",
        "",
        "Aerodynamic force and moment",
        f"- D is the drag magnitude, D≥0, along -X_a. Y is the signed side force, positive along {side_axis}. L is the signed lift, positive along {lift_axis}.",
        "- l_a, m_a, and n_a are positive by the right-hand rule about +X_a, +Y_a, and +Z_a.",
        "- X, Y_b, and Z are positive along +X_b, +Y_b, and +Z_b. Y is the airflow side force, and Y_b is the airframe lateral force.",
        "- l_A, m_A, and n_A are moments about A, positive by the right-hand rule about +X_b, +Y_b, and +Z_b.",
        "- In the forward-right-down airframe, a positive pitch moment about +Y_b is nose-up, and a negative value is nose-down.",
        "",
        "Dynamic pressure and coefficients",
        "- ρ>0, V>0, and the dynamic pressure is q_bar=ρ*V^2/2. S>0 is the reference area, b>0 is the reference span, and c_bar>0 is the reference mean aerodynamic chord.",
        "- D=q_bar*S*C_D, Y=q_bar*S*C_Ya, and L=q_bar*S*C_L.",
        "- l_a=q_bar*S*b*C_la, m_a=q_bar*S*c_bar*C_ma, and n_a=q_bar*S*b*C_na.",
        "- l_A=q_bar*S*b*C_lA, m_A=q_bar*S*c_bar*C_mA, and n_A=q_bar*S*b*C_nA.",
        "",
        "Reference point",
        "- A is the moment reference point of the original aerodynamic data. G is the center of mass and the moment reference point after the transfer. The airframe axes do not change during the transfer.",
        "- N is the nose reference. N, A, and a tail reference are collinear on the airframe axis. L_f is the distance from N to the tail reference, L_A is the distance from A to the tail reference, and 0<L_A<L_f.",
        "- X_N is the distance from A to N and is computed in the process.",
        "- x_Ge is the axial projection from N rearward to G, and rearward is positive. y_Ge and z_Ge are measured from A along +Y_b and +Z_b.",
        "- x_Ge is opposite +X_b. y_Ge and z_Ge have the same sense as the corresponding airframe components. The declaration does not give the coordinates of G relative to A, and it does not give the reference-point transfer formula.",
        "",
    ]
def render(sample: dict, record: dict) -> str:
    angles = record["angles"]
    questions = {item["id"]: item for item in record["questions"]}
    blocks = [
        ("1", "ground frame to airframe", range(1, 21)),
        ("2", "ground frame to track frame", range(21, 35)),
        ("3", "airframe, stability frame, and airflow frame", range(35, 52)),
        ("4", "ground-frame and airframe closure", range(52, 60)),
        ("5", "aerodynamic force and moment from the airflow frame to the airframe", range(60, 103)),
        ("6", "moment transfer from the aerodynamic reference point to the center of mass", range(103, 132)),
    ]
    before = {
        28: [
            "Compute the ground-frame airspeed at φ=40° and μ=10°.",
        ],
        52: [
            "vg is the ground-frame unit airspeed from the track azimuth and the track inclination.",
            "gRb and gRp map airframe and track-frame components into the ground frame. bRa maps airflow-frame components into the airframe, and pRa maps airflow-frame components into the track frame.",
            "gRp contains no track roll. pRa contains only the roll of the airflow frame relative to the track frame about their shared airspeed axis.",
            "The attitude angles, track angles, angle of attack, and sideslip in this subprocess describe one flight state, and numerical rounding is ignored. These angles cannot be chosen independently.",
            "In that state, the airframe unit airspeed mapped into the ground frame by gRb equals the track-frame unit airspeed mapped into the ground frame by gRp.",
            "The full attitude also satisfies gRb*bRa=gRp*pRa. Equality of the airspeed vectors checks only the shared airspeed axis, not the roll about that axis.",
        ],
        79: [
            "Before the product is formed, write the expansion of each factor in the symbols just chosen.",
        ],
        85: [
            "Keep only lift. Set β=0, Y=0, D=0, L>0, and 0°<α<90°.",
        ],
        101: [
            "This subprocess gives the resultant F_b=[X, Y_b, Z] in the airframe and the moment M_A=[l_A, m_A, n_A] about A.",
            "Every component is already in the same airframe. This subprocess changes only the moment reference point and does not change the axis directions.",
        ],
        112: [
            "Below, r_AG is written [x, y, z].",
            "This part checks only the direction of the added moment. Set x>0, y>0, and z>0, and keep one positive force component at a time. Every force component that is not named is zero.",
        ],
        118: [
            "Restore every signed force component. Do not assume that they are positive.",
        ],
    }
    body = []
    for key, heading, numbers in blocks:
        if key == "5":
            body.extend(_load_definition(angles, record["lift_axis"], record["side_axis"]))
        body.append(f"### Subprocess {key}: {heading}")
        body.append("")
        for number in numbers:
            if number in before:
                body.extend(before[number])
                body.append("")
            body.append(_line(questions[number]))
        body.append("")
    return "\n".join([
        f"# Data {sample['id']} {sample['title']}",
        "",
        "This is a flight-mechanics review manuscript. Blanks are numbered 1 through 131. The first four subprocesses convert coordinates among the ground, track, airframe, and airflow. The last two convert the six aerodynamic components and transfer the moment reference point. The load subprocess and the reference-point subprocess are injected separately: the reference-point subprocess states the airframe force and moment directly and does not use the answers of the load subprocess. This manuscript does not judge the flight envelope and does not apply an error correction.",
        "",
        f"Definition of this item: {sample['change']}",
        f"Single-definition perturbation: {sample['factor']}",
        "",
        "Review notes, labels, answers, and generation notes are not sent to the model. The output-format rules are sent. The first four subprocesses use the declaration and the background above. The last two use the aerodynamic-load and reference-point declaration and do not carry the text of the first four. The model also receives the current subprocess name and the text from the start of that subprocess through the period of the current sentence. Blanks before the current sentence are filled with the answer text. Each sentence has one open blank.",
        "",
        "## Output format",
        "",
        "- choice: multiple choice. Write only the option letter. Example: A",
        "- vector: three components on one line, as [a, b, c]. Put exactly one space after each comma. The order is X, Y, Z. Example: [1, 0, 0]",
        "- matrix: one 3×3 matrix, as [[a, b, c], [d, e, f], [g, h, i]]. Put exactly one space after each comma. Example: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "- expression: one line. Use * for multiplication. Example: a*cos(α)+b",
        "",
        "## Declaration",
        "",
        "Fixed conventions",
        "- Coordinate transforms are passive. Attitude angles, track angles, and airflow angles use an intrinsic moving-axis sequence.",
        "- Vectors are column vectors, and a transform matrix multiplies a column vector on the left. Component transforms act from right to left.",
        "- gRb is the attitude of the airframe relative to the ground frame and maps airframe components into the ground frame. The reverse transform is its transpose.",
        f"- For the moving-axis attitude yaw, pitch, roll, the component transform from the ground frame to the airframe is bRg={record['body_from_ground']}, and the reverse transform is its transpose. Every other transform takes its product order from the stated start frame, target frame, and intermediate frame.",
        "- Components are written in the X, Y, Z order of each frame.",
        "- Take -90°<α, β, θ, μ<90°, so the attack projection, the horizontal track projection, and the attitude branch used here do not meet a degeneracy that this manuscript does not handle. Azimuth and roll use the principal interval [-180°, 180°).",
        "",
        "Mathematical rules",
        "- The transpose of an orthogonal matrix equals its inverse.",
        "- The determinant of any valid three-dimensional rotation matrix is +1.",
        "",
        "Absolute directions and chirality",
        "- The ground frame, airframe, stability frame, track frame, and airflow frame are right-handed.",
        "- The positive axial direction is forward, the positive lateral direction is right, and the positive normal direction is down. Forward × right = down.",
        "- Axial, lateral, and normal are physical directions, not fixed coordinate-axis names. The local positive direction of each frame axis does not change with the attitude angles.",
        "",
        "Frames",
        "- Ground frame {g}: Xg is forward in the horizontal plane, Yg is right in the horizontal plane, and Zg is vertically down. The ground frame is the fixed reference.",
        "- Airframe {b}: xb follows the nose, yb points right on the airframe, and zb points down on the airframe. The airframe is fixed to the aircraft. The plane of symmetry is the xb-zb plane.",
        "- Stability frame {s}: Xs is the forward projection of the airspeed onto the airframe plane of symmetry, Ys is perpendicular to that plane and points right, and Zs lies in the plane, is perpendicular to Xs, and points down.",
        "- Track frame {p}: Xp follows the airspeed, Zp lies in the vertical plane through the airspeed, is perpendicular to the airspeed, and takes the downward branch, and Yp=Zp×Xp. The track frame has no roll. The track here uses the airspeed direction, not the ground speed.",
        "- Airflow frame {a}: Xa follows the airspeed. Za lies in the airframe plane of symmetry, is perpendicular to Xa, and is the branch whose dot product with Zb is positive. Ya=Za×Xa.",
        "- The airflow frame rolls relative to the track frame about their shared airspeed axis. γ follows the right-hand rule about +Xp, and the sign is then the sign in this angle declaration.",
        "",
        "Angles",
        f"- Yaw ψ: {angles[PSI]}.",
        f"- Pitch θ: {angles[THETA]}.",
        f"- Roll Φ: {angles[PHI]}.",
        f"- Track azimuth φ: {angles[VARPHI]}.",
        f"- Track inclination μ: {angles[MU]}.",
        f"- Track roll γ: {angles[GAMMA]}. It is the roll of the airflow frame relative to the track frame and does not change the track frame itself.",
        f"- Angle of attack α: {angles[ALPHA]}.",
        f"- Sideslip β: {angles[BETA]}.",
        "",
        "Elementary rotation matrices",
        "Let cη=cos(η) and sη=sin(η). The following matrices are passive transforms about the current coordinate axis. A positive angle follows the right-hand rule about that axis.",
        "- Cx(η)=[[1, 0, 0], [0, cη, sη], [0, -sη, cη]], about the axial axis X.",
        "- Cy(η)=[[cη, 0, -sη], [0, 1, 0], [sη, 0, cη]], about the lateral axis Y.",
        "- Cz(η)=[[cη, sη, 0], [-sη, cη, 0], [0, 0, 1]], about the normal axis Z.",
        "",
        "Airspeed direction",
        "- Airspeed is the velocity of the aircraft relative to the air, with magnitude V>0. This manuscript does not use ground speed and does not introduce a wind.",
        "- Track Xp and airflow Xa both follow the airspeed, so the unit airspeed in both frames is [1, 0, 0].",
        "",
        "## Background",
        "",
        "Ground Zg points down. A positive ground normal component points down, and a negative component points up. Each angle is taken about the absolute axis named in the declaration, and its positive sense is the sense in the declaration. Coordinate transforms are passive. Attitude, track, and airflow angles use an intrinsic moving-axis sequence.",
        "",
        "## Full process",
        "",
        *body,
        "# Generation notes",
        "",
        "This section is not sent to the model. The elementary rotations are passive. gRb maps airframe components into the ground frame and equals the transpose of bRg. For this item gRb="
        + record["ground_body"]
        + ", bRg="
        + record["body_from_ground"]
        + ". The track frame has no roll, gRp="
        + record["ground_track"]
        + ", pRa="
        + record["track_air"]
        + ", and gRa=gRp*pRa. bRs="
        + record["body_stability"]
        + ", bRa="
        + record["body_air"]
        + ". After an angle sign changes, these factors, the airspeed components, and the aerodynamic matrices are recomputed together. The declaration does not write gRb, gRp, bRs, bRa, or the unit-airspeed components, so a derived result cannot be copied from it.",
        "",
        "The aerodynamic loads use one orthogonal basis. T_i←j maps components of frame j into frame i, and its rows are the axes of frame i written in frame j. The declaration does not write the entries of T_s←b or T_a←s. At the standard angle of attack, the side-force term in the airframe normal force is -Y*sin(α)*sin(β). The reference-point transfer in right-handed components uses M_G=M_A-r_AG×F_b, and the declaration does not write that formula.",
        "",
        "The checks are: the airspeed axes close in one flight state, and one track roll makes gRb*bRa=gRp*pRa; the airflow basis is orthogonal and right-handed; the product of the two stages equals the full transform; the force and the moment at one point keep their lengths; the reference-length scale of a moment coefficient agrees with the dimensional moment; the reference-point transfer can be reversed; rotating first and transferring the point first agree.",
        "",
        "",
    ])


def samples() -> list[dict]:
    def flipped(angle: str) -> dict:
        spec = deepcopy(STANDARD_SPEC)
        spec[angle] = {"sign": NEGATIVE}
        return spec

    rows = [
        {"id": "01", "factor": "none", "title": "common default", "change": "All five frames and all eight angles use the standard definition.", "spec": deepcopy(STANDARD_SPEC)},
        {"id": "02", "factor": "yaw", "title": "yaw reversed", "change": "Yaw is negative when the nose yaws to the right of the ground.", "spec": flipped(PSI)},
        {"id": "03", "factor": "pitch", "title": "pitch reversed", "change": "Pitch is negative when the nose pitches above the ground.", "spec": flipped(THETA)},
        {"id": "04", "factor": "roll", "title": "roll reversed", "change": "Roll is negative when the aircraft rolls to its right.", "spec": flipped(PHI)},
        {"id": "05", "factor": "track inclination", "title": "track inclination reversed", "change": "Track inclination is negative when the aircraft climbs.", "spec": flipped(MU)},
        {"id": "06", "factor": "angle of attack", "title": "angle of attack reversed", "change": "Angle of attack is negative when the airspeed projection lies below the airframe axis.", "spec": flipped(ALPHA)},
        {"id": "07", "factor": "sideslip", "title": "sideslip reversed", "change": "Sideslip is negative when the airspeed lies to the right of the aircraft.", "spec": flipped(BETA)},
        {"id": "08", "factor": "track azimuth", "title": "track azimuth reversed", "change": "Track azimuth is negative when the horizontal airspeed projection lies to the right of Xg.", "spec": flipped(VARPHI)},
        {"id": "09", "factor": "positive side force", "title": "side force positive to the left", "change": "Side force is positive along -Y_a of the airflow frame.", "spec": deepcopy(STANDARD_SPEC), "side_y": -1},
        {"id": "10", "factor": "positive lift", "title": "lift positive downward", "change": "Lift is positive along +Z_a of the airflow frame.", "spec": deepcopy(STANDARD_SPEC), "lift_z": 1},
    ]
    factors = [row["factor"] for row in rows]
    if factors[0] != "none" or len(factors) != 10 or len(set(factors)) != 10:
        raise RuntimeError("The flight-mechanics main experiment must be one standard definition and nine distinct single-definition perturbations.")
    return rows


def _check_standard(record: dict) -> None:
    by_id = {item["id"]: item["answer"] for item in record["questions"]}
    expected = {
        4: "right on the ground",
        5: "upward",
        12: "Cz(-ψ)*Cy(-θ)*Cx(-Φ)",
        15: "[cos(20°)*cos(30°), sin(30°)*cos(20°), -sin(20°)]",
        16: "upward",
        22: "upward",
        9: "the rightmost factor",
        25: "Cz(-φ)*Cy(-μ)",
        33: "the directions of airflow Ya and Za relative to the track frame",
        57: "at least one error",
        28: "-sin(10°)",
        29: "upward",
        34: "[cos(φ)*cos(μ), sin(φ)*cos(μ), -sin(μ)]",
        36: "downward",
        37: "right",
        40: "Cy(α)",
        41: "Cz(-β)",
        42: "Cy(α)*Cz(-β)",
        45: "[cos(α)*cos(β), sin(β), sin(α)*cos(β)]",
        48: "atan2(vy, sqrt(vx^2+vz^2))",
        49: "positive",
        50: "positive",
        51: "atan2(vz, vx)",
        99: "[-q_bar*S*C_D, q_bar*S*C_Ya, -q_bar*S*C_L]",
        103: "A",
        68: "[-D, Y, -L]",
        72: "downward",
        73: "positive",
        74: "[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]",
        75: "right",
        76: "positive",
        77: "[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]",
        79: "[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]",
        80: "[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]",
        81: "[[cos(α)*cos(β), sin(β), sin(α)*cos(β)], [-cos(α)*sin(β), cos(β), -sin(α)*sin(β)], [-sin(α), 0, cos(α)]]",
        83: "[[cos(α)*cos(β), -cos(α)*sin(β), -sin(α)], [sin(β), cos(β), 0], [sin(α)*cos(β), -sin(α)*sin(β), cos(α)]]",
        84: "-D*cos(α)*cos(β) - Y*cos(α)*sin(β) + L*sin(α)",
        86: "-D*sin(α)*cos(β) - Y*sin(α)*sin(β) - L*cos(α)",
        90: "[0, 0, -4]",
        93: "l_a*cos(α)*cos(β) - m_a*cos(α)*sin(β) - n_a*sin(α)",
        102: "(b/c_bar)*C_la*sin(β) + C_ma*cos(β)",
        105: "L_f - L_A",
        109: "[X_N - x_Ge, y_Ge, z_Ge]",
        114: "negative",
        119: "positive",
        120: "l_A + z*Y_b - y*Z",
        124: "nose-down",
        127: "C_lA + (z/b)*C_Yb - (y/b)*C_Z",
    }
    mismatches = {number: by_id[number] for number, answer in expected.items() if by_id[number] != answer}
    if mismatches:
        raise AssertionError(mismatches)


def _index(rows: list[dict]) -> str:
    lines = [
        "# Flight-mechanics coordinate transforms",
        "",
        "The main experiment is one standard definition and nine single-definition perturbations. Each item has 131 blanks, numbered 1 through 131. A single-definition perturbation changes one definition. The first four subprocesses convert coordinates. The last two convert the six aerodynamic components and transfer the moment reference point, and they are injected separately.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- [Data {row['id']} {row['title']}](sample_{row['id']}.md): "
            f"single-definition perturbation: {row['factor']}. {row['change']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    index_rows = []
    for sample in samples():
        record = build(sample["spec"], lift_z=sample.get("lift_z", -1), side_y=sample.get("side_y", 1))
        if sample["id"] == "01":
            _check_standard(record)
        document = render(sample, record)
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(document, encoding="utf-8")
        payload.append({
            "id": sample["id"],
            "factor": sample["factor"],
            "title": sample["title"],
            "change": sample["change"],
            "domain": "flight mechanics",
            "process": "spatial transforms",
            "declaration_products": {
                "gRb": record["ground_body"],
                "gRp": record["ground_track"],
                "pRa": record["track_air"],
                "bRg": record["body_from_ground"],
                "bRs": record["body_stability"],
                "sRa": record["stability_air"],
                "bRa": record["body_air"],
            },
            "questions": [
                {
                    "id": str(item["id"]),
                    "label": item["label"],
                    "answer": item["answer"] if item["options"] is None else answer_letter(item["answer"], item["options"]),
                    "options": item["options"] or [],
                    "format": item["format"],
                }
                for item in record["questions"]
            ],
        })
        index_rows.append(sample)
        print(f"{sample['id']} {sample['title']}: {len(record['questions'])} blanks")
    (OUT_DIR / "samples.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "00_index.md").write_text(_index(index_rows), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
