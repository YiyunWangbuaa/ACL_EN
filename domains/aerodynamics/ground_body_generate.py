# -*- coding: utf-8 -*-
"""Cloze for the ground-frame to body-frame attitude transform.

The manuscript is the ground frame to the airframe. Geometry and rotation sense come from
FlightFrameModel. Passive rotation and intrinsic order are fixed: yaw, then
pitch, then roll. The first angle's matrix is the rightmost factor.

[xb, yb, zb] = R1*R2*R3*R4*[xg, yg, zg]
R4 is R_ψ, R3 is R_θ, R2 is R_Φ, and R1 is RP_g, the zero-angle alignment that
renames ground components into body components. Those four factors are choices.
RP_g itself and the product SθψΦ are written matrices. The total is that product.
Body coordinate names in the cloze are xb, yb, zb. The model stores them as Xb, Yb, Zb.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations

import sympy as sp

from flight_frame_model import (
    ANGLE_SPEC,
    ATTITUDE_ORDER,
    COORDINATE_AXES,
    GEOMETRY,
    NEGATIVE,
    NEGATIVE_ROTATION,
    PHI,
    POSITIVE,
    POSITIVE_ROTATION,
    PSI,
    ROTATION_ANGLE,
    STANDARD_SPEC,
    THETA,
    TRACK_ORDER,
    DefinitionError,
    FlightFrameModel,
)


PROCESS = "ground to body"
BLANK_COUNT = 32
BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
UNDRECTED_AXES = ["Xg", "Yg", "Zg", "xb", "yb", "zb"]
GROUND_DIRECTED = ["+Xg", "-Xg", "+Yg", "-Yg", "+Zg", "-Zg"]
BODY_DIRECTED = ["+xb", "-xb", "+yb", "-yb", "+zb", "-zb"]
TRANSFORMS = ["active", "passive"]
ROTATIONS = ["extrinsic fixed axes", "intrinsic moving axes"]
SIDES = ["the leftmost factor", "the rightmost factor"]
YAW_SIDES = ["left", "right"]
PITCH_MOTIONS = ["downward", "upward"]
ROLL_SIDES = ["right", "left"]
TURNS = ["positive", "reverse"]
PRODUCT_CODES = ["R_ψ", "R_θ", "R_Φ", "RP_g"]
SYMBOL = {
    PSI: sp.Symbol("ψ"),
    THETA: sp.Symbol("θ"),
    PHI: sp.Symbol("Φ"),
}


def _order_text(names: tuple[str, ...]) -> str:
    return ", ".join(ANGLE_SPEC[name].symbol for name in names)


ATTITUDE_ORDERS = [_order_text(item) for item in permutations(ATTITUDE_ORDER)]
TRACK_ORDERS = [_order_text(item) for item in permutations(TRACK_ORDER)]
ANGLE_ORDERS = ATTITUDE_ORDERS + TRACK_ORDERS


def _show(name: str) -> str:
    return BODY_SHOW.get(name, name)


def _text(expr) -> str:
    return str(sp.simplify(expr))


def _matrix_text(matrix: sp.Matrix) -> str:
    rows = []
    for row in range(matrix.rows):
        cells = ", ".join(_text(matrix[row, column]) for column in range(matrix.cols))
        rows.append(f"[{cells}]")
    return "[" + ", ".join(rows) + "]"


def _dot(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _directed(model: FlightFrameModel, owner: str, direction: str) -> str:
    geometry = model.geometry[owner]
    target = geometry.vector[direction]
    for name, axis in model.frames[owner].items():
        alignment = _dot(axis.vector(geometry), target)
        shown = _show(name)
        if alignment == 1:
            return f"+{shown}"
        if alignment == -1:
            return f"-{shown}"
    raise DefinitionError(f"No {owner} axis lies on {direction}.")


def _binding(model: FlightFrameModel, angle: str):
    found = [item for item in model.conversion("ground", "body") if item.angle == angle]
    if len(found) != 1 or not found[0].axis:
        raise DefinitionError(f"{angle} is not bound on the ground-to-body transform.")
    return found[0]


def _angle_matrix(model: FlightFrameModel, angle: str) -> sp.Matrix:
    binding = _binding(model, angle)
    axis = model.frames[binding.frame][binding.axis]
    if axis.rotation_matrix is None:
        raise DefinitionError(f"{binding.axis} has no passive matrix.")
    symbol = SYMBOL[angle]
    if binding.rotation_when_positive == POSITIVE_ROTATION:
        value = symbol
    elif binding.rotation_when_positive == NEGATIVE_ROTATION:
        value = -symbol
    else:
        raise DefinitionError(f"{angle} has no rotation sense on the ground-to-body transform.")
    return sp.simplify(axis.rotation_matrix.subs(ROTATION_ANGLE, value))


def _yaw_side(model: FlightFrameModel) -> str:
    """Which side of +Xg the nose moves toward when yaw is positive."""
    toward = model.binding(PSI).positive_toward
    vector = model.geometry["ground"].vector[toward]
    if vector == GEOMETRY["ground"].vector["ground_right"]:
        return "right"
    if vector == GEOMETRY["ground"].vector["ground_left"]:
        return "left"
    raise DefinitionError(f"Positive yaw moves toward {toward}, which is not beside Xg.")


def _pitch_motion(model: FlightFrameModel) -> str:
    toward = model.binding(THETA).positive_toward
    if toward == "ground_up":
        return "upward"
    if toward == "ground_down":
        return "downward"
    raise DefinitionError(f"Positive pitch moves toward {toward}.")


def _roll_side(model: FlightFrameModel) -> str:
    if model.binding(PHI).sign == POSITIVE:
        return "right"
    if model.binding(PHI).sign == NEGATIVE:
        return "left"
    raise DefinitionError("Roll sign is missing.")


def _turn(model: FlightFrameModel, angle: str) -> str:
    sense = _binding(model, angle).rotation_when_positive
    if sense == POSITIVE_ROTATION:
        return "positive"
    if sense == NEGATIVE_ROTATION:
        return "reverse"
    raise DefinitionError(f"{angle} has no rotation sense.")


def legal_ground_frames() -> list[dict]:
    found = []
    geometry = GEOMETRY["ground"]
    for chirality in ("right", "left"):
        for name in ("Yg", "Zg"):
            for role in ("normal", "lateral"):
                for sense in (POSITIVE, NEGATIVE):
                    spec = {
                        "chirality": chirality,
                        name: {
                            "along": f"ground_{role}",
                            "points": geometry.direction(role, sense),
                        },
                    }
                    probe = deepcopy(STANDARD_SPEC)
                    probe["ground_frame"] = spec
                    try:
                        FlightFrameModel().declare(probe)
                    except DefinitionError:
                        continue
                    found.append(spec)
    return found


def legal_body_frames() -> list[dict]:
    """Two-axis body declarations. X and Y are declared; chirality fixes Z."""
    found = []
    points = {
        "axial": ("forward", "rearward"),
        "normal": ("up", "down"),
        "lateral": ("left", "right"),
    }
    roles = ("axial", "normal", "lateral")
    for chirality in ("right", "left"):
        for x_role in roles:
            for y_role in roles:
                if y_role == x_role:
                    continue
                for x_sense in points[x_role]:
                    for y_sense in points[y_role]:
                        spec = {
                            "chirality": chirality,
                            "Xb": {"along": f"body_{x_role}", "points": f"body_{x_sense}"},
                            "Yb": {"along": f"body_{y_role}", "points": f"body_{y_sense}"},
                        }
                        probe = deepcopy(STANDARD_SPEC)
                        probe["body_frame"] = spec
                        try:
                            FlightFrameModel().declare(probe)
                        except DefinitionError:
                            continue
                        found.append(spec)
    return found


def generate(spec: dict) -> dict:
    """Build the 27 cloze answers for one flight-mechanics declaration."""
    model = FlightFrameModel().declare(spec)
    for owner in ("ground", "body"):
        missing = [name for name in COORDINATE_AXES[owner] if not model.frames[owner][name].points_to]
        if missing:
            raise DefinitionError(f"{owner} frame is incomplete: {', '.join(missing)}.")
    for angle in ATTITUDE_ORDER:
        if model.binding(angle).sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{ANGLE_SPEC[angle].title} needs a sign.")

    yaw = _angle_matrix(model, PSI)
    pitch = _angle_matrix(model, THETA)
    roll = _angle_matrix(model, PHI)
    alignment = sp.simplify(model.alignment("ground", "body"))
    # Rightmost factor is the first angle. Alignment renames axes last.
    total = sp.simplify(alignment * roll * pitch * yaw)
    ground = model.geometry["ground"]
    body = model.geometry["body"]

    rows = [
        (1, "C2", "Transform used from the ground frame to the airframe", "passive", TRANSFORMS),
        (2, "C2", "How the attitude angles are rotated", "intrinsic moving axes", ROTATIONS),
        (3, "C2", "Place of the matrix of the angle rotated first", "the rightmost factor", SIDES),
        (4, "A1", "Angle order that leaves axes already turned undisturbed", _order_text(ATTITUDE_ORDER), ANGLE_ORDERS),
        (5, "C1", "Directed axis of the positive ground normal", _directed(model, "ground", ground.positive["normal"]), GROUND_DIRECTED),
        (6, "C1", "Directed axis of the positive ground lateral direction", _directed(model, "ground", ground.positive["lateral"]), GROUND_DIRECTED),
        (7, "C1", "Directed axis of the positive ground axial direction", _directed(model, "ground", ground.positive["axial"]), GROUND_DIRECTED),
        (8, "C1", "Directed axis of the positive airframe normal", _directed(model, "body", body.positive["normal"]), BODY_DIRECTED),
        (9, "C1", "Directed axis of the positive airframe lateral direction", _directed(model, "body", body.positive["lateral"]), BODY_DIRECTED),
        (10, "C1", "Directed axis of the positive airframe axial direction", _directed(model, "body", body.positive["axial"]), BODY_DIRECTED),
        (11, "C1", "Side of Xg for the airframe-axis projection when yaw is positive", _yaw_side(model), YAW_SIDES),
        (12, "A1", "Undirected axis of the yaw rotation", _show(_binding(model, PSI).axis), UNDRECTED_AXES),
        (13, "A2", "Sense of rotating ψ about that axis", _turn(model, PSI), TURNS),
        (14, "A2", "Passive rotation matrix of ψ", _matrix_text(yaw), None),
        (15, "C1", "Where the nose points when pitch is positive", _pitch_motion(model), PITCH_MOTIONS),
        (16, "A1", "Undirected axis of the pitch rotation", _show(_binding(model, THETA).axis), UNDRECTED_AXES),
        (17, "A2", "Sense of rotating θ about that axis", _turn(model, THETA), TURNS),
        (18, "A2", "Passive rotation matrix of θ", _matrix_text(pitch), None),
        (19, "C1", "Side the aircraft rolls toward when roll is positive", _roll_side(model), ROLL_SIDES),
        (20, "A1", "Undirected axis of the roll rotation", _show(_binding(model, PHI).axis), UNDRECTED_AXES),
        (21, "A1", "Sense of rotating Φ about that axis", _turn(model, PHI), TURNS),
        (22, "A2", "Passive rotation matrix of Φ", _matrix_text(roll), None),
        (23, "A2", "RP_g that aligns the ground axes with the airframe axes", _matrix_text(alignment), None),
        (24, "A2", "R1 in the ground to body product", "RP_g", PRODUCT_CODES),
        (25, "A2", "R2 in the ground to body product", "R_Φ", PRODUCT_CODES),
        (26, "A2", "R3 in the ground to body product", "R_θ", PRODUCT_CODES),
        (27, "A2", "R4 in the ground to body product", "R_ψ", PRODUCT_CODES),
        (28, "A2", "Expansion of R1 before the total matrix", _matrix_text(alignment), None),
        (29, "A2", "Expansion of R2 before the total matrix", _matrix_text(roll), None),
        (30, "A2", "Expansion of R3 before the total matrix", _matrix_text(pitch), None),
        (31, "A2", "Expansion of R4 before the total matrix", _matrix_text(yaw), None),
        (32, "A2", "SθψΦ = R1*R2*R3*R4", _matrix_text(total), None),
    ]
    items = []
    for number, label, prompt, answer, options in rows:
        if options is not None and answer not in options:
            raise DefinitionError(f"ID {number} answer {answer} is outside its options.")
        items.append({
            "manuscript_id": str(number),
            "step": _step(number),
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer,
        })
    if len(items) != BLANK_COUNT:
        raise DefinitionError(f"Expected {BLANK_COUNT} blanks, built {len(items)}.")
    by_id = {item["manuscript_id"]: item for item in items}
    if by_id["23"]["answer"] != _matrix_text(alignment):
        raise DefinitionError("RP_g is not the zero-angle alignment.")
    if [by_id[str(number)]["answer"] for number in (24, 25, 26, 27)] != ["RP_g", "R_Φ", "R_θ", "R_ψ"]:
        raise DefinitionError("The product codes are not RP_g, R_Φ, R_θ, R_ψ.")
    if any(by_id[str(number)]["label"] != "A2" for number in (24, 25, 26, 27)):
        raise DefinitionError("The product-code choices are not A2.")
    if by_id["32"]["answer"] != _matrix_text(total):
        raise DefinitionError("The total matrix is not RP_g*R_Φ*R_θ*R_ψ.")
    if [by_id[str(number)]["answer"] for number in (28, 29, 30, 31)] != [
        _matrix_text(alignment), _matrix_text(roll), _matrix_text(pitch), _matrix_text(yaw),
    ]:
        raise DefinitionError("A factor expansion does not match its symbol.")
    return {"process": PROCESS, "items": items}


def _step(number: int) -> int:
    if number <= 4:
        return 1
    if number <= 10:
        return 2
    if number <= 14:
        return 3
    if number <= 18:
        return 4
    if number <= 22:
        return 5
    if number == 23:
        return 6
    if number <= 31:
        return 7
    return 8


def _check_standard() -> None:
    payload = generate(deepcopy(STANDARD_SPEC))
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    expected = {
        "1": "passive",
        "2": "intrinsic moving axes",
        "3": "the rightmost factor",
        "4": "ψ, θ, Φ",
        "5": "+Zg",
        "6": "+Yg",
        "7": "+Xg",
        "8": "+zb",
        "9": "+yb",
        "10": "+xb",
        "11": "right",
        "12": "Zg",
        "13": "positive",
        "14": "[[cos(ψ), sin(ψ), 0], [-sin(ψ), cos(ψ), 0], [0, 0, 1]]",
        "15": "upward",
        "16": "Yg",
        "17": "positive",
        "18": "[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]",
        "19": "right",
        "20": "Xg",
        "21": "positive",
        "22": "[[1, 0, 0], [0, cos(Φ), sin(Φ)], [0, -sin(Φ), cos(Φ)]]",
        "23": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "24": "RP_g",
        "25": "R_Φ",
        "26": "R_θ",
        "27": "R_ψ",
    }
    for blank_id, answer in expected.items():
        if by_id[blank_id] != answer:
            raise AssertionError(f"ID {blank_id} is {by_id[blank_id]}, expected {answer}.")
    if not str(by_id["32"]).startswith("[["):
        raise AssertionError("SθψΦ is not a written matrix.")
    labels = {item["label"] for item in payload["items"]}
    if labels != {"C1", "C2", "A1", "A2"}:
        raise AssertionError(labels)
    matrix_labels = {item["label"] for item in payload["items"] if str(item["answer"]).startswith("[[")}
    if matrix_labels != {"A2"}:
        raise AssertionError(matrix_labels)
    grounds = legal_ground_frames()
    bodies = legal_body_frames()
    if len(grounds) != 16:
        raise AssertionError(f"Expected 16 ground frames, found {len(grounds)}.")
    if STANDARD_SPEC["ground_frame"] not in grounds:
        raise AssertionError("The standard ground frame is missing from the legal list.")
    if len(bodies) != 48:
        raise AssertionError(f"Expected 48 body frames, found {len(bodies)}.")
    if any(item["label"] == "A1" and str(item["answer"]).startswith("[[") for item in payload["items"]):
        raise AssertionError("A matrix is still labeled A1.")


if __name__ == "__main__":
    _check_standard()
    print(f"{PROCESS}: {BLANK_COUNT} blanks checked")
