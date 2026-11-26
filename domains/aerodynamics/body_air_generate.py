# -*- coding: utf-8 -*-
"""Cloze for the body frame to the airflow frame through the stability frame.

The manuscript is the airframe to the airflow frame. Geometry and rotation sense come from
FlightFrameModel. Passive rotation is fixed, and the first angle's matrix is
the rightmost factor.

The stability axial axis is the airspeed projected into the symmetry plane, so
body and stability differ by angle of attack, and stability and airflow differ
by sideslip. The manuscript's set answers name those two angles the other way
around; the later steps rotate alpha on the first leg and beta on the second,
and this generator follows those steps.

[xs, ys, zs] = R1*R2*[xb, yb, zb]
R2 is angle of attack and R1 is the zero-angle alignment RP_s.
[xa, ya, za] = R1*R2*[xs, ys, zs]
R2 is sideslip and R1 is the zero-angle alignment RP_sa.
The body-to-airflow matrix is the stability-to-airflow matrix times the
body-to-stability matrix.

Alpha stays on the stability lateral axis and beta on the body normal axis,
which is where the model binds them. Body coordinate names in the cloze are
xb, yb, zb. The model stores them as Xb, Yb, Zb. The manuscript leaves the two
turn blanks unlabeled; they use A2. Its last choice was numbered 40 and the
total matrix reused 34; both are renumbered so the blanks run from 1 to 35.
"""

from __future__ import annotations

from copy import deepcopy

import sympy as sp

from domains.aerodynamics.ground_body_generate import legal_body_frames
from domains.aerodynamics.ground_track_air_generate import legal_airflow_frames
from flight_frame_model import (
    ALPHA,
    BETA,
    COORDINATE_AXES,
    NEGATIVE,
    NEGATIVE_ROTATION,
    POSITIVE,
    POSITIVE_ROTATION,
    ROTATION_ANGLE,
    STANDARD_SPEC,
    DefinitionError,
    FlightFrameModel,
)


PROCESS = "body to airflow"
BLANK_COUNT = 41
BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
UNDRECTED_AXES = ["xb", "yb", "zb", "Xs", "Ys", "Zs", "Xa", "Ya", "Za"]
BODY_DIRECTED = ["+xb", "-xb", "+yb", "-yb", "+zb", "-zb"]
STABILITY_DIRECTED = ["+Xs", "-Xs", "+Ys", "-Ys", "+Zs", "-Zs"]
AIR_DIRECTED = ["+Xa", "-Xa", "+Ya", "-Ya", "+Za", "-Za"]
FRAMES = ["track frame", "stability frame", "ground frame"]
ANGLE_SETS = ["α", "β", "α, β"]
TRANSFORMS = ["active", "passive"]
ROTATIONS = ["extrinsic fixed axes", "intrinsic moving axes"]
FACTOR_PLACE = ["the leftmost factor", "the rightmost factor"]
SIDE_WORDS = ["left", "right"]
VERTICALS = ["upward", "downward"]
TURNS = ["positive", "reverse"]
STABILITY_CODES = ["R_α", "RP_s"]
AIR_CODES = ["R_α", "R_β", "RP_s", "RP_sa"]
TOTAL_CODES = ["Sβ", "Sα"]
SYMBOL = {
    ALPHA: sp.Symbol("α"),
    BETA: sp.Symbol("β"),
}


def legal_stability_frames() -> list[dict]:
    """Two-axis stability declarations. X and Y are declared; chirality fixes Z."""
    return _legal_xy_frames("stability")


def _legal_xy_frames(owner: str) -> list[dict]:
    x_name, y_name, _z_name = COORDINATE_AXES[owner]
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
                            x_name: {"along": f"{owner}_{x_role}", "points": f"{owner}_{x_sense}"},
                            y_name: {"along": f"{owner}_{y_role}", "points": f"{owner}_{y_sense}"},
                        }
                        probe = deepcopy(STANDARD_SPEC)
                        probe[f"{owner}_frame"] = spec
                        try:
                            FlightFrameModel().declare(probe)
                        except DefinitionError:
                            continue
                        found.append(spec)
    return found


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


LEG = {ALPHA: ("body", "stability"), BETA: ("stability", "airflow")}


def _record(model: FlightFrameModel, angle: str):
    """Binding of the angle on its own leg transform, not on the direct one."""
    origin, destination = LEG[angle]
    found = [item for item in model.conversion(origin, destination) if item.angle == angle]
    if len(found) != 1 or not found[0].axis:
        raise DefinitionError(f"{angle} is not bound on {origin}->{destination}.")
    if found[0].rotation_when_positive not in (POSITIVE_ROTATION, NEGATIVE_ROTATION):
        raise DefinitionError(f"{angle} has no rotation sense.")
    return found[0]


def _angle_matrix(model: FlightFrameModel, angle: str) -> sp.Matrix:
    item = _record(model, angle)
    axis = model.frames[item.frame][item.axis]
    if axis.rotation_matrix is None:
        raise DefinitionError(f"{item.axis} has no passive matrix.")
    symbol = SYMBOL[angle]
    value = symbol if item.rotation_when_positive == POSITIVE_ROTATION else -symbol
    return sp.simplify(axis.rotation_matrix.subs(ROTATION_ANGLE, value))


def _attack_side(model: FlightFrameModel) -> str:
    toward = model.binding(ALPHA).positive_toward
    if toward == "body_down":
        return "downward"
    if toward == "body_up":
        return "upward"
    raise DefinitionError(f"Positive angle of attack moves toward {toward}.")


def _sideslip_side(model: FlightFrameModel) -> str:
    toward = model.binding(BETA).positive_toward
    if toward == "body_right":
        return "right"
    if toward == "body_left":
        return "left"
    raise DefinitionError(f"Positive sideslip moves toward {toward}.")


def _turn(model: FlightFrameModel, angle: str) -> str:
    sense = _record(model, angle).rotation_when_positive
    if sense == POSITIVE_ROTATION:
        return "positive"
    return "reverse"

def generate(spec: dict) -> dict:
    """Build the 35 cloze answers for one flight-mechanics declaration."""
    model = FlightFrameModel().declare(spec)
    for owner in ("body", "stability", "airflow"):
        missing = [name for name in COORDINATE_AXES[owner] if not model.frames[owner][name].points_to]
        if missing:
            raise DefinitionError(f"{owner} frame is incomplete: {', '.join(missing)}.")
    for angle in (ALPHA, BETA):
        if model.binding(angle).sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{angle} needs a sign.")

    attack = _angle_matrix(model, ALPHA)
    sideslip = _angle_matrix(model, BETA)
    align_stability = sp.simplify(model.alignment("body", "stability"))
    align_air = sp.simplify(model.alignment("stability", "airflow"))
    stability_total = sp.simplify(align_stability * attack)
    air_total = sp.simplify(align_air * sideslip)
    full_total = sp.simplify(air_total * stability_total)
    body = model.geometry["body"]
    stability = model.geometry["stability"]
    airflow = model.geometry["airflow"]

    rows = [
        (1, "A1", "Intermediate frame from the airframe to the airflow frame", "stability frame", FRAMES),
        (2, "A1", "Euler angles between the airframe and the airflow frame", "α, β", ANGLE_SETS),
        (3, "A1", "Angle between the airframe and the stability frame", "α", ANGLE_SETS),
        (4, "A1", "Angle between the stability frame and the airflow frame", "β", ANGLE_SETS),
        (5, "C2", "Transform used from the airframe to the stability frame", "passive", TRANSFORMS),
        (6, "C2", "How the airflow angles are rotated", "intrinsic moving axes", ROTATIONS),
        (7, "C2", "Place of the matrix of the angle rotated first", "the rightmost factor", FACTOR_PLACE),
        (8, "C1", "Directed axis of the positive airframe normal", _directed(model, "body", body.positive["normal"]), BODY_DIRECTED),
        (9, "C1", "Directed axis of the positive airframe lateral direction", _directed(model, "body", body.positive["lateral"]), BODY_DIRECTED),
        (10, "C1", "Directed axis of the positive airframe axial direction", _directed(model, "body", body.positive["axial"]), BODY_DIRECTED),
        (11, "C1", "Directed axis of the positive stability normal", _directed(model, "stability", stability.positive["normal"]), STABILITY_DIRECTED),
        (12, "C1", "Directed axis of the positive stability lateral direction", _directed(model, "stability", stability.positive["lateral"]), STABILITY_DIRECTED),
        (13, "C1", "Directed axis of the positive stability axial direction", _directed(model, "stability", stability.positive["axial"]), STABILITY_DIRECTED),
        (14, "C1", "Where the airspeed in the plane of symmetry points when angle of attack is positive", _attack_side(model), VERTICALS),
        (15, "A1", "Undirected axis of the angle-of-attack rotation", _show(_record(model, ALPHA).axis), UNDRECTED_AXES),
        (16, "A2", "Sense of rotating α about that axis", _turn(model, ALPHA), TURNS),
        (17, "A2", "Passive rotation matrix of α", _matrix_text(attack), None),
        (18, "A2", "RP_s that aligns the airframe axes with the stability axes", _matrix_text(align_stability), None),
        (19, "A2", "R1 in the product from the airframe to the stability frame", "RP_s", STABILITY_CODES),
        (20, "A2", "R2 in the product from the airframe to the stability frame", "R_α", STABILITY_CODES),
        (21, "A2", "Expansion of R1 before the total matrix", _matrix_text(align_stability), None),
        (22, "A2", "Expansion of R2 before the total matrix", _matrix_text(attack), None),
        (23, "A2", "Sα = R1*R2", _matrix_text(stability_total), None),
        (24, "C1", "Directed axis of the positive airflow normal", _directed(model, "airflow", airflow.positive["normal"]), AIR_DIRECTED),
        (25, "C1", "Directed axis of the positive airflow lateral direction", _directed(model, "airflow", airflow.positive["lateral"]), AIR_DIRECTED),
        (26, "C1", "Directed axis of the positive airflow axial direction", _directed(model, "airflow", airflow.positive["axial"]), AIR_DIRECTED),
        (27, "C1", "Side of the plane of symmetry where the airspeed lies when sideslip is positive", _sideslip_side(model), SIDE_WORDS),
        (28, "A1", "Undirected axis of the sideslip rotation", _show(_record(model, BETA).axis), UNDRECTED_AXES),
        (29, "A2", "Sense of rotating β about that axis", _turn(model, BETA), TURNS),
        (30, "A2", "Passive rotation matrix of β", _matrix_text(sideslip), None),
        (31, "A2", "RP_sa that aligns the stability axes with the airflow axes", _matrix_text(align_air), None),
        (32, "A2", "R1 in the product from the stability frame to the airflow frame", "RP_sa", AIR_CODES),
        (33, "A2", "R2 in the product from the stability frame to the airflow frame", "R_β", AIR_CODES),
        (34, "A2", "Expansion of R1 before the total matrix", _matrix_text(align_air), None),
        (35, "A2", "Expansion of R2 before the total matrix", _matrix_text(sideslip), None),
        (36, "A2", "Sβ = R1*R2", _matrix_text(air_total), None),
        (37, "A2", "R1 in the body to airflow product", "Sβ", TOTAL_CODES),
        (38, "A2", "R2 in the body to airflow product", "Sα", TOTAL_CODES),
        (39, "A2", "Expansion of R1 before the total matrix", _matrix_text(air_total), None),
        (40, "A2", "Expansion of R2 before the total matrix", _matrix_text(stability_total), None),
        (41, "A2", "Sαβ = R1*R2", _matrix_text(full_total), None),
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
    by_id = {item["manuscript_id"]: item["answer"] for item in items}
    if by_id["23"] != _matrix_text(stability_total) or by_id["36"] != _matrix_text(air_total):
        raise DefinitionError("A leg total is not its alignment times its rotation.")
    if by_id["41"] != _matrix_text(full_total):
        raise DefinitionError("Sαβ is not Sβ*Sα.")
    return {"process": PROCESS, "items": items}


def _step(number: int) -> int:
    bounds = (4, 7, 13, 17, 18, 22, 23, 26, 30, 31, 35, 36, 41)
    for step, bound in enumerate(bounds, start=1):
        if number <= bound:
            return step
    raise DefinitionError(f"No step for blank {number}.")


def _check_standard() -> None:
    payload = generate(deepcopy(STANDARD_SPEC))
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    identity = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    expected = {
        "1": "stability frame",
        "2": "α, β",
        "3": "α",
        "4": "β",
        "5": "passive",
        "6": "intrinsic moving axes",
        "7": "the rightmost factor",
        "8": "+zb",
        "9": "+yb",
        "10": "+xb",
        "11": "+Zs",
        "12": "+Ys",
        "13": "+Xs",
        "14": "downward",
        "15": "yb",
        "16": "reverse",
        "17": "[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]",
        "18": identity,
        "19": "RP_s",
        "20": "R_α",
        "24": "+Za",
        "25": "+Ya",
        "26": "+Xa",
        "27": "right",
        "28": "Zs",
        "29": "positive",
        "30": "[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]",
        "31": identity,
        "32": "RP_sa",
        "33": "R_β",
        "37": "Sβ",
        "38": "Sα",
    }
    for blank_id, answer in expected.items():
        if by_id[blank_id] != answer:
            raise AssertionError(f"ID {blank_id} is {by_id[blank_id]}, expected {answer}.")
    alpha, beta = SYMBOL[ALPHA], SYMBOL[BETA]
    attack = sp.Matrix([
        [sp.cos(alpha), 0, sp.sin(alpha)],
        [0, 1, 0],
        [-sp.sin(alpha), 0, sp.cos(alpha)],
    ])
    sideslip = sp.Matrix([
        [sp.cos(beta), sp.sin(beta), 0],
        [-sp.sin(beta), sp.cos(beta), 0],
        [0, 0, 1],
    ])
    if by_id["23"] != _matrix_text(attack):
        raise AssertionError("Sα is not the angle-of-attack matrix when the alignment is identity.")
    if by_id["36"] != _matrix_text(sideslip):
        raise AssertionError("Sβ is not the sideslip matrix when the alignment is identity.")
    if by_id["41"] != _matrix_text(sp.simplify(sideslip * attack)):
        raise AssertionError(f"Sαβ is {by_id['41']}.")
    labels = {item["label"] for item in payload["items"]}
    if labels != {"C1", "C2", "A1", "A2"}:
        raise AssertionError(labels)
    matrix_labels = {item["label"] for item in payload["items"] if str(item["answer"]).startswith("[[")}
    if matrix_labels != {"A2"}:
        raise AssertionError(matrix_labels)
    code_ids = {"19", "20", "32", "33", "37", "38"}
    if any(item["label"] != "A2" for item in payload["items"] if item["manuscript_id"] in code_ids):
        raise AssertionError("A matrix-code choice is not A2.")
    bodies = legal_body_frames()
    stables = legal_stability_frames()
    airs = legal_airflow_frames()
    if len(bodies) != 48 or len(stables) != 48 or len(airs) != 48:
        raise AssertionError(f"Frame counts: body {len(bodies)}, stability {len(stables)}, airflow {len(airs)}.")
    if STANDARD_SPEC["body_frame"] in bodies or STANDARD_SPEC["stability_frame"] in stables:
        raise AssertionError("An X-and-Z standard frame was listed as an X-and-Y declaration.")
    if STANDARD_SPEC["airflow_frame"] in airs:
        raise AssertionError("The standard airflow frame was listed as an X-and-Y declaration.")

    negative = deepcopy(STANDARD_SPEC)
    negative[ALPHA] = {"sign": NEGATIVE}
    negative[BETA] = {"sign": NEGATIVE}
    flipped = {item["manuscript_id"]: item["answer"] for item in generate(negative)["items"]}
    if flipped["14"] != "upward" or flipped["27"] != "left":
        raise AssertionError("A reversed airflow angle still points to the standard side.")
    if flipped["16"] != "positive" or flipped["29"] != "reverse":
        raise AssertionError("A reversed airflow angle kept the standard turn.")
    if [flipped[item] for item in ("19", "20", "32", "33", "37", "38")] != [
        "RP_s", "R_α", "RP_sa", "R_β", "Sβ", "Sα",
    ]:
        raise AssertionError("Matrix codes changed with the angle signs.")
    if flipped["17"] != "[[cos(α), 0, -sin(α)], [0, 1, 0], [sin(α), 0, cos(α)]]":
        raise AssertionError(f"Negative angle-of-attack matrix is {flipped['17']}.")
    if flipped["30"] != "[[cos(β), -sin(β), 0], [sin(β), cos(β), 0], [0, 0, 1]]":
        raise AssertionError(f"Negative sideslip matrix is {flipped['30']}.")


if __name__ == "__main__":
    _check_standard()
    counted = {}
    for item in generate(deepcopy(STANDARD_SPEC))["items"]:
        counted[item["label"]] = counted.get(item["label"], 0) + 1
    print(f"{PROCESS}: {BLANK_COUNT} blanks checked")
    print("Labels: " + ", ".join(f"{name} {counted[name]}" for name in ("C1", "C2", "A1", "A2")))
