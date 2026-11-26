# -*- coding: utf-8 -*-
"""Cloze for ground frame to track frame, then track frame to airflow frame.

The manuscript is ground to track and airflow.md. Geometry and rotation sense come
from FlightFrameModel. Passive rotation and intrinsic order are fixed. Ground
to track rotates azimuth, then inclination. Track to airflow rotates bank.
The first angle's matrix is the rightmost factor.

[xp, yp, zp] = R1*R2*R3*[xg, yg, zg]
R3 is azimuth, R2 is inclination, and R1 is the zero-angle alignment RP_gp.
[xa, ya, za] = R1*R2*[xp, yp, zp]
R2 is bank and R1 is the zero-angle alignment RP_pa.
The ground-to-airflow matrix is the track-to-airflow matrix times the
ground-to-track matrix.

Track coordinate names in the cloze are Xp, Yp, Zp. The model stores them
as Xk, Yk, Zk. The manuscript leaves the azimuth and bank turn blanks
unlabeled; they use A2, the same label as the inclination turn.
"""

from __future__ import annotations

from copy import deepcopy

import sympy as sp

from domains.aerodynamics.ground_body_generate import legal_ground_frames
from flight_frame_model import (
    COORDINATE_AXES,
    GAMMA,
    GEOMETRY,
    MU,
    NEGATIVE,
    NEGATIVE_ROTATION,
    POSITIVE,
    POSITIVE_ROTATION,
    ROTATION_ANGLE,
    STANDARD_SPEC,
    VARPHI,
    DefinitionError,
    FlightFrameModel,
)


PROCESS = "ground to track and airflow"
BLANK_COUNT = 48
TRACK_SHOW = {"Xk": "Xp", "Yk": "Yp", "Zk": "Zp"}
UNDRECTED_AXES = ["Xg", "Yg", "Zg", "Xp", "Yp", "Zp", "Xa", "Ya", "Za"]
GROUND_DIRECTED = ["+Xg", "-Xg", "+Yg", "-Yg", "+Zg", "-Zg"]
TRACK_DIRECTED = ["+Xp", "-Xp", "+Yp", "-Yp", "+Zp", "-Zp"]
AIR_DIRECTED = ["+Xa", "-Xa", "+Ya", "-Ya", "+Za", "-Za"]
FRAMES = ["track frame", "stability frame", "airframe"]
ANGLE_SETS = ["μ", "φ", "γ", "μ, φ, γ", "μ, φ", "φ, γ", "μ, γ"]
TRACK_SEQUENCES = ["φμγ", "μγφ", "γμφ", "μφ", "φμ", "φγ"]
TRANSFORMS = ["active", "passive"]
ROTATIONS = ["extrinsic fixed axes", "intrinsic moving axes"]
SIDES = ["the leftmost factor", "the rightmost factor"]
YAW_SIDES = ["left", "right"]
VERTICALS = ["upward", "downward"]
BANK_SIDES = ["left", "right"]
TURNS = ["positive", "reverse"]
TRACK_CODES = ["R_φ", "R_μ", "RP_gp"]
AIR_CODES = ["R_φ", "R_μ", "RP_gp", "R_γ", "RP_pa"]
TOTAL_CODES = ["Sγ", "Sμφ"]
SYMBOL = {
    VARPHI: sp.Symbol("φ"),
    MU: sp.Symbol("μ"),
    GAMMA: sp.Symbol("γ"),
}


def _show(name: str) -> str:
    return TRACK_SHOW.get(name, name)


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


def _binding(model: FlightFrameModel, origin: str, destination: str, angle: str):
    found = [item for item in model.conversion(origin, destination) if item.angle == angle]
    if len(found) != 1 or not found[0].axis:
        raise DefinitionError(f"{angle} is not bound on {origin} to {destination}.")
    return found[0]


def _angle_matrix(model: FlightFrameModel, origin: str, destination: str, angle: str) -> sp.Matrix:
    binding = _binding(model, origin, destination, angle)
    axis = model.frames[binding.frame][binding.axis]
    if axis.rotation_matrix is None:
        raise DefinitionError(f"{binding.axis} has no passive matrix.")
    symbol = SYMBOL[angle]
    if binding.rotation_when_positive == POSITIVE_ROTATION:
        value = symbol
    elif binding.rotation_when_positive == NEGATIVE_ROTATION:
        value = -symbol
    else:
        raise DefinitionError(f"{angle} has no rotation sense on {origin} to {destination}.")
    return sp.simplify(axis.rotation_matrix.subs(ROTATION_ANGLE, value))


def _azimuth_side(model: FlightFrameModel) -> str:
    """Which side of +Xg the airspeed projection moves toward when azimuth is positive."""
    toward = model.binding(VARPHI).positive_toward
    vector = model.geometry["ground"].vector[toward]
    if vector == GEOMETRY["ground"].vector["ground_right"]:
        return "right"
    if vector == GEOMETRY["ground"].vector["ground_left"]:
        return "left"
    raise DefinitionError(f"Positive azimuth moves toward {toward}, which is not beside Xg.")


def _inclination(model: FlightFrameModel) -> str:
    toward = model.binding(MU).positive_toward
    if toward == "ground_up":
        return "upward"
    if toward == "ground_down":
        return "downward"
    raise DefinitionError(f"Positive inclination moves toward {toward}.")


def _bank_side(model: FlightFrameModel) -> str:
    if model.binding(GAMMA).sign == POSITIVE:
        return "right"
    if model.binding(GAMMA).sign == NEGATIVE:
        return "left"
    raise DefinitionError("Track-bank sign is missing.")


def _turn(model: FlightFrameModel, origin: str, destination: str, angle: str) -> str:
    sense = _binding(model, origin, destination, angle).rotation_when_positive
    if sense == POSITIVE_ROTATION:
        return "positive"
    if sense == NEGATIVE_ROTATION:
        return "reverse"
    raise DefinitionError(f"{angle} has no rotation sense.")


def _legal_xy_frames(owner: str) -> list[dict]:
    """Two-axis declarations. X and Y are declared; chirality fixes Z."""
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


def legal_track_frames() -> list[dict]:
    return _legal_xy_frames("track")


def legal_airflow_frames() -> list[dict]:
    return _legal_xy_frames("airflow")


def generate(spec: dict) -> dict:
    """Build the 41 cloze answers for one flight-mechanics declaration."""
    model = FlightFrameModel().declare(spec)
    for owner in ("ground", "track", "airflow"):
        missing = [name for name in COORDINATE_AXES[owner] if not model.frames[owner][name].points_to]
        if missing:
            raise DefinitionError(f"{owner} frame is incomplete: {', '.join(missing)}.")
    for angle in (VARPHI, MU, GAMMA):
        if model.binding(angle).sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{angle} needs a sign.")

    azimuth = _angle_matrix(model, "ground", "track", VARPHI)
    inclination = _angle_matrix(model, "ground", "track", MU)
    bank = _angle_matrix(model, "track", "airflow", GAMMA)
    align_track = sp.simplify(model.alignment("ground", "track"))
    align_air = sp.simplify(model.alignment("track", "airflow"))
    track_total = sp.simplify(align_track * inclination * azimuth)
    air_total = sp.simplify(align_air * bank)
    full_total = sp.simplify(air_total * track_total)
    ground = model.geometry["ground"]
    track = model.geometry["track"]
    airflow = model.geometry["airflow"]

    rows = [
        (1, "A1", "Intermediate frame from the ground frame to the airflow frame", "track frame", FRAMES),
        (2, "A1", "Euler angles between the ground frame and the track frame", "μ, φ", ANGLE_SETS),
        (3, "A1", "Euler angles between the ground frame and the airflow frame", "μ, φ, γ", ANGLE_SETS),
        (4, "A1", "Euler angle between the track frame and the airflow frame", "γ", ANGLE_SETS),
        (5, "C2", "Transform used from the ground frame to the track frame", "passive", TRANSFORMS),
        (6, "C2", "How the track angles are rotated", "intrinsic moving axes", ROTATIONS),
        (7, "C2", "Place of the matrix of the angle rotated first", "the rightmost factor", SIDES),
        (8, "A1", "Track-angle order that leaves axes already turned undisturbed", "φμ", TRACK_SEQUENCES),
        (9, "C1", "Directed axis of the positive ground normal", _directed(model, "ground", ground.positive["normal"]), GROUND_DIRECTED),
        (10, "C1", "Directed axis of the positive ground lateral direction", _directed(model, "ground", ground.positive["lateral"]), GROUND_DIRECTED),
        (11, "C1", "Directed axis of the positive ground axial direction", _directed(model, "ground", ground.positive["axial"]), GROUND_DIRECTED),
        (12, "C1", "Directed axis of the positive track normal", _directed(model, "track", track.positive["normal"]), TRACK_DIRECTED),
        (13, "C1", "Directed axis of the positive track lateral direction", _directed(model, "track", track.positive["lateral"]), TRACK_DIRECTED),
        (14, "C1", "Directed axis of the positive track axial direction", _directed(model, "track", track.positive["axial"]), TRACK_DIRECTED),
        (15, "C1", "Side of Xg for the airspeed projection when azimuth is positive", _azimuth_side(model), YAW_SIDES),
        (16, "A1", "Undirected axis of the azimuth rotation", _show(_binding(model, "ground", "track", VARPHI).axis), UNDRECTED_AXES),
        (17, "A2", "Sense of rotating φ about that axis", _turn(model, "ground", "track", VARPHI), TURNS),
        (18, "A2", "Passive rotation matrix of φ", _matrix_text(azimuth), None),
        (19, "A1", "Direction relative to the ground when track inclination is positive", _inclination(model), VERTICALS),
        (20, "A1", "Undirected axis of the inclination rotation", _show(_binding(model, "ground", "track", MU).axis), UNDRECTED_AXES),
        (21, "A1", "Sense of rotating μ about that axis", _turn(model, "ground", "track", MU), TURNS),
        (22, "A2", "Passive rotation matrix of μ", _matrix_text(inclination), None),
        (23, "A2", "RP_gp that aligns the ground axes with the track axes", _matrix_text(align_track), None),
        (24, "A2", "R1 in the product from the ground frame to the track frame", "RP_gp", TRACK_CODES),
        (25, "A2", "R2 in the product from the ground frame to the track frame", "R_μ", TRACK_CODES),
        (26, "A2", "R3 in the product from the ground frame to the track frame", "R_φ", TRACK_CODES),
        (27, "A2", "Expansion of R1 before the total matrix", _matrix_text(align_track), None),
        (28, "A2", "Expansion of R2 before the total matrix", _matrix_text(inclination), None),
        (29, "A2", "Expansion of R3 before the total matrix", _matrix_text(azimuth), None),
        (30, "A2", "Sμφ = R1*R2*R3", _matrix_text(track_total), None),
        (31, "C1", "Directed axis of the positive airflow normal", _directed(model, "airflow", airflow.positive["normal"]), AIR_DIRECTED),
        (32, "C1", "Directed axis of the positive airflow lateral direction", _directed(model, "airflow", airflow.positive["lateral"]), AIR_DIRECTED),
        (33, "C1", "Directed axis of the positive airflow axial direction", _directed(model, "airflow", airflow.positive["axial"]), AIR_DIRECTED),
        (34, "C1", "Side the aircraft rolls toward when track roll is positive", _bank_side(model), BANK_SIDES),
        (35, "A1", "Undirected axis of the track-roll rotation", _show(_binding(model, "track", "airflow", GAMMA).axis), UNDRECTED_AXES),
        (36, "A2", "Sense of rotating γ about that axis", _turn(model, "track", "airflow", GAMMA), TURNS),
        (37, "A2", "Passive rotation matrix of γ", _matrix_text(bank), None),
        (38, "A2", "RP_pa that aligns the track axes with the airflow axes", _matrix_text(align_air), None),
        (39, "A2", "R1 in the product from the track frame to the airflow frame", "RP_pa", AIR_CODES),
        (40, "A2", "R2 in the product from the track frame to the airflow frame", "R_γ", AIR_CODES),
        (41, "A2", "Expansion of R1 before the total matrix", _matrix_text(align_air), None),
        (42, "A2", "Expansion of R2 before the total matrix", _matrix_text(bank), None),
        (43, "A2", "Sγ = R1*R2", _matrix_text(air_total), None),
        (44, "A2", "R1 in the product from the ground frame to the airflow frame", "Sγ", TOTAL_CODES),
        (45, "A2", "R2 in the product from the ground frame to the airflow frame", "Sμφ", TOTAL_CODES),
        (46, "A2", "Expansion of R1 before the total matrix", _matrix_text(air_total), None),
        (47, "A2", "Expansion of R2 before the total matrix", _matrix_text(track_total), None),
        (48, "A2", "Sμφγ = R1*R2", _matrix_text(full_total), None),
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
    if by_id["30"] != _matrix_text(track_total):
        raise DefinitionError("Sμφ is not RP_gp*R_μ*R_φ.")
    if by_id["43"] != _matrix_text(air_total):
        raise DefinitionError("Sγ is not RP_pa*R_γ.")
    if by_id["48"] != _matrix_text(full_total):
        raise DefinitionError("Sμφγ is not Sγ*Sμφ.")
    return {"process": PROCESS, "items": items}


def _step(number: int) -> int:
    bounds = (4, 8, 14, 18, 22, 23, 29, 30, 33, 37, 38, 42, 43, 48)
    for step, bound in enumerate(bounds, start=1):
        if number <= bound:
            return step
    raise DefinitionError(f"No step for blank {number}.")


def _check_standard() -> None:
    payload = generate(deepcopy(STANDARD_SPEC))
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    identity = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    expected = {
        "1": "track frame",
        "2": "μ, φ",
        "3": "μ, φ, γ",
        "4": "γ",
        "5": "passive",
        "6": "intrinsic moving axes",
        "7": "the rightmost factor",
        "8": "φμ",
        "9": "+Zg",
        "10": "+Yg",
        "11": "+Xg",
        "12": "+Zp",
        "13": "+Yp",
        "14": "+Xp",
        "15": "right",
        "16": "Zg",
        "17": "positive",
        "18": "[[cos(φ), sin(φ), 0], [-sin(φ), cos(φ), 0], [0, 0, 1]]",
        "19": "upward",
        "20": "Yg",
        "21": "positive",
        "22": "[[cos(μ), 0, -sin(μ)], [0, 1, 0], [sin(μ), 0, cos(μ)]]",
        "23": identity,
        "24": "RP_gp",
        "25": "R_μ",
        "26": "R_φ",
        "31": "+Za",
        "32": "+Ya",
        "33": "+Xa",
        "34": "right",
        "35": "Xp",
        "36": "positive",
        "37": "[[1, 0, 0], [0, cos(γ), sin(γ)], [0, -sin(γ), cos(γ)]]",
        "38": identity,
        "39": "RP_pa",
        "40": "R_γ",
        "44": "Sγ",
        "45": "Sμφ",
    }
    for blank_id, answer in expected.items():
        if by_id[blank_id] != answer:
            raise AssertionError(f"ID {blank_id} is {by_id[blank_id]}, expected {answer}.")
    phi, mu, gamma = SYMBOL[VARPHI], SYMBOL[MU], SYMBOL[GAMMA]
    azimuth = sp.Matrix([
        [sp.cos(phi), sp.sin(phi), 0],
        [-sp.sin(phi), sp.cos(phi), 0],
        [0, 0, 1],
    ])
    inclination = sp.Matrix([
        [sp.cos(mu), 0, -sp.sin(mu)],
        [0, 1, 0],
        [sp.sin(mu), 0, sp.cos(mu)],
    ])
    bank = sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(gamma), sp.sin(gamma)],
        [0, -sp.sin(gamma), sp.cos(gamma)],
    ])
    if by_id["30"] != _matrix_text(sp.simplify(inclination * azimuth)):
        raise AssertionError(f"Sμφ is {by_id['30']}.")
    if by_id["43"] != _matrix_text(sp.simplify(bank)):
        raise AssertionError("Sγ is not the bank matrix when the alignment is identity.")
    if by_id["48"] != _matrix_text(sp.simplify(bank * inclination * azimuth)):
        raise AssertionError(f"Sμφγ is {by_id['48']}.")
    labels = {item["label"] for item in payload["items"]}
    if labels != {"C1", "C2", "A1", "A2"}:
        raise AssertionError(labels)
    matrix_labels = {item["label"] for item in payload["items"] if str(item["answer"]).startswith("[[")}
    if matrix_labels != {"A2"}:
        raise AssertionError(matrix_labels)
    code_ids = {"24", "25", "26", "39", "40", "44", "45"}
    if any(item["label"] != "A2" for item in payload["items"] if item["manuscript_id"] in code_ids):
        raise AssertionError("A matrix-code choice is not A2.")
    grounds = legal_ground_frames()
    tracks = legal_track_frames()
    airs = legal_airflow_frames()
    if len(grounds) != 16 or len(tracks) != 48 or len(airs) != 48:
        raise AssertionError(f"Frame counts: ground {len(grounds)}, track {len(tracks)}, airflow {len(airs)}.")
    if STANDARD_SPEC["ground_frame"] not in grounds:
        raise AssertionError("The standard ground frame is missing from the legal list.")
    if STANDARD_SPEC["track_frame"] in tracks or STANDARD_SPEC["airflow_frame"] in airs:
        raise AssertionError("An X-and-Z standard frame was listed as an X-and-Y declaration.")

    negative = deepcopy(STANDARD_SPEC)
    negative[VARPHI] = {"sign": NEGATIVE}
    negative[MU] = {"sign": NEGATIVE}
    negative[GAMMA] = {"sign": NEGATIVE}
    flipped = {item["manuscript_id"]: item["answer"] for item in generate(negative)["items"]}
    if flipped["15"] != "left" or flipped["19"] != "downward" or flipped["34"] != "left":
        raise AssertionError("A reversed track angle still points to the standard side.")
    if flipped["17"] != "reverse" or flipped["21"] != "reverse" or flipped["36"] != "reverse":
        raise AssertionError("A reversed track angle still uses the standard turn.")
    if [flipped[item] for item in ("24", "25", "26", "39", "40", "44", "45")] != [
        "RP_gp", "R_μ", "R_φ", "RP_pa", "R_γ", "Sγ", "Sμφ",
    ]:
        raise AssertionError("Matrix codes changed with the angle signs.")
    if flipped["18"] != "[[cos(φ), -sin(φ), 0], [sin(φ), cos(φ), 0], [0, 0, 1]]":
        raise AssertionError(f"Negative azimuth matrix is {flipped['18']}.")


if __name__ == "__main__":
    _check_standard()
    counted = {}
    for item in generate(deepcopy(STANDARD_SPEC))["items"]:
        counted[item["label"]] = counted.get(item["label"], 0) + 1
    print(f"{PROCESS}: {BLANK_COUNT} blanks checked")
    print("Labels: " + ", ".join(f"{name} {counted[name]}" for name in ("C1", "C2", "A1", "A2")))
