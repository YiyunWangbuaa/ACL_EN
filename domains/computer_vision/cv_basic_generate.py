# -*- coding: utf-8 -*-
"""Basic stereo-vision questions. The world frame is camera 1.

No object frame and no mechanism angles. Camera 2's pose relative to camera 1
is R21, t21. One line is parallel to camera 1's optical axis and misses the
optical center. An image point is triangulated back into camera 1.
"""
from __future__ import annotations

import json

import sympy as sp

from domain_layout import SETS
from domains.computer_vision import cv_model as cm
from domains.computer_vision.cv_dataset import _injection
from domains.computer_vision.cv_generate import (
    CAM_EULER,
    C1,
    C2,
    DIR_CN,
    DIR_NAME,
    K1,
    K2,
    SIX,
    STANDARD_SPEC,
    THREE_SIGN,
    _ROLE_POINTING,
    _axis_pointing,
    _bias,
    _choice,
    signed_rotation,
    _write_matrix,
    _write_vector,
    u1,
    u2,
    v1,
    v2,
)
from domains.computer_vision import review_export as rx

OUT_DIR = SETS["basic.vision"]["review"]
DATA_PATH = SETS["basic.vision"]["samples"]

CAM_AXES = {
    1: ("X1", "Y1", "Z1"),
    2: ("X2", "Y2", "Z2"),
}
AXIS_NAMES = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]
ROLE_CN = {"axial": "axial", "normal": "normal", "lateral": "lateral"}
POINT_CN = {
    "forward": "forward", "rearward": "rearward", "right": "right",
    "left": "left", "down": "down", "up": "up",
}
HAND_CN = {"left": "left-handed", "right": "right-handed"}
BASIC_ANGLES = ("pan", "tilt", "roll")
ANGLE_CHANGE = {
    "pan": "Camera azimuth is positive when the optical axis deflects to the left of camera 1.",
    "tilt": "Camera pitch is positive when the optical axis deflects downward in camera 1.",
    "roll": "Camera roll is positive when the right edge of the image deflects upward in the camera.",
}
FRAME_TITLE = {"wind": "world frame", "camera1": "camera-1 frame", "camera2": "camera-2 frame"}
FORBIDDEN = (
    "object frame", "fuselage", "wind tunnel", "schlieren", "airspeed", "shock",
    "angle of attack", "sideslip", "mechanism", "upright", "Xb", "Yb", "Zb",
    "Xw", "Yw", "Zw", "R_cw",
)


def _placed(item: dict, paragraph: int, lead: str, tail: str = "") -> dict:
    item["paragraph"] = paragraph
    item["lead"] = lead
    item["tail"] = tail
    return item


def _signed(names: tuple[str, ...]) -> list[str]:
    return list(names) + ["-" + name for name in names]


def _skew(vector: sp.Matrix) -> sp.Matrix:
    px, py, pz = vector[0], vector[1], vector[2]
    return sp.Matrix([[0, -pz, py], [pz, 0, -px], [-py, px, 0]])


def _coord_unit(frame: dict, names: tuple[str, ...], direction: str) -> sp.Matrix:
    """Unit vector of an absolute direction, written in this frame's coordinates."""
    target = cm.DIRECTIONS[direction]
    column = sp.zeros(3, 1)
    for index, name in enumerate(names):
        column[index] = int(cm._dot(frame[name], target))
    return column


def _camera2_from_camera1(scene: cm.Scene):
    """Camera1 is the world frame. R21 = R_φ2 · R_θ2 · R_ψ2 · RP1."""
    source = scene.cam1
    target = scene.cam2
    names = CAM_AXES[2]
    angles = (scene.psi2, scene.theta2, scene.phi2)
    symbols = ("ψ2", "θ2", "φ2")
    questions: list[dict] = []
    rp1 = sp.simplify(cm.basis(target, names).T * cm.basis(source, CAM_AXES[1]))
    axis_options = _signed(CAM_AXES[1] + names)
    role_options = [
        "camera-2 absolute axial", "camera-2 absolute normal", "camera-2 absolute lateral",
        "camera-1 absolute axial", "camera-1 absolute normal", "camera-1 absolute lateral",
    ]
    role_gold = {
        "pan": "camera-2 absolute normal",
        "tilt": "camera-2 absolute lateral",
        "roll": "camera-2 absolute axial",
    }

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        "First determine the transform from camera 1 to camera 2. What is the conversion direction?",
        [
            "camera-2 frame to camera-1 frame", "camera-1 frame to camera-2 frame",
            "camera-2 image to camera-1 frame", "camera-1 image to camera-2 frame",
        ],
        "camera-1 frame to camera-2 frame", "C2",
    ), 1, "First determine the transform from camera 1 to camera 2. The conversion direction is ", ". ")
    put(_choice("Is every coordinate transform active or passive?", ["active", "passive"], "passive", "C2"),
        1, "Every coordinate transform is ", ", ")
    put(_choice("Is the transform treated as moving axes or fixed axes?", ["moving axes", "fixed axes"], "moving axes", "C2"),
        1, "treated as ", ", ")
    put(_choice("Does the rotation use extrinsic or intrinsic?", ["extrinsic", "intrinsic"], "intrinsic", "C2"),
        1, "and uses ", ". ")
    target_leads = [
        "The absolute axial (forward), normal (up), and lateral (right) directions of camera 2 correspond to ",
        ", ", ", and ",
    ]
    source_leads = [
        ". The same three directions of camera 1 correspond to ",
        ", ", ", and ",
    ]
    for lead, direction in zip(target_leads, ("forward", "up", "right")):
        put(_choice(
            f"Which axis is camera 2's absolute {direction} direction?",
            axis_options, _axis_pointing(target, direction), "C1",
        ), 1, lead)
    for lead, direction in zip(source_leads, ("forward", "up", "right")):
        put(_choice(
            f"Which axis is camera 1's absolute {direction} direction?",
            axis_options, _axis_pointing(source, direction), "C1",
        ), 1, lead)
    put(_write_matrix("Write RP1, the preliminary matrix that aligns camera 1 with camera 2.", rp1),
        1, ". A preliminary matrix RP1 first makes the camera-1 axes agree with camera 2 and forms the level intermediate frame of camera 2. RP1 is ")
    euler_leads = [
        ". From that intermediate frame, the Euler order is ",
        ", then ",
        ", then ",
    ]
    for lead, name in zip(euler_leads, CAM_EULER):
        tail = "." if name == "roll" else ""
        put(_choice("Which angle is this step of camera 2's Euler order?", CAM_EULER, name, "A1"), 1, lead, tail)

    specs = (
        ("pan", "azimuth", symbols[0], angles[0], "axial", 2),
        ("tilt", "pitch", symbols[1], angles[1], "axial", 3),
        ("roll", "roll", symbols[2], angles[2], "lateral", 4),
    )
    matrices = {}
    for key, angle_name, symbol_name, angle, moving_role, paragraph in specs:
        role = {"pan": "normal", "tilt": "lateral", "roll": "axial"}[key]
        toward = scene.toward_vector(scene.angle_toward[key], target)
        side = DIR_CN[DIR_NAME[toward]]
        axis = cm.axis_on(target, role)[0]
        _moving_name, moving = cm.axis_on(target, moving_role)
        sign, matrix = signed_rotation(target, names, axis, moving, toward, angle)
        matrices[key] = matrix
        if key == "roll":
            opening = f"When camera 2 roll {symbol_name} is positive, the right edge of the image deflects toward "
        else:
            opening = f"When camera 2 {angle_name} {symbol_name} is positive, the optical axis deflects toward "
        put(_choice(
            f"When camera 2 {angle_name} {symbol_name} is positive, which way does it deflect in camera 1?",
            SIX, side, "A1",
        ), paragraph, opening, " of camera 1. ")
        put(_choice(
            f"Camera 2 {angle_name} {symbol_name} is a rotation about which absolute axis?",
            role_options, role_gold[key], "C2",
        ), paragraph, "This is a rotation about ", ". ")
        phrase, direction = _ROLE_POINTING[role]
        put(_choice(
            f"Which axis is the absolute {phrase} of camera 2?",
            axis_options, _axis_pointing(target, direction), "C1",
        ), paragraph, f"The absolute {phrase} of camera 2 corresponds to ", ". ")
        put(_choice(
            f"Which camera-2 axis does {symbol_name} rotate about?",
            AXIS_NAMES, axis, "C1",
        ), paragraph, "The rotation uses camera-2 axis ", " with sense ")
        put(_choice("Is the sense positive or negative?", THREE_SIGN, _bias(sign), "A1"),
            paragraph, "", ". ")
        put(_write_matrix(f"Write the rotation matrix R_{symbol_name}.", matrix),
            paragraph, f"The rotation matrix R_{symbol_name} is ", ".")

    factors = [f"R_{symbols[0]}", f"R_{symbols[1]}", f"R_{symbols[2]}", "RP1"]
    ordered = [factors[2], factors[1], factors[0], "RP1"]
    factor_leads = [
        "The coordinate transform from camera 1 to camera 2 is R21=R1·R2·R3·R4. R1 is ",
        ". R2 is ", ". R3 is ", ". R4 is ",
    ]
    for lead, name in zip(factor_leads, ordered):
        tail = ". " if name == "RP1" else ""
        put(_choice("Which factor is this term of R21=R1·R2·R3·R4?", factors, name, "A2"), 5, lead, tail)
    total = sp.simplify(matrices["roll"] * matrices["tilt"] * matrices["pan"] * rp1)
    for number, matrix in enumerate((matrices["roll"], matrices["tilt"], matrices["pan"], rp1), start=1):
        lead = (
            "Before the product, expand each factor with the symbols just chosen. The expansion of R1 is "
            if number == 1 else f"The expansion of R{number} is "
        )
        put(_write_matrix(f"Expansion of R{number} before the product.", matrix), 5, lead, ". ")
    put(_write_matrix("Write the fully expanded rotation R21 from camera 1 to camera 2.", total),
        5, "The fully expanded rotation from camera 1 to camera 2 is R21=", ".")
    if len(questions) != 41:
        raise RuntimeError(f"camera 2 should have 41 blanks, got {len(questions)}.")
    return questions, total


def build(spec: dict) -> dict:
    scene = cm.Scene().declare(spec)
    questions: list[dict] = []

    # Read the declared "wind" up/right senses on camera 1: the world frame is camera 1.
    scene.wind = {"Xw": scene.cam1["X1"], "Yw": scene.cam1["Y1"], "Zw": scene.cam1["Z1"]}
    cam2_questions, r21 = _camera2_from_camera1(scene)
    questions.extend(cam2_questions)

    d_w = _coord_unit(scene.cam1, CAM_AXES[1], "forward")
    point = sp.simplify(scene.length * _coord_unit(scene.cam1, CAM_AXES[1], "up"))
    t21 = sp.simplify(-r21 * C2)
    p_c1 = sp.simplify(point)
    p_c2 = sp.simplify(r21 * point + t21)
    if sp.simplify(_skew(p_c1) * d_w) == sp.zeros(3, 1):
        raise RuntimeError("The line passes through the camera-1 optical center, so the image line vanishes.")
    s1 = cm.camera_layout_permutation(scene.cam1, CAM_AXES[1])
    s2 = cm.camera_layout_permutation(scene.cam2, CAM_AXES[2])
    k1it = sp.simplify((K1 * s1).inv().T)
    k2it = sp.simplify((K2 * s2).inv().T)

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        "Which formula does the line l in the camera-1 image satisfy? d_w is its direction in camera 1.",
        [
            "l = (K1·S1)^(-T) · R21 · [P_c1]_× · d_w",
            "l = (K1·S1)^(-1) · [P_c1]_× · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · d_w + t21",
            "l = [P_c1]_× · (K1·S1)^(-T) · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · R21^T · d_w",
        ],
        "l = (K1·S1)^(-T) · [P_c1]_× · d_w", "A2",
    ), 1, "The line l in the camera-1 image satisfies ", ", where d_w is the direction of this spatial line in camera 1. ")
    put(_choice(
        "Which formula does the line l in the camera-2 image satisfy? d_w is its direction in camera 1.",
        [
            "l = (K2·S2)^(-T) · R21 · [P_c2]_× · d_w",
            "l = (K2·S2)^(-1) · [P_c2]_× · R21 · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · R21 · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · R21^T · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · d_w",
            "l = [P_c2]_× · (K2·S2)^(-T) · R21 · d_w",
        ],
        "l = (K2·S2)^(-T) · [P_c2]_× · R21 · d_w", "A2",
    ), 1, "The line l in the camera-2 image satisfies ", ", where d_w is the direction of this spatial line in camera 1.")
    put(_write_matrix("Write the inverse transpose (K1·S1)^(-T) of the camera-1 effective intrinsic.", k1it),
        2, "Expand (K1·S1)^(-T) by itself and do not multiply it by another symbol. (K1·S1)^(-T)=", ". ")
    put(_write_matrix("Write the inverse transpose (K2·S2)^(-T) of the camera-2 effective intrinsic.", k2it),
        2, "Expand (K2·S2)^(-T) by itself and do not multiply it by another symbol. (K2·S2)^(-T)=", ".")
    put(_choice(
        "What is the form of the translation t21 of camera 2 relative to camera 1?",
        ["t21 = R21·C2", "t21 = −R21^T·C2", "t21 = −R21·C2", "t21 = C2 − R21"],
        "t21 = −R21·C2", "A2",
    ), 3, "The translation t21 of camera 2 relative to camera 1 has the form ", ". ")
    put(_write_vector("Write the translation t21 of camera 2 relative to camera 1.", t21),
        3, "Write t21=", ".")
    put(_choice(
        "What is the formula for the point P_w on the line?",
        ["P_w = L·e_n", "P_w = L·d_w", "P_w = L·[0, 1, 0]^T", "P_w = e_n"],
        "P_w = L·e_n", "A2",
    ), 4, "The line is parallel to the positive optical axis of camera 1 and passes through P_w. e_n is the unit vector of camera 1's positive normal in camera 1, and the distance from P_w to the optical center is L. The formula for P_w is ", ".")
    put(_write_vector("Write the coordinate P_w of that point in camera 1.", point),
        4, "Write P_w=", ". ")
    put(_choice(
        "What is the form of P_c1, the same point in camera 1?",
        ["P_c1 = R21·P_w + t21", "P_c1 = P_w", "P_c1 = R21^T·P_w + t21", "P_c1 = P_w + t21"],
        "P_c1 = P_w", "A2",
    ), 5, "The point P_c1 in camera 1 has the form ", ". ")
    put(_write_vector("Write P_c1 in camera 1.", p_c1),
        5, "Write P_c1=", ". ")
    put(_choice(
        "What is the form of P_c2, the same point in camera 2?",
        ["P_c2 = R21·P_w − t21", "P_c2 = R21^T·P_w + t21", "P_c2 = R21·P_w + t21", "P_c2 = P_w + t21"],
        "P_c2 = R21·P_w + t21", "A2",
    ), 5, "The point P_c2 in camera 2 has the form ", ". ")
    put(_write_vector("Write P_c2 in camera 2.", p_c2),
        5, "Write P_c2=", ".")
    put(_write_matrix("Write the skew matrix [P_c1]_×.", sp.simplify(_skew(p_c1))),
        6, "Expand [P_c1]_× by itself and do not multiply it by another symbol. [P_c1]_×=", ". ")
    put(_write_matrix("Write the skew matrix [P_c2]_×.", sp.simplify(_skew(p_c2))),
        6, "Expand [P_c2]_× by itself and do not multiply it by another symbol. [P_c2]_×=", ".")

    put(_write_vector("Write the direction d_w of this line.", d_w),
        1,
        "The line is parallel to the positive optical axis of camera 1. Its direction is d_w. "
        "The image line is l=(K1·S1)^(-T)·[P_c1]_×·d_w in camera 1 and "
        "l=(K2·S2)^(-T)·[P_c2]_×·R21·d_w in camera 2. d_w=",
        ".")

    put(_choice(
        "The essential matrix is?",
        ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
        "E = skew(t21)*R21", "A2",
    ), 1, "The essential matrix is ", ".")
    put(_choice(
        "The fundamental matrix is?",
        ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)",
         "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
        "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2",
    ), 2, "The fundamental matrix is ", ".")
    put(_choice(
        "The epipolar constraint is?",
        ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
         "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
        "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2",
    ), 3, "The epipolar constraint is ", ".")
    d1 = sp.simplify((K1 * s1).inv() * sp.Matrix([u1, v1, 1]))
    d2 = sp.simplify((K2 * s2).inv() * sp.Matrix([u2, v2, 1]))
    put(_write_vector(
        "The homogeneous image coordinate in camera 1 is [u1,v1,1], and d1=(K1·S1)^(-1)·[u1,v1,1]^T.", d1),
        4, "The homogeneous image coordinate in camera 1 is [u1,v1,1]. The ray direction is d1=(K1·S1)^(-1)·[u1,v1,1]^T, where d1=", ".")
    put(_write_vector(
        "The homogeneous image coordinate in camera 2 is [u2,v2,1], and d2=(K2·S2)^(-1)·[u2,v2,1]^T.", d2),
        5, "The homogeneous image coordinate in camera 2 is [u2,v2,1]. The ray direction is d2=(K2·S2)^(-1)·[u2,v2,1]^T, where d2=", ".")
    put(_choice(
        "The target point in camera 1 lies along the ray. X_c1 = ?",
        ["X_c1 = λ1·d1", "X_c1 = d1 + λ1", "X_c1 = λ1 + d1", "X_c1 = d1 − λ1"],
        "X_c1 = λ1·d1", "A2",
    ), 6, "The target point X_c1 in camera 1 lies along the ray d1, where λ1 is the depth. X_c1=", ".")
    put(_choice(
        "Both cameras see the same target point. What is the constraint?",
        ["λ2·d2 = R21·λ1·d1 + t21", "λ2·d2 = R21·λ1·d1 − t21",
         "λ2·d2 = R21^T·λ1·d1 + t21", "λ1·d1 = R21·λ2·d2 + t21"],
        "λ2·d2 = R21·λ1·d1 + t21", "A2",
    ), 7, "Both cameras see the same target point. The transform from camera 1 to camera 2 is X_c2=R21·X_c1+t21, so the constraint is ", ".")
    put(_choice(
        "Cross both sides by which vector to eliminate λ2?",
        ["d2", "d1", "R21·d1", "t21"],
        "d2", "A2",
    ), 8, "Cross both sides of the constraint by ", ", eliminate λ2, and obtain d2×(λ1·R21·d1+t21)=0.")
    put(_choice(
        "Under the least-squares criterion, and with d2×R21·d1 ≠ 0, "
        "which λ1 minimizes the squared residual of d2×(λ1·R21·d1+t21)=0?",
        ["λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = −(d2×t21) / (d2×R21·d1)",
         "λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²"],
        "λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²", "A2",
    ), 9,
        "Under the least-squares criterion, and with d2×R21·d1 ≠ 0, the λ1 that minimizes the squared residual of d2×(λ1·R21·d1+t21)=0 is ",
        ".")
    put(_choice(
        "Given X_c1 in camera 1, what is its world coordinate X_w?",
        ["X_w = X_c1", "X_w = R21^T·(X_c1 − t21)", "X_w = R21·X_c1 + t21", "X_w = X_c1 + t21"],
        "X_w = X_c1", "A2",
    ), 10, "Given X_c1 in camera 1, and because camera 1 is the world frame, its world coordinate is ", ".")

    bounds = [
        (41, "camera_attitude"),
        (55, "projection"),
        (56, "line_direction"),
        (66, "epipolar"),
    ]
    if len(questions) != bounds[-1][0]:
        raise RuntimeError(f"vision basics should have {bounds[-1][0]} blanks, got {len(questions)}.")

    def segment_of(index: int) -> str:
        for bound, name in bounds:
            if index <= bound:
                return name
        return "epipolar"

    for index, item in enumerate(questions, start=1):
        item["local_id"] = index
        item["id"] = str(index)
        item["process"] = "vision basics"
        item["segment"] = segment_of(index)
        item["format_code"] = item.pop("format")
    return {
        "spec": spec,
        "declaration": _declaration(spec, scene),
        "questions": questions,
        "n_blanks": len(questions),
    }


def _frame_phrase(title: str, spec: dict, x_name: str, y_name: str) -> str:
    return (
        f"{title} is now {HAND_CN[spec['chirality']]}; "
        f"{x_name} is {ROLE_CN[spec[x_name]['along']]} and points {POINT_CN[spec[x_name]['points']]}, "
        f"{y_name} is {ROLE_CN[spec[y_name]['along']]} and points {POINT_CN[spec[y_name]['points']]}"
    )


def _basic_key(spec: dict) -> str:
    angles = {name: spec["angles"][name] for name in BASIC_ANGLES}
    payload = {
        "camera1": spec["camera1"],
        "camera2": spec["camera2"],
        "angles": angles,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _change_text(spec: dict, source: str) -> str:
    phrases = []
    for key, x_name, y_name in (
        ("camera1", "X1", "Y1"),
        ("camera2", "X2", "Y2"),
    ):
        if spec[key] != STANDARD_SPEC[key]:
            phrases.append(_frame_phrase(FRAME_TITLE[key], spec[key], x_name, y_name))
    for name in BASIC_ANGLES:
        if spec["angles"][name] != STANDARD_SPEC["angles"][name]:
            phrases.append(ANGLE_CHANGE[name])
    if not phrases:
        return "Shared default. Camera 1 and camera 2 keep the default axes, and pitch, azimuth, and roll keep the default positive sense."
    origin = f"From {source}. " if source else ""
    return origin + "Relative to the shared default, only the two-camera differences remain: " + "; ".join(phrases) + "."


def _declaration(spec: dict, scene: cm.Scene) -> str:
    hand_word = {"right": "right-handed", "left": "left-handed"}

    def frame_block(title: str, names: tuple, frame: dict, frame_spec: dict) -> str:
        hand = hand_word[frame_spec.get("chirality", "left")]
        titled = title[0].upper() + title[1:]
        lines = [f"{titled} ({hand}):"]
        for name in names:
            vector = frame[name]
            role = cm.role_of(vector)
            pointing = POINT_CN[DIR_NAME[vector]]
            where = "as declared" if name in frame_spec else "inferred from chirality"
            lines.append(f"- {name}: along the {ROLE_CN[role]} direction, pointing {pointing} ({where}).")
        return "\n".join(lines)

    pitch = "positive" if scene.angle_toward["tilt"] == "wind_up" else "negative"
    yaw = "positive" if scene.angle_toward["pan"] == "wind_right" else "negative"
    roll = "positive" if scene.angle_toward["roll"] == "camera_down" else "negative"
    return "\n".join([
        "[Declaration]",
        "",
        "Fixed conventions",
        "- Coordinate transforms are passive. Attitude rotations use an intrinsic moving-axis sequence.",
        "- Camera imaging uses the pinhole model x = (K·S)·[R|t]·X. An image line is l = (K·S)^(-T)·n, where n is the normal of the plane spanned by the optical center and the spatial line.",
        "- The epipolar constraint is x2^T·F·x1 = 0, with F = (K2·S2)^(-T)·E·(K1·S1)^(-1) and E = [t21]_×·R21.",
        "- The world frame is camera 1: the optical center of camera 1 is the world origin, and the three camera-1 axes are the world axes. The rotation of camera 1 relative to the world is the identity and the translation is zero. Only the rotation R21 and translation t21 from camera 1 to camera 2 are required.",
        "- A line lies in camera 1. It is parallel to the positive optical axis of camera 1. d_w is the unit vector of that direction in camera 1. e_n is the unit vector of camera 1's positive normal in camera 1. The line passes through P_w = L·e_n and does not pass through the optical center.",
        "- Every quantity is symbolic: L, ψ2, θ2, φ2, u1, v1, u2, v2, λ1. The center of camera 2 in camera 1 (the world frame) is C2=(x2,y2,z2).",
        "- The intrinsics are K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]] and K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]. K acts in the order [lateral positive (right), normal positive (up), axial positive (forward)]. S reorders that camera's coordinates into this order, including sign flips. The effective intrinsic is K·S.",
        "",
        "Absolute directions",
        "The world and a camera each have three absolute axes. The positive senses are the same: axial positive is forward, normal positive is up, and lateral positive is right. The six absolute directions are the two senses of these three axes.",
        "Axial positive, normal positive, and lateral positive are these absolute senses, not the positive senses of the coordinate axes. A coordinate-axis declaration only assigns each axis to an absolute axis and a sense. That sense may agree with the absolute positive direction or oppose it.",
        "The two frames share one orientation rule: forward × right = down. Looking along forward, right is the right-hand side and down is vertically downward. This rule does not change with the definition.",
        "The world's three axes and six directions:",
        "- Axial: the world's fore-aft axis. The positive sense is forward. Rearward is the opposite sense.",
        "- Normal: the world's vertical axis. The positive sense is up, upward is vertically up, and downward is vertically down.",
        "- Lateral: the world's horizontal transverse axis. The positive sense is right. Left is the opposite sense.",
        "The camera's three axes and six directions, relative to the camera:",
        "- Axial: the optical axis. The positive sense is forward, along the viewing direction. Rearward is the opposite sense.",
        "- Normal: the vertical direction of the image. The positive sense is up. Down is the opposite sense.",
        "- Lateral: the horizontal direction of the image. The positive sense is right. Left is the opposite sense.",
        "",
        "Scene",
        "Two calibrated, synchronously exposed cameras observe one fixed line. Camera 1 is the reference camera, and its frame is the world frame. The optical axis follows the camera axial direction, the image horizontal follows the camera lateral direction, and the image vertical follows the camera normal direction.",
        "There are two tasks. The first maps the line into both camera images and writes the image lines. The second is given the same target point in both images and, from the stereo model, recovers it in camera 1 (the world frame).",
        "The subtasks are: from the placement of camera 2 relative to camera 1, write the rotation R21 and translation t21 from camera 1 to camera 2; from the line direction and one point on the line, write its expression in both images; from the target point in both images, recover its camera-1 coordinate through the rays, the epipolar constraint, and the depth. Camera 1 is the world frame, so that coordinate is the world coordinate.",
        "",
        "Angles",
        "Camera pitch θ2: the angle between the longitudinal axis of camera 2 and the horizontal plane of camera 1. "
        f"The angle is {pitch} when the longitudinal axis of camera 2 rises above camera 1.",
        "Camera azimuth ψ2: the angle between the horizontal projection of the longitudinal axis of camera 2 and the axial forward reference axis of camera 1. "
        f"The angle is {yaw} when the longitudinal axis of camera 2 deflects to the right of camera 1.",
        "Camera roll φ2: the angle between the normal of camera 2 and the vertical plane through its longitudinal axis. "
        f"The angle is {roll} when camera 2 rolls to its right.",
        "Camera 2 uses the intrinsic order azimuth, pitch, and roll, written ψ2, θ2, φ2.",
        "",
        "Frames",
        "The axis along the axial direction and pointing forward is the reference axis. The other two axes are bound between the normal and lateral directions, and the binding must match the chirality of the frame.",
        frame_block("world frame (= camera-1 frame)", CAM_AXES[1], scene.cam1, spec["camera1"]),
        "",
        frame_block("camera-2 frame", CAM_AXES[2], scene.cam2, spec["camera2"]),
        "",
        "Background",
        "Axial positive, normal positive, and lateral positive are forward, up, and right. Camera azimuth rotates about the absolute normal axis, camera pitch about the absolute lateral axis, and camera roll about the absolute axial axis. The positive sense of each angle is the sense in the declaration. Coordinate transforms are passive. Camera attitude uses an intrinsic moving-axis sequence.",
    ])


def _samples() -> list[dict]:
    source_path = SETS["cross.wind_vision"]["samples"]
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [{
        "source": "",
        "spec": json.loads(json.dumps(STANDARD_SPEC)),
    }]
    seen = {_basic_key(selected[0]["spec"])}
    for row in rows:
        key = _basic_key(row["spec"])
        if key in seen:
            print(f"skip {row['id']}: world and cameras match an earlier sample")
            continue
        seen.add(key)
        selected.append({"source": row["id"], "spec": row["spec"]})
    samples = []
    for index, item in enumerate(selected, start=1):
        samples.append({
            "id": f"basic-{index:02d}",
            "source": item["source"] or "standard",
            "title": "vision basics" if item["source"] else "vision basics, shared default",
            "change": _change_text(item["spec"], item["source"]),
            "spec": item["spec"],
        })
    return samples


def _check(record: dict) -> None:
    blob = record["declaration"] + record["change"]
    for item in record["questions"]:
        blob += item["prompt"] + item.get("lead", "") + item.get("tail", "")
        blob += "".join(item.get("options") or [])
    bad = [word for word in FORBIDDEN if word in blob]
    if bad:
        raise RuntimeError(f"{record['id']} still uses terms outside vision basics: {bad}")
    for item in record["questions"]:
        if item["options"] and item["answer"] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise RuntimeError(f"{record['id']} blank {item['id']} answer is not an option letter.")


def _titles() -> dict:
    return {
        "frame": "Absolute directions and coordinate axes",
        "camera_attitude": "Camera attitude",
        "projection": "Imaging and a point on the line",
        "line_direction": "Direction of an image line",
        "epipolar": "Epipolar geometry and triangulation",
    }


def _export(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    intro = (
        "This file is the review draft for vision basics. The world frame is camera 1, and there is no object frame. "
        "The line is parallel to the positive optical axis of camera 1, passes through the point a distance L along the positive normal from the optical center, and does not pass through the optical center. "
        "The pose of camera 2 relative to camera 1 is R21, t21. A target point in the images is recovered in camera 1 by stereo rays. "
        "The label and answer inside a slot are for review; the blank shown to a model omits both."
    )
    titles = _titles()
    for index, row in enumerate(rows, start=1):
        (OUT_DIR / f"sample_{index:02d}.md").write_text(
            rx._sample_markdown(row, index, intro=intro, titles=titles), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    lines = [
        "# vision basics",
        "",
        "These items are simplified from the stereo pose questions. They cover the pinhole model, camera pose, line projection, stereo extrinsics, and recovering an image point.",
        "There is no object frame, no mechanism, and no installation or object attitude. The world frame is camera 1. The pose of camera 2 relative to camera 1 is R21, t21.",
        "The geometric object is a line in camera 1: parallel to the positive optical axis, through the point a distance L along the positive normal from the optical center, and not through the optical center.",
        "The other task gives the same target point in both images and recovers its coordinate in camera 1 by stereo rays.",
        "",
        f"Each sample has {rows[0]['n_blanks']} blanks. The formal input format is in 00_injection_format.md. The end-to-end question is in [end to end](end_to_end.md).",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [Data {index:02d}](sample_{index:02d}.md): {row['id']}, {row['change']}")
    lines.append("")
    (OUT_DIR / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "00_injection_format.md").write_text(rx._format_document(example), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


def main() -> None:
    rows = []
    for sample in _samples():
        built = build(sample["spec"])
        record = {
            "id": sample["id"],
            "source": sample["source"],
            "title": sample["title"],
            "change": sample["change"],
            "spec": {
                "wind": sample["spec"]["wind"],
                "camera1": sample["spec"]["camera1"],
                "camera2": sample["spec"]["camera2"],
                "angles": {name: sample["spec"]["angles"][name] for name in BASIC_ANGLES},
            },
            "declaration": built["declaration"],
            "questions": built["questions"],
            "n_blanks": built["n_blanks"],
        }
        _check(record)
        rows.append(record)
        print(f"ok {record['id']} {record['n_blanks']} blanks  {record['change']}", flush=True)
    DATA_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {DATA_PATH}")
    _export(rows)


if __name__ == "__main__":
    main()
