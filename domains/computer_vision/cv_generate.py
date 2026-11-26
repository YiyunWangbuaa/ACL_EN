# -*- coding: utf-8 -*-
"""Stereo-schlieren attitude questions in the wind-tunnel process style.

All quantities are symbolic: α, β, γ, L, ψ1, θ1, φ1, ψ2, θ2, φ2, fx/fy/cx/cy
and the camera positions. Each rotation matrix is preceded by the step-by-step
process: direction bias -> axis -> rotation sign -> matrix, then the Euler order,
the transform convention and the total matrix (written out in full).
"""
from __future__ import annotations

import sympy as sp

from domains.computer_vision import cv_model as cm

# Choice labels. Axis bullets use the shorter point words in POINT_WORD.
DIR_CN = {"forward": "forward", "rearward": "rearward", "right": "right",
          "left": "left", "down": "downward", "up": "upward"}
DIR_NAME = {value: key for key, value in cm.DIRECTIONS.items()}
POINT_WORD = {"forward": "forward", "rearward": "rearward", "right": "right",
              "left": "left", "down": "down", "up": "up"}
SIX = ["forward", "rearward", "right", "left", "downward", "upward"]
THREE_SIGN = ["positive", "negative"]
EULER = ["angle of attack", "sideslip", "roll"]
CAM_EULER = ["azimuth", "pitch", "roll"]
AXIS_ALL = ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb",
            "X1", "Y1", "Z1", "X2", "Y2", "Z2"]
AXIS_OPTIONS = AXIS_ALL + ["-" + name for name in AXIS_ALL]

fx1, fy1, cx1, cy1 = sp.symbols("fx1 fy1 cx1 cy1")
fx2, fy2, cx2, cy2 = sp.symbols("fx2 fy2 cx2 cy2")
x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2")
u1, v1, u2, v2 = sp.symbols("u1 v1 u2 v2")

K1 = sp.Matrix([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
K2 = sp.Matrix([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
C1 = sp.Matrix([x1, y1, z1])
C2 = sp.Matrix([x2, y2, z2])

ROLE_OF = {"pan": "normal", "tilt": "lateral", "roll": "axial",
           "attack": "lateral", "sideslip": "normal", "bank": "axial"}
_ROLE_POINTING = {
    "normal": ("normal (up)", "up"),
    "lateral": ("lateral (right)", "right"),
    "axial": ("axial (forward)", "forward"),
}
SUBJECT = {"pan": "optical axis", "tilt": "optical axis", "roll": "right edge of the image",
           "attack": "object nose", "sideslip": "object nose", "bank": "object right edge"}
INDEX = {"pan": 0, "tilt": 1, "roll": 2, "attack": 0, "sideslip": 1, "bank": 2}

FRAME_NAMES = {
    "wind": ("Xw", "Yw", "Zw"),
    "body": ("Xb", "Yb", "Zb"),
    "camera1": ("X1", "Y1", "Z1"),
    "camera2": ("X2", "Y2", "Z2"),
}
FRAME_TITLE = {
    "wind": "world frame",
    "body": "object frame",
    "camera1": "camera-1 frame",
    "camera2": "camera-2 frame",
}
ROLE_POINTS = {
    "axial": ("forward", "rearward"),
    "normal": ("up", "down"),
    "lateral": ("left", "right"),
}
ANGLE_FLIP = {
    "pan": "wind_left",
    "tilt": "wind_down",
    "roll": "camera_up",
    "attack": "body_down",
    "sideslip": "body_right",
    "bank": "body_up",
}
ANGLE_ZH = {
    "pan": "Camera azimuth is positive when the optical axis deflects to the left of the world.",
    "tilt": "Camera pitch is positive when the optical axis deflects downward in the world.",
    "roll": "Camera roll is positive when the right edge of the image deflects upward in the camera.",
    "attack": "Mechanism pitch is positive when the object axis deflects downward in the world.",
    "sideslip": "Mechanism azimuth is positive when the object axis deflects to the right of the world.",
    "bank": "Mechanism roll is positive when rotation about the object's absolute axial direction (forward) opposes the right-hand rule.",
}
FACTORS = ["wind", "body", "camera1", "camera2", *ANGLE_FLIP]


def _frame(x_name: str, y_name: str, x_role: str, x_points: str,
           y_role: str, y_points: str, chirality: str = "left") -> dict:
    return {
        "chirality": chirality,
        x_name: {"along": x_role, "points": x_points},
        y_name: {"along": y_role, "points": y_points},
    }


STANDARD_SPEC = {
    "wind": _frame("Xw", "Yw", "axial", "forward", "lateral", "right"),
    "body": _frame("Xb", "Yb", "axial", "forward", "lateral", "right"),
    "camera1": _frame("X1", "Y1", "lateral", "right", "normal", "up"),
    "camera2": _frame("X2", "Y2", "lateral", "right", "normal", "up"),
    "angles": dict(cm.DEFAULT_TOWARD),
}


def _alternate_free_frames(key: str) -> list[dict]:
    """47 legal frames other than the standard one. Two declared axes fix the third."""
    names = FRAME_NAMES[key]
    x_name, y_name, _z_name = names
    standard = cm.build_frame(key, names, STANDARD_SPEC[key])
    found = []
    for chirality in ("right", "left"):
        for x_role, points in ROLE_POINTS.items():
            for x_points in points:
                for y_role, y_points_all in ROLE_POINTS.items():
                    if y_role == x_role:
                        continue
                    for y_points in y_points_all:
                        spec = _frame(x_name, y_name, x_role, x_points, y_role, y_points, chirality)
                        if cm.build_frame(key, names, spec) == standard:
                            continue
                        found.append(spec)
    if len(found) != 47:
        raise RuntimeError(f"{key} has {len(found)} non-standard frames, expected 47.")
    return found


def _alternate_wind_frames() -> list[dict]:
    """Xw stays on the axial direction and points forward. Yw and Zw still vary."""
    names = FRAME_NAMES["wind"]
    x_name, y_name, _z_name = names
    standard = cm.build_frame("wind", names, STANDARD_SPEC["wind"])
    found = []
    for chirality in ("right", "left"):
        for y_role, y_points_all in ROLE_POINTS.items():
            if y_role == "axial":
                continue
            for y_points in y_points_all:
                spec = _frame(x_name, y_name, "axial", "forward", y_role, y_points, chirality)
                if cm.build_frame("wind", names, spec) == standard:
                    continue
                found.append(spec)
    if len(found) != 7:
        raise RuntimeError(f"wind has {len(found)} non-standard frames with Xw fixed, expected 7.")
    return found


def alternate_frames(key: str) -> list[dict]:
    if key == "wind":
        return _alternate_wind_frames()
    return _alternate_free_frames(key)


def _m(matrix) -> str:
    return "[" + ", ".join("[" + ", ".join(sp.sstr(v) for v in row) + "]" for row in matrix.tolist()) + "]"


def _choice(prompt: str, options: list[str], correct: str, label: str) -> dict:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return {"prompt": prompt, "label": label, "format": "choice",
            "options": list(options), "answer": letters[options.index(correct)]}


def _write_matrix(prompt: str, matrix: sp.Matrix, label: str = "A2") -> dict:
    return {"prompt": prompt, "label": label, "format": "matrix",
            "options": [], "answer": _m(matrix)}


def _v(matrix: sp.Matrix) -> str:
    cells = [sp.sstr(matrix[index, 0]) for index in range(matrix.rows)]
    return "[" + ", ".join(cells) + "]"


def _write_vector(prompt: str, matrix: sp.Matrix, label: str = "A2") -> dict:
    return {"prompt": prompt, "label": label, "format": "vector",
            "options": [], "answer": _v(matrix)}


def _nine_axes() -> list[str]:
    return ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb",
            "X1", "Y1", "Z1", "X2", "Y2", "Z2"]


def _axis_cn(frame: dict, role: str) -> str:
    name, _ = cm.axis_on(frame, role)
    return name


def _axis_pointing(frame: dict, direction: str) -> str:
    vec = cm.DIRECTIONS[direction]
    for name, value in frame.items():
        if value == vec:
            return name
        if value == cm._neg(vec):
            return "-" + name
    return ""


def _bias(sign: int) -> str:
    return "positive" if sign > 0 else "negative"


_BODY_AXES = ("Xb", "Yb", "Zb")
_WIND_AXES = ("Xw", "Yw", "Zw")
_PITCH_WAY = {"body_up": "upward", "body_down": "downward"}
_AZIMUTH_WAY = {"body_left": "the left", "body_right": "the right"}
_FRAME_PAIR = ["object frame", "world frame"]
_CONVERSION = [
    "object frame to object frame", "object frame to world frame",
    "world frame to object frame", "world frame to world frame",
]
_DIRECTED = (
    ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb"]
    + ["-Xw", "-Yw", "-Zw", "-Xb", "-Yb", "-Zb"]
)
_BODY_AXIS_OPTIONS = ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb"]
_FACTORS = ["RP1", "RP2", "R_θ_M", "R_ψ_M", "R_φ_M"]


def _mechanism_definition(scene: cm.Scene) -> list[str]:
    """State the positive senses. The Euler order is inferred from the mechanism."""
    attack = _PITCH_WAY[scene.angle_toward["attack"]]
    sideslip = _AZIMUTH_WAY[scene.angle_toward["sideslip"]]
    agrees = scene.angle_toward["bank"] == "body_down"
    roll_sense = "agrees with" if agrees else "opposes"
    return [
        "Mechanism pitch θ_M: the longitudinal mechanism moves in a vertical plane and is treated as a rotation about the lateral axis of the world frame. "
        "It is the angle between the object longitudinal axis and the horizontal plane. "
        f"The angle is positive when that axis deflects {attack} in the world.",
        "Mechanism azimuth ψ_M: the lateral mechanism moves in the plane that is perpendicular to the vertical plane and contains the longitudinal mechanism, so it is driven from the longitudinal mechanism. "
        "The angle is between the object longitudinal axis and the vertical plane. "
        f"It is positive when that axis deflects toward {sideslip} of the world.",
        "Mechanism roll φ_M: the rotation of the object about the lateral mechanism. "
        "When the angle is positive, rotation about the object's absolute axial direction (forward) "
        f"{roll_sense} the right-hand rule.",
        "Only the upright installation of the object is considered.",
    ]


def _active_matrix(index: int, angle) -> sp.Matrix:
    """Active rotation of a coordinate vector about axis 0, 1, or 2.

    Used only to decide the algebraic sign. The cloze matrix is the passive
    transpose: a positive angle about Y is
    [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]].
    """
    c, s = sp.cos(angle), sp.sin(angle)
    if index == 0:
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if index == 1:
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if index == 2:
        return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(index)


def _passive_matrix(index: int, angle) -> sp.Matrix:
    """Passive change of coordinates for the same rotation: p_new = R * p_old."""
    return _active_matrix(index, angle).T


def _algebraic_sign_named(frame: dict, names: tuple[str, ...], axis_name: str, moving, toward) -> int:
    """+1 when a positive active rotation about that coordinate axis moves `moving` toward `toward`."""
    index = names.index(axis_name)
    basis = cm.basis(frame, names)
    coord = sp.simplify(basis.T * sp.Matrix(moving))
    symbol = sp.Symbol("t")
    image = _active_matrix(index, symbol) * coord
    disp = basis * sp.Matrix([sp.diff(image[i], symbol).subs(symbol, 0) for i in range(3)])
    dot = sum(int(disp[i]) * toward[i] for i in range(3))
    if dot == 0:
        raise RuntimeError(f"{axis_name} positive rotation does not reach the declared direction.")
    return 1 if dot > 0 else -1


def _assert_passive_opposes(frame: dict, names: tuple[str, ...], moving, toward, matrix, symbol, axis_name: str) -> None:
    """The passive matrix must be the textbook form of the right-hand signed angle.

    In a right-handed basis a fixed vector then shifts opposite the declared
    motion. A left-handed basis reverses that component shift.
    """
    basis = cm.basis(frame, names)
    coord = sp.simplify(basis.T * sp.Matrix(moving))
    image = sp.simplify(matrix * coord)
    deriv = [sp.diff(image[i], symbol).subs(symbol, 0) for i in range(3)]
    disp = basis * sp.Matrix(deriv)
    dot = sp.simplify(sum(disp[i] * toward[i] for i in range(3)))
    if not dot.is_number or int(dot) * _orientation(frame, names) >= 0:
        raise RuntimeError(f"{axis_name} passive matrix disagrees with the right-hand angle (dot {dot}).")


def _orientation(frame: dict, names: tuple[str, ...]) -> int:
    """+1 when the named axes are right-handed in the embedding, otherwise -1."""
    triple = cm.basis(frame, names).det()
    if triple == 0:
        raise RuntimeError(f"{names} are coplanar, so chirality is undefined.")
    return 1 if triple > 0 else -1


def signed_rotation(frame: dict, names: tuple[str, ...], axis_name: str, moving, toward, symbol):
    """Right-hand sign about the positive axis, and the passive matrix of that signed angle.

    A positive sense writes the textbook passive matrix of a positive angle. A
    negative sense writes the same matrix with the angle negated. A left-handed
    basis does not get a second sign flip inside the matrix.
    """
    component = _algebraic_sign_named(frame, names, axis_name, moving, toward)
    handed = cm.sense(frame[axis_name], moving, toward)
    if component != _orientation(frame, names) * handed:
        raise RuntimeError(f"{axis_name} positive component disagrees with the right-hand screw.")
    matrix = sp.simplify(_passive_matrix(names.index(axis_name), handed * symbol))
    _assert_passive_opposes(frame, names, moving, toward, matrix, symbol, axis_name)
    return handed, matrix


def _placed(item: dict, paragraph: int, lead: str, tail: str = "") -> dict:
    item["paragraph"] = paragraph
    item["lead"] = lead
    item["tail"] = tail
    return item


def _body_unit(frame: dict, role: str) -> sp.Matrix:
    """Unit components of the declared body axis that lies on this absolute role."""
    name, _vector = cm.axis_on(frame, role)
    component = sp.zeros(3, 1)
    component[{"Xb": 0, "Yb": 1, "Zb": 2}[name]] = 1
    return component


def _body_mechanism_questions(scene: cm.Scene):
    """Object attitude in the world frame. Only the upright installation is used."""
    questions: list[dict] = []
    wind_basis = cm.basis(scene.wind, _WIND_AXES)
    body_basis = cm.basis(scene.body, _BODY_AXES)
    rp1 = sp.simplify(body_basis.T * wind_basis)
    rp2 = sp.eye(3)

    questions.extend([
        _placed(_choice(
            "Which conversion is used to derive the coordinate transform from the positive attitude angles?",
            _CONVERSION, "world frame to object frame", "C2",
        ), 1, "Initially the object is level. The attitude change produces three attitude angles θ_M, ψ_M, and φ_M and changes the object attitude in the world frame. For convenience, derive the coordinate transform from the positive attitude angles by "),
        _placed(_choice(
            "Which operation turns the world-to-object matrix into the object-to-world transform?",
            ["transpose", "invert", "negate", "leave unchanged"], "transpose", "C2",
        ), 1, ", then apply "),
        _placed(_choice(
            "After transposing the world-to-object matrix, which conversion is obtained?",
            _CONVERSION, "object frame to world frame", "C2",
        ), 1, ", which gives the transform from ", " and the two object axes in the world frame."),
    ])

    wind_leads = [
        "The absolute normal (up), axial (forward), and lateral (right) directions of the world frame correspond to ",
        ", ", ", and ",
    ]
    body_leads = [
        ". The same three directions of the object frame correspond to ",
        ", ", ", and ",
    ]
    for lead, direction in zip(wind_leads, ("up", "forward", "right")):
        questions.append(_placed(_choice(
            f"Which axis is the world frame's absolute {direction} direction?",
            _DIRECTED, _axis_pointing(scene.wind, direction), "C1",
        ), 2, lead))
    for lead, direction in zip(body_leads, ("up", "forward", "right")):
        questions.append(_placed(_choice(
            f"Which axis is the object frame's absolute {direction} direction?",
            _DIRECTED, _axis_pointing(scene.body, direction), "C1",
        ), 2, lead))
    questions.extend([
        _placed(_write_matrix("Write the preliminary transform RP1 from the world frame to the level object frame.", rp1),
                2, ". The preliminary transform RP1 from the world frame to the level object frame is therefore "),
        _placed(_choice("Which frame's axes does RP1 make agree with its absolute directions?", _FRAME_PAIR, "object frame", "C2"),
                2, ". That matrix makes the axes of "),
        _placed(_choice("Which frame is the comparison for RP1?", _FRAME_PAIR, "world frame", "C2"),
                2, " agree with the absolute directions in the same way as "),
        _placed(_choice("What is the object installation?", ["upright", "reversed"], "upright", "C1"),
                2, ". The object installation is "),
        _placed(_choice("Are the object's absolute normal (up) and lateral (right) the same as the world's?",
                        ["same", "opposite"], "same", "A1"),
                2, ". The object's absolute normal (up) and lateral (right) are ",
                " the world's absolute normal and lateral directions. "),
        _placed(_write_matrix("Write the second preliminary matrix RP2.", rp2),
                2, "A second preliminary matrix RP2 is therefore "),
        _placed(_choice("Which frame's axes does RP2 align?", _FRAME_PAIR, "object frame", "C2"),
                2, ". It makes the axes of "),
        _placed(_choice("Which frame supplies the directions that RP2 matches?", _FRAME_PAIR, "world frame", "C2"),
                2, " point the same way as ", "."),
    ])

    questions.extend([
        _placed(_choice("From the attitude definition, what is the first rotation angle?", EULER, "angle of attack", "A1"),
                3, "From the attitude definition, the first rotation angle is "),
        _placed(_choice("What is the second rotation angle?", EULER, "sideslip", "A1"),
                3, ". The second is "),
        _placed(_choice("What is the third rotation angle?", EULER, "roll", "A1"),
                3, ". The third is ", "."),
    ])

    specs = (
        ("angle of attack", "α_M", "attack", "lateral", cm.FORWARD, 4),
        ("sideslip", "β_M", "sideslip", "normal", cm.FORWARD, 5),
        ("roll", "γ_M", "bank", "axial", cm.RIGHT, 6),
    )
    matrices = []
    for _title, symbol_name, key, role, moving, paragraph in specs:
        toward_name = scene.angle_toward[key]
        if key == "bank":
            toward_vec = cm.UP if toward_name == "body_up" else cm.DOWN
            hand = "opposite" if toward_name == "body_up" else "same"
        else:
            toward_vec = cm.DIRECTIONS[_TOWARD_KEY[toward_name]]
            nose_side = _PITCH_WAY[toward_name] if key == "attack" else _AZIMUTH_WAY[toward_name]
        axis_name, _axis_vec = cm.axis_on(scene.body, role)
        symbol = sp.Symbol(symbol_name)
        sign, matrix = signed_rotation(scene.body, _BODY_AXES, axis_name, moving, toward_vec, symbol)
        matrices.append(matrix)
        if key == "bank":
            questions.append(
                _placed(_choice(
                    "When φ_M is positive, is rotation about the object's absolute axial direction (forward) the same as the right-hand rule or opposite?",
                    ["same", "opposite"], hand, "C1"),
                        paragraph, "When φ_M is positive, rotation about the object's absolute axial direction (forward) is ",
                        " the right-hand rule. "),
            )
            axis_lead = "The rotation is about axis "
            matrix_lead = f", through the angle {symbol_name}. The rotation matrix R_{symbol_name} is "
        else:
            if key == "attack":
                opening = (
                    f"When {symbol_name} is positive, the object longitudinal axis deflects {nose_side} in the world. "
                    "The object uses "
                )
            else:
                opening = (
                    f"When {symbol_name} is positive, the object longitudinal axis deflects toward {nose_side} of the world. "
                    "The object uses "
                )
            questions.append(
                _placed(_choice(
                    f"When {symbol_name} is positive, which installation does the object use?",
                    ["upright", "inverted"], "upright", "C1"),
                        paragraph, opening),
            )
            axis_lead = ", rotating about "
            matrix_lead = f", through the angle {symbol_name}. The rotation matrix R_{symbol_name} is "
        questions.append(
            _placed(_choice(
                f"This is the level pose of which frame, rotating through {symbol_name}?",
                _FRAME_PAIR, "object frame", "C2"),
                    paragraph,
                    "This is the level pose of " if key == "bank" else ". This is the level pose of ",
                    ". " if key == "bank" else ""),
        )
        if key == "bank":
            questions.append(_placed(_choice(
                "Which axis is the object's absolute axial direction (forward)?",
                _DIRECTED, _axis_pointing(scene.body, "forward"), "C1",
            ), paragraph, "The object's absolute axial direction (forward) corresponds to ", ". "))
        sense_lead = ". The sense about that axis is "
        questions.extend([
            _placed(_choice(f"Which object-frame axis does {symbol_name} rotate about?", _BODY_AXIS_OPTIONS, axis_name, "C1"),
                    paragraph, axis_lead),
            _placed(_choice("What is the sense about that axis?", ["positive", "negative", "indeterminate"], _bias(sign), "A1"),
                    paragraph, sense_lead),
            _placed(_write_matrix(f"Write the rotation matrix R_{symbol_name}.", matrix),
                    paragraph, matrix_lead, "."),
        ])

    questions.extend([
        _placed(_choice("Does this coordinate conversion use active or passive?", ["active", "passive"], "passive", "C2"),
                7, "This process converts coordinates, so it uses the ", " convention. "),
        _placed(_choice("Is the object frame moving axes or fixed axes?", ["moving axes", "fixed axes"], "moving axes", "C2"),
                7, "The object frame is "),
        _placed(_choice("Does the rotation use intrinsic or extrinsic?", ["intrinsic", "extrinsic"], "intrinsic", "C2"),
                7, ", so the rotation uses ",
                ". The axis correspondence is transformed before the attitude. "
                "Write [Xb,Yb,Zb]^T=R1*R2*R3*R4*R5*[Xw,Yw,Zw]^T. "
                "Choose R1 through R5 from the matrix names already obtained. R1 in the formula is "),
    ])
    order = ["R_φ_M", "R_ψ_M", "R_θ_M", "RP2", "RP1"]
    factor_leads = [
        "",
        ". R2 is ",
        ". R3 is ",
        ". R4 is ",
        ". R5 is ",
    ]
    for lead, name in zip(factor_leads, order):
        tail = "." if name == "RP1" else ""
        questions.append(_placed(_choice(
            f"Which factor is {name} in [Xb,Yb,Zb]^T=R1*R2*R3*R4*R5*[Xw,Yw,Zw]^T?",
            _FACTORS, name, "A2",
        ), 7, lead, tail))
    total = sp.simplify(matrices[2] * matrices[1] * matrices[0])
    for number, matrix in enumerate((matrices[2], matrices[1], matrices[0]), start=1):
        lead = (
            "Before the product, expand each factor with the symbols just chosen. The expansion of R1 is "
            if number == 1 else f"The expansion of R{number} is "
        )
        questions.append(_placed(
            _write_matrix(f"Expansion of R{number} before the product.", matrix),
            8, lead, ". ",
        ))
    questions.append(_placed(
        _write_matrix("Write the simplified product R_t=R1*R2*R3.", total),
        8, "The simplified product R_t=R1*R2*R3 is ", ".",
    ))
    questions.append(_placed(
        _write_matrix("Write the transform R_t' from the object frame to the world frame.", sp.simplify(total.T)),
        9, "The transform R_t' from the object frame to the world frame is therefore ", ".",
    ))
    if len(questions) != 49:
        raise RuntimeError(f"object attitude should have 49 blanks, got {len(questions)}.")
    return questions, total, rp1, rp2


_TOWARD_KEY = {
    "body_up": "up", "body_down": "down",
    "body_left": "left", "body_right": "right",
}


def _signed(names: tuple[str, ...]) -> list[str]:
    return list(names) + ["-" + name for name in names]


def _camera_chain(scene: cm.Scene, index: int):
    """Wind-to-camera process, same paragraph shape for both cameras.

    R_cw = R_φ · R_θ · R_ψ · RP1 maps wind coordinates into this camera.
    """
    cam = scene.cam1 if index == 1 else scene.cam2
    names = ("X1", "Y1", "Z1") if index == 1 else ("X2", "Y2", "Z2")
    wind_names = ("Xw", "Yw", "Zw")
    angles = (scene.psi1, scene.theta1, scene.phi1) if index == 1 else (
        scene.psi2, scene.theta2, scene.phi2)
    symbols = ("ψ1", "θ1", "φ1") if index == 1 else ("ψ2", "θ2", "φ2")
    frame_label = f"camera-{index}"
    questions: list[dict] = []
    wind_basis = cm.basis(scene.wind, wind_names)
    cam_basis = cm.basis(cam, names)
    rp1 = sp.simplify(cam_basis.T * wind_basis)
    axis_options = _signed(wind_names + names)
    role_options = [
        f"{frame_label} absolute axial", f"{frame_label} absolute normal", f"{frame_label} absolute lateral",
        "world absolute axial", "world absolute normal", "world absolute lateral",
    ]
    role_gold = {
        "pan": f"{frame_label} absolute normal",
        "tilt": f"{frame_label} absolute lateral",
        "roll": f"{frame_label} absolute axial",
    }
    paragraph_base = 0 if index == 1 else 6
    if index == 1:
        direction_options = [
            "camera-1 frame to world frame", "world frame to camera-1 frame",
            "world frame to object frame", "object frame to world frame",
        ]
        direction_gold = "world frame to camera-1 frame"
    else:
        direction_options = [
            "camera-2 frame to world frame", "world frame to camera-2 frame",
            "world frame to object frame", "object frame to world frame",
        ]
        direction_gold = "world frame to camera-2 frame"

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        f"First determine the transform from the world frame to camera {index}. What is the conversion direction?",
        direction_options, direction_gold, "C2",
    ), 1 + paragraph_base, f"First determine the transform from the world frame to camera {index}. The conversion direction is ", ". ")
    put(_choice("Is every coordinate transform active or passive?", ["active", "passive"], "passive", "C2"),
        1 + paragraph_base, "Every coordinate transform is ", ", ")
    put(_choice("Is the transform treated as moving axes or fixed axes?", ["moving axes", "fixed axes"], "moving axes", "C2"),
        1 + paragraph_base, "treated as ", ", ")
    put(_choice("Does the rotation use extrinsic or intrinsic?", ["extrinsic", "intrinsic"], "intrinsic", "C2"),
        1 + paragraph_base, "and uses ", ". ")
    cam_leads = [
        f"The absolute axial (forward), normal (up), and lateral (right) directions of camera {index} correspond to ",
        ", ", ", and ",
    ]
    wind_leads = [
        ". The same three directions of the world frame correspond to ",
        ", ", ", and ",
    ]
    for lead, direction in zip(cam_leads, ("forward", "up", "right")):
        put(_choice(f"Which axis is camera {index}'s absolute {direction} direction?",
                    axis_options, _axis_pointing(cam, direction), "C1"), 1 + paragraph_base, lead)
    for lead, direction in zip(wind_leads, ("forward", "up", "right")):
        put(_choice(f"Which axis is the world frame's absolute {direction} direction?",
                    axis_options, _axis_pointing(scene.wind, direction), "C1"), 1 + paragraph_base, lead)
    put(_write_matrix(f"Write RP1, the preliminary matrix that aligns the world frame with camera {index}.", rp1),
        1 + paragraph_base,
        f". A preliminary matrix RP1 first makes the world-frame axes agree with camera {index} and forms the level intermediate frame of camera {index}. RP1 is ")
    euler_leads = [
        ". From that intermediate camera frame, the attitude follows the same Euler principle as ordinary flight mechanics. The Euler order is ",
        ", then ",
        ", then ",
    ]
    for lead, name in zip(euler_leads, CAM_EULER):
        tail = "." if name == "roll" else ""
        put(_choice(f"Which angle is this step of camera {index}'s Euler order?", CAM_EULER, name, "A1"),
            1 + paragraph_base, lead, tail)

    specs = (
        ("pan", "azimuth", symbols[0], angles[0], "axial", 2 + paragraph_base),
        ("tilt", "pitch", symbols[1], angles[1], "axial", 3 + paragraph_base),
        ("roll", "roll", symbols[2], angles[2], "lateral", 4 + paragraph_base),
    )
    matrices = {}
    for key, angle_name, symbol_name, angle, moving_role, paragraph in specs:
        role = ROLE_OF[key]
        toward = scene.toward_vector(scene.angle_toward[key], cam)
        side = DIR_CN[DIR_NAME[toward]]
        axis = _axis_cn(cam, role)
        _moving_name, moving = cm.axis_on(cam, moving_role)
        sign, matrix = signed_rotation(cam, names, axis, moving, toward, angle)
        matrices[key] = matrix
        if key == "roll":
            opening = (
                f"When camera {index} roll {symbol_name} is positive, the right edge of the image deflects toward "
            )
        else:
            opening = (
                f"When camera {index} {angle_name} {symbol_name} is positive, the optical axis deflects toward "
            )
        put(_choice(f"When camera {index} {angle_name} {symbol_name} is positive, which way does it deflect?",
                    SIX, side, "A1"),
            paragraph, opening, " of the world. ")
        put(_choice(f"Camera {index} {angle_name} {symbol_name} is a rotation about which absolute axis?",
                    role_options, role_gold[key], "C2"),
            paragraph, "This is a rotation about ", ". ")
        phrase, direction = _ROLE_POINTING[role]
        put(_choice(f"Which axis is the absolute {phrase} of camera {index}?",
                    axis_options, _axis_pointing(cam, direction), "C1"),
            paragraph, f"The absolute {phrase} of camera {index} corresponds to ", ". ")
        put(_choice(f"Which camera-{index} axis does {symbol_name} rotate about?",
                    _nine_axes(), axis, "C1"),
            paragraph, f"The rotation uses camera-{index} axis ", " with sense ")
        put(_choice("Is the sense positive or negative?", THREE_SIGN, _bias(sign), "A1"),
            paragraph, "", ". ")
        put(_write_matrix(f"Write the rotation matrix R_{symbol_name}.", matrix),
            paragraph, f"The rotation matrix R_{symbol_name} is ", ".")

    factors = [f"R_{symbols[0]}", f"R_{symbols[1]}", f"R_{symbols[2]}", "RP1"]
    # Intrinsic: first angle is rightmost; RP1 is applied before the angles.
    ordered = [factors[2], factors[1], factors[0], "RP1"]
    factor_leads = [
        f"The coordinate transform from the world frame to camera {index} is R_cw{index}=R1·R2·R3·R4. R1 is ",
        ". R2 is ", ". R3 is ", ". R4 is ",
    ]
    for lead, name in zip(factor_leads, ordered):
        tail = ". " if name == "RP1" else ""
        put(_choice(f"Which factor is this term of R_cw{index}=R1·R2·R3·R4?", factors, name, "A2"),
            5 + paragraph_base, lead, tail)
    total = sp.simplify(matrices["roll"] * matrices["tilt"] * matrices["pan"] * rp1)
    for number, matrix in enumerate((matrices["roll"], matrices["tilt"], matrices["pan"], rp1), start=1):
        lead = (
            "Before the product, expand each factor with the symbols just chosen. The expansion of R1 is "
            if number == 1 else f"The expansion of R{number} is "
        )
        put(_write_matrix(f"Expansion of R{number} before the product.", matrix), 5 + paragraph_base, lead, ". ")
    put(_write_matrix(
        f"From ψ{index}, θ{index}, φ{index} and the current axes, write the rotation from the world frame to camera {index}.",
        total),
        5 + paragraph_base, f"The rotation from the world frame to camera {index} is R_cw{index}=", ".")
    center = C1 if index == 1 else C2
    t_cw = sp.simplify(-total * center)
    correct = f"t_cw{index} = −R_cw{index}·C{index}"
    put(_choice(
        f"What is the form of the pose translation t_cw{index} of camera {index}?",
        [
            f"t_cw{index} = R_cw{index}·C{index}",
            f"t_cw{index} = −R_cw{index}^T·C{index}",
            correct,
            f"t_cw{index} = C{index} − R_cw{index}",
        ],
        correct, "A2"),
        6 + paragraph_base, f"The pose translation t_cw{index} of camera {index} has the form ", ". ")
    put(_write_vector(f"Write the pose translation t_cw{index} of camera {index}.", t_cw),
        6 + paragraph_base, f"Write t_cw{index}=", ".")
    if len(questions) != 43:
        raise RuntimeError(f"camera {index} should have 43 blanks, got {len(questions)}.")
    return questions, total


def build(spec: dict) -> dict:
    scene = cm.Scene().declare(spec)
    wind_x = scene.wind["Xw"]
    if cm.role_of(wind_x) != "axial" or DIR_NAME[wind_x] != "forward":
        raise RuntimeError("Xw of the world frame must lie on the axial direction and point forward.")
    questions: list[dict] = []

    cam1_questions, r_cw1 = _camera_chain(scene, 1)
    questions.extend(cam1_questions)
    cam2_questions, r_cw2 = _camera_chain(scene, 2)
    questions.extend(cam2_questions)

    body_questions, r_t, rp1, rp2 = _body_mechanism_questions(scene)
    questions.extend(body_questions)

    t_cw1 = sp.simplify(-r_cw1 * C1)
    t_cw2 = sp.simplify(-r_cw2 * C2)
    # Body axis in wind coordinates: p_wind = RP1^T · RP2^T · R_t' · e_body.
    wind_basis = cm.basis(scene.wind, ("Xw", "Yw", "Zw"))
    to_wind = sp.simplify(rp1.T * rp2.T * r_t.T)
    # Nose is L along the body longitudinal axis, expressed in the wind frame.
    nose = sp.simplify(scene.length * to_wind * _body_unit(scene.body, "axial"))
    air = sp.simplify(wind_basis.T * sp.Matrix([1, 0, 0]))
    bx = sp.simplify(to_wind * _body_unit(scene.body, "axial"))
    bz = sp.simplify(to_wind * _body_unit(scene.body, "normal"))

    def _skew(v):
        px, py, pz = v[0], v[1], v[2]
        return sp.Matrix([[0, -pz, py], [pz, 0, -px], [-py, px, 0]])

    s1 = cm.camera_layout_permutation(scene.cam1, ("X1", "Y1", "Z1"))
    s2 = cm.camera_layout_permutation(scene.cam2, ("X2", "Y2", "Z2"))
    k1it = sp.simplify((K1 * s1).inv().T)
    k2it = sp.simplify((K2 * s2).inv().T)
    p_c1 = sp.simplify(r_cw1 * nose + t_cw1)
    p_c2 = sp.simplify(r_cw2 * nose + t_cw2)
    skew1 = sp.simplify(_skew(p_c1))
    skew2 = sp.simplify(_skew(p_c2))

    def _put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    def _line_lead(c: str) -> str:
        if c == "1":
            return (
                "l is the line in the camera-1 image. K1 is the camera-1 intrinsic. "
                "S1 reorders camera-1 coordinates into lateral positive (right), normal positive (up), axial positive (forward), including sign flips. "
                "(K1·S1)^(-T) is the inverse transpose of the effective intrinsic K1·S1. "
                "[P_c1]_× is the skew matrix of the object nose P_c1 in camera 1. "
                "R_cw1 rotates the world frame into camera 1. d_w is the direction of this spatial line in the world frame. "
                "The line l in the camera-1 image satisfies "
            )
        return (
            "l is the line in the camera-2 image. K2 is the camera-2 intrinsic. "
            "S2 reorders camera-2 coordinates into the same right, up, forward order, including sign flips. "
            "(K2·S2)^(-T) is the inverse transpose of K2·S2. "
            "[P_c2]_× is the skew matrix of the object nose P_c2 in camera 2. "
            "R_cw2 rotates the world frame into camera 2. d_w is the direction of this spatial line in the world frame. "
            "The line l in the camera-2 image satisfies "
        )

    def _line_formula(cam: str) -> dict:
        c = cam
        correct = f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · R_cw{c} · d_w"
        options = [
            f"l = (K{c}·S{c})^(-T) · R_cw{c} · [P_c{c}]_× · d_w",
            f"l = (K{c}·S{c})^(-1) · [P_c{c}]_× · R_cw{c} · d_w",
            correct,
            f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · R_cw{c}^T · d_w",
            f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · d_w",
            f"l = [P_c{c}]_× · (K{c}·S{c})^(-T) · R_cw{c} · d_w",
        ]
        return _choice(_line_lead(c), options, correct, "A2")

    def _nose_formula() -> dict:
        correct = "p_nose = L·RP1^T·R_t'·e"
        options = [
            "p_nose = L·R_t'·e",
            correct,
            "p_nose = RP1^T·R_t'·e",
            "p_nose = L·[1, 0, 0]^T",
        ]
        return _choice(
            "e is the unit vector of the object longitudinal axis in the object frame. What is the form of p_nose?",
            options, correct, "A2")

    def _point_formula(cam: str) -> dict:
        c = cam
        correct = f"P_c{c} = R_cw{c}·p_nose + t_cw{c}"
        options = [
            f"P_c{c} = R_cw{c}·p_nose − t_cw{c}",
            f"P_c{c} = R_cw{c}^T·p_nose + t_cw{c}",
            correct,
            f"P_c{c} = p_nose + t_cw{c}",
        ]
        return _choice(f"What is the form of the object nose P_c{c} in camera {c}?", options, correct, "A2")

    _put(_line_formula("1"), 1, _line_lead("1"), ". ")
    _put(_line_formula("2"), 1, _line_lead("2"), ".")
    _put(_write_matrix(
        "Expand (K1·S1)^(-T) by itself. Do not multiply it by [P_c1]_×, R_cw1, or d_w.", k1it),
         2, "Expand (K1·S1)^(-T) by itself and do not multiply it by another symbol. (K1·S1)^(-T)=", ". ")
    _put(_write_matrix(
        "Expand (K2·S2)^(-T) by itself. Do not multiply it by [P_c2]_×, R_cw2, or d_w.", k2it),
         2, "Expand (K2·S2)^(-T) by itself and do not multiply it by another symbol. (K2·S2)^(-T)=", ".")
    _put(_nose_formula(), 3,
         "e is the unit vector of the object longitudinal axis in the object frame. The object nose p_nose in the world frame has the form ", ". ")
    _put(_write_vector("Write the object nose p_nose in the world frame.", nose),
         3, "Write p_nose=", ".")
    _put(_point_formula("1"), 4, "The object nose P_c1 in camera 1 has the form ", ". ")
    _put(_write_vector("Write the object nose P_c1 in camera 1.", p_c1),
         4, "Write P_c1=", ". ")
    _put(_point_formula("2"), 4, "The object nose P_c2 in camera 2 has the form ", ". ")
    _put(_write_vector("Write the object nose P_c2 in camera 2.", p_c2),
         4, "Write P_c2=", ".")
    _put(_write_matrix(
        "Expand [P_c1]_× by itself. Do not multiply it by (K1·S1)^(-T), R_cw1, or d_w.", skew1),
         5, "Expand [P_c1]_× by itself and do not multiply it by another symbol. [P_c1]_×=", ". ")
    _put(_write_matrix(
        "Expand [P_c2]_× by itself. Do not multiply it by (K2·S2)^(-T), R_cw2, or d_w.", skew2),
         5, "Expand [P_c2]_× by itself and do not multiply it by another symbol. [P_c2]_×=", ".")

    _put(_write_vector(
        "The reference direction in the world frame is d_wV. "
        "Its image line is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV in camera 1 and "
        "l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV in camera 2. Write d_wV.",
        air),
        1,
        "The reference direction in the world frame is d_wV=",
        ". Its image line is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV in camera 1 and l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV in camera 2. ")
    _put(_write_vector(
        "e is the unit longitudinal axis in the object frame. Write its world direction d_wL.",
        bx),
        1,
        "e is the unit longitudinal axis in the object frame. Its world direction is d_wL=",
        ". Its image line is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL in camera 1 and l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL in camera 2. ")
    _put(_write_vector(
        "e is the unit normal axis in the object frame. Write its world direction d_wN.",
        bz),
        1,
        "e is the unit normal axis in the object frame. Its world direction is d_wN=",
        ". Its image line is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN in camera 1 and l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN in camera 2.")

    r21 = sp.simplify(r_cw2 * r_cw1.T)
    t21 = sp.simplify(t_cw2 - r21 * t_cw1)
    _put(_write_matrix("Expansion of R_cw2 before the product.", r_cw2),
         1, "Before the product, expand each factor with the symbols just chosen. The expansion of R_cw2 is ", ". ")
    _put(_write_matrix("Expansion of R_cw1 before the product.", r_cw1),
         1, "The expansion of R_cw1 is ", ". ")
    _put(_write_matrix(
        "From C1=(x1,y1,z1), C2=(x2,y2,z2) and the attitudes, write the rotation R21 from camera 1 to camera 2.",
        r21),
         1,
         "The optical centers in the world frame are C1=(x1,y1,z1) and C2=(x2,y2,z2), with attitudes ψ1, θ1, φ1 and ψ2, θ2, φ2. "
         "Derive the rotation R21 and translation t21 from camera 1 to camera 2. R21=",
         "")
    _put(_write_vector(
        "From C1=(x1,y1,z1), C2=(x2,y2,z2) and the attitudes, write the translation t21 from camera 1 to camera 2.",
        t21),
         1, " and t21=", ".")
    _put(_choice("The essential matrix is?",
                 ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
                 "E = skew(t21)*R21", "A2"),
         1, "The essential matrix is ", ".")
    _put(_choice("The fundamental matrix is?",
                 ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)", "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
                 "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2"),
         2, "The fundamental matrix is ", ".")
    _put(_choice("The epipolar constraint is?",
                 ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
                  "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
                 "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2"),
         3, "The epipolar constraint is ", ".")
    inv1 = sp.simplify((K1 * s1).inv())
    inv2 = sp.simplify((K2 * s2).inv())
    d1 = sp.simplify(inv1 * sp.Matrix([u1, v1, 1]))
    d2 = sp.simplify(inv2 * sp.Matrix([u2, v2, 1]))

    _ray = (
        "The homogeneous image coordinates of the object reference point are [u1,v1,1] in camera 1 and [u2,v2,1] in camera 2. "
        "The ray directions are d1=(K1·S1)^(-1)·[u1,v1,1]^T and d2=(K2·S2)^(-1)·[u2,v2,1]^T. "
        "Expand each ray as a vector. "
    )
    _put(_write_vector(_ray + "Write d1.", d1), 4, _ray + "d1=", "")
    _put(_write_vector(_ray + "Write d2.", d2), 4, " and d2=", ".")
    _put(_choice("The object reference point X_c1 in camera 1 lies along the ray. X_c1 = ?",
                 ["X_c1 = λ1·d1", "X_c1 = d1 + λ1", "X_c1 = λ1 + d1", "X_c1 = d1 − λ1"],
                 "X_c1 = λ1·d1", "A2"),
         5, "The object reference point X_c1 in camera 1 lies along the ray d1, where λ1 is the depth. X_c1=", ".")
    _put(_choice("Both cameras see the same object reference point. What is the constraint?",
                 ["λ2·d2 = R21·λ1·d1 + t21", "λ2·d2 = R21·λ1·d1 − t21",
                  "λ2·d2 = R21^T·λ1·d1 + t21", "λ1·d1 = R21·λ2·d2 + t21"],
                 "λ2·d2 = R21·λ1·d1 + t21", "A2"),
         6, "Both cameras see the same object reference point. The transform from camera 1 to camera 2 is X_c2=R21·X_c1+t21, so the constraint is ", ".")
    _put(_choice("Cross both sides by which vector to eliminate λ2?",
                 ["d2", "d1", "R21·d1", "t21"],
                 "d2", "A2"),
         7, "Cross both sides of the constraint by ", ", eliminate λ2, and obtain d2×(λ1·R21·d1+t21)=0.")
    _put(_choice(
        "Under the least-squares criterion, and with d2×R21·d1 ≠ 0, "
        "which λ1 minimizes the squared residual of d2×(λ1·R21·d1+t21)=0?",
        ["λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = −(d2×t21) / (d2×R21·d1)",
         "λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²"],
        "λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²", "A2"),
         8,
         "Under the least-squares criterion, and with d2×R21·d1 ≠ 0, the λ1 that minimizes the squared residual of d2×(λ1·R21·d1+t21)=0 is ",
         ".")
    _put(_choice("Given X_c1 in camera 1, what is its world coordinate X_w?",
                 ["X_w = R_cw1^T·(X_c1 − t_cw1)", "X_w = R_cw1·X_c1 + t_cw1",
                  "X_w = R_cw1^T·X_c1 − t_cw1", "X_w = R_cw1·(X_c1 − t_cw1)"],
                 "X_w = R_cw1^T·(X_c1 − t_cw1)", "A2"),
         9, "Given the object reference point X_c1 in camera 1, its world coordinate is ", ".")
    _put(_choice("Given X_w in the world frame, what is its object coordinate X_b?",
                 ["X_b = R_t·RP1·(X_w − p_nose)", "X_b = RP1^T·R_t'·(X_w − p_nose)",
                  "X_b = R_t·RP1·(X_w + p_nose)", "X_b = R_t'·RP1·(X_w − p_nose)"],
                 "X_b = R_t·RP1·(X_w − p_nose)", "A2"),
         10, "Given the object reference point X_w in the world frame, its object coordinate is ", ".")

    SEGMENT_BOUNDS = [
        (86, "camera_attitude"),
        (135, "body_attitude"),
        (147, "projection"),
        (150, "axis_wind"),
        (154, "extrinsic"),
        (165, "epipolar"),
    ]

    def _segment_of(index: int) -> str:
        for bound, seg in SEGMENT_BOUNDS:
            if index <= bound:
                return seg
        return "epipolar"

    for index, item in enumerate(questions, start=1):
        item["local_id"] = index
        item["id"] = str(index)
        item["process"] = "stereo pose"
        item["segment"] = _segment_of(index)
        item["format_code"] = item.pop("format")
    return {
        "spec": spec,
        "declaration": _declaration(spec, scene),
        "questions": questions,
    }


def _declaration(spec: dict, scene: cm.Scene) -> str:
    hand_word = {"right": "right-handed", "left": "left-handed"}

    def frame_block(title: str, names: tuple, frame: dict, frame_spec: dict, origin: bool = False) -> str:
        hand = hand_word[frame_spec.get("chirality", "left")]
        titled = title[0].upper() + title[1:]
        if origin:
            head = f"{titled} ({hand}); the object origin is at the object nose:"
        else:
            head = f"{titled} ({hand}):"
        lines = [head]
        for name in names:
            vector = frame[name]
            role = cm.role_of(vector)
            pointing = POINT_WORD[DIR_NAME[vector]]
            where = "as declared" if name in frame_spec else "inferred from chirality"
            lines.append(f"- {name}: along the {role} direction, pointing {pointing} ({where}).")
        return "\n".join(lines)

    wind_block = frame_block("world frame", ("Xw", "Yw", "Zw"), scene.wind, spec["wind"])
    body_block = frame_block("object frame", ("Xb", "Yb", "Zb"), scene.body, spec["body"], origin=True)
    cam1_block = frame_block("camera-1 frame", ("X1", "Y1", "Z1"), scene.cam1, spec["camera1"])
    cam2_block = frame_block("camera-2 frame", ("X2", "Y2", "Z2"), scene.cam2, spec["camera2"])

    def _camera_attitude_lines() -> list[str]:
        """Camera pitch, azimuth, and roll. The sign still follows this sample."""
        pitch = "positive" if scene.angle_toward["tilt"] == "wind_up" else "negative"
        yaw = "positive" if scene.angle_toward["pan"] == "wind_right" else "negative"
        roll = "positive" if scene.angle_toward["roll"] == "camera_down" else "negative"
        return [
            "Camera pitch θ1, θ2: the angle between the camera longitudinal axis and the world horizontal plane. "
            f"The angle is {pitch} when the camera longitudinal axis rises above the world.",
            "Camera azimuth ψ1, ψ2: the angle between the horizontal projection of the camera longitudinal axis and Xw of the world frame. "
            f"The angle is {yaw} when the projection lies to the right of Xw.",
            "Camera roll φ1, φ2: the angle between the camera normal and the vertical plane through the camera longitudinal axis. "
            f"The angle is {roll} when the camera rolls to its right.",
            "The two cameras use the ordinary Euler order and are not constrained by the mechanism. "
            "Camera 1 uses azimuth, pitch, and roll ψ1, θ1, φ1. Camera 2 uses ψ2, θ2, φ2.",
        ]

    return "\n".join([
        "[Declaration]",
        "",
        "Fixed conventions",
        "- Coordinate transforms are passive. Attitude rotations use an intrinsic moving-axis sequence.",
        "- Camera imaging uses the pinhole model x = (K·S)·[R|t]·X. An image line is l = (K·S)^(-T)·n, where n is the normal of the plane spanned by the optical center and the spatial line.",
        "- The epipolar constraint is x2^T·F·x1 = 0, with F = (K2·S2)^(-T)·E·(K1·S1)^(-1) and E = [t21]_×·R21.",
        "- Everything below is described in the world frame.",
        "- The object-frame origin is at the object nose. The distance from the object nose to the world origin is L.",
        "- Every quantity is symbolic: L, θ_M, ψ_M, φ_M, ψ1, θ1, φ1, ψ2, θ2, φ2. The camera centers in the world frame are camera-1 C1=(x1,y1,z1) and camera-2 C2=(x2,y2,z2).",
        "- The intrinsics are K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]] and K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]. K acts in the order [lateral positive (right), normal positive (up), axial positive (forward)]. S reorders that camera's coordinates into this order, including sign flips. The effective intrinsic is K·S.",
        "",
        "Absolute directions",
        "The world, the object, and a camera each have three absolute axes. The three sets use the same positive senses: axial positive is forward, normal positive is up, and lateral positive is right. The six absolute directions are the two senses of these three axes.",
        "Axial positive, normal positive, lateral positive, and the positive world, object, and camera axes named below are these absolute senses, not the positive senses of the coordinate axes. A coordinate-axis declaration only assigns each axis to an absolute axis and a sense. That sense may agree with the absolute positive direction or oppose it.",
        "The three frames share one orientation rule: forward × right = down. Looking along forward, right is the right-hand side and down is vertically downward. This rule does not change with the definition.",
        "The world's three axes and six directions, relative to the world:",
        "- Axial: the world's fore-aft axis. In the horizontal plane the positive sense is forward, and forward is the world forward direction. Rearward is the opposite sense.",
        "- Normal: the world's vertical axis. The positive sense is up, upward is vertically up, and downward is vertically down.",
        "- Lateral: the world's horizontal transverse axis. In the horizontal plane the positive sense is right. Left is the opposite sense.",
        "The object's three axes and six directions, relative to the object:",
        "- Axial: the axis joining the object rear to the object nose. The positive sense is forward, toward the nose. Rearward points toward the rear.",
        "- Normal: the object's up-down axis. The positive sense is up. Down is the opposite sense.",
        "- Lateral: the object's left-right axis. The positive sense is right. Left is the opposite sense.",
        "The camera's three axes and six directions, relative to the camera:",
        "- Axial: the optical axis. The positive sense is forward, along the viewing direction. Rearward is the opposite sense.",
        "- Normal: the vertical direction of the image. The positive sense is up. Down is the opposite sense.",
        "- Lateral: the horizontal direction of the image. The positive sense is right. Left is the opposite sense.",
        "",
        "Scene",
        "At the initial pose the object is level in the world frame, and the object front points along the world forward direction. Two calibrated, synchronously exposed cameras photograph the object from different directions. The optical axis follows the camera axial direction, the image horizontal follows the camera lateral direction, and the image vertical follows the camera normal direction. The rear of the object is fixed to a mechanism that uses a nonstandard Euler sequence to place the object in a special attitude. The sequence differs from the usual order, so the attitude angles have to be understood before the order is inferred. Those definitions are given below.",
        "There are two tasks, and the object has a placement attitude in the world frame. The first projects the object's longitudinal axis and normal axis into a camera image and writes both axes in image coordinates. The second is given one reference point on the object in a camera image and, from the stereo camera model, recovers that point in the object frame. The subtasks are: from the object-to-world attitude, express the longitudinal and normal axes in the world frame; from the camera placement relative to the world, derive the world-to-camera transform and then the essential and fundamental matrices of the stereo pair; from the two world-frame axes and the stereo model, write the axes in image coordinates; from the reference point in one camera image, reconstruct it in the object frame.",
        "",
        "Angles",
        *_mechanism_definition(scene),
        *_camera_attitude_lines(),
        "",
        "Frames",
        "",
        wind_block,
        "",
        body_block,
        "",
        cam1_block,
        "",
        cam2_block,
        "",
        "Background",
        "Axial positive, normal positive, and lateral positive are forward, up, and right. Pitch alone rotates about the absolute lateral axis. Azimuth alone rotates about the absolute normal axis. Roll alone rotates about the absolute axial axis. The positive sense of each angle is the sense in the declaration. Coordinate transforms are passive. Attitude angles use an intrinsic moving-axis sequence.",
    ])


def generate(spec: dict) -> dict:
    return build(spec)


if __name__ == "__main__":
    for key in FRAME_NAMES:
        print(key, len(alternate_frames(key)))
    sample = build(STANDARD_SPEC)
    print("standard questions", len(sample["questions"]))
