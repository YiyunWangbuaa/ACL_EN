# -*- coding: utf-8 -*-
"""CV-CTRL: a computer-vision control aligned blank-by-blank with the pure vision slice.

The pure vision slice of the main data is camera_attitude (42) + projection (12)
+ extrinsic (2) + epipolar (5), 61 blanks. This control keeps those 61 blanks,
uses the world frame in place of the former wind frame and the world origin
and world axes in place of the former nose and body axes, and drops
body_attitude and axis_wind.
"""
from __future__ import annotations

import sympy as sp

from domains.computer_vision import cv_model as cm

DIR_CN = {"forward": "forward", "rearward": "rearward", "right": "right",
          "left": "left", "down": "downward", "up": "upward"}
DIR_NAME = {value: key for key, value in cm.DIRECTIONS.items()}
SIX = ["forward", "rearward", "right", "left", "downward", "upward"]
AXIS_ALL = ["Xw", "Yw", "Zw", "X1", "Y1", "Z1", "X2", "Y2", "Z2"]
AXIS_OPTIONS = AXIS_ALL + ["-" + name for name in AXIS_ALL]
CAM_EULER = ["azimuth", "pitch", "roll"]

fx1, fy1, cx1, cy1 = sp.symbols("fx1 fy1 cx1 cy1")
fx2, fy2, cx2, cy2 = sp.symbols("fx2 fy2 cx2 cy2")
x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2")
K1 = sp.Matrix([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
K2 = sp.Matrix([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
C1 = sp.Matrix([x1, y1, z1])
C2 = sp.Matrix([x2, y2, z2])

SPECS = {
    25: {
        "world": {"chirality": "left", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "up"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "right"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    50: {
        "world": {"chirality": "left", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "up"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "rearward"},
                    "X1": {"along": "lateral", "points": "left"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    75: {
        "world": {"chirality": "right", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "down"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "right"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    100: {
        "world": {"chirality": "right", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "down"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "left"}},
        "camera2": {"chirality": "right", "Z2": {"along": "normal", "points": "down"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
}
DUMMY_BODY = {"chirality": "left", "Xb": {"along": "axial", "points": "forward"},
              "Zb": {"along": "normal", "points": "up"}}


def _m(matrix):
    return "[" + ", ".join("[" + ", ".join(sp.sstr(v) for v in row) + "]" for row in matrix.tolist()) + "]"


def _choice(prompt, options, correct, label):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return {"prompt": prompt, "label": label, "format_code": "choice",
            "options": list(options), "answer": letters[options.index(correct)]}


def _matrix(prompt, matrix, label="A2"):
    return {"prompt": prompt, "label": label, "format_code": "matrix", "options": [],
            "answer": _m(matrix)}


def _vector(prompt, vec, label="A2"):
    return {"prompt": prompt, "label": label, "format_code": "vector", "options": [],
            "answer": "[" + ", ".join(sp.sstr(v) for v in vec) + "]"}


def build(tier):
    spec = {"wind": SPECS[tier]["world"], "body": DUMMY_BODY,
            "camera1": SPECS[tier]["camera1"], "camera2": SPECS[tier]["camera2"]}
    scene = cm.Scene().declare(spec)
    cam1_s = scene.camera_senses(scene.cam1)
    cam2_s = scene.camera_senses(scene.cam2)
    psi1, th1, ph1 = scene.psi1, scene.theta1, scene.phi1
    psi2, th2, ph2 = scene.psi2, scene.theta2, scene.phi2

    qs = []

    def angle_process(cam, senses, ang, name, key):
        role = {"pan": "normal", "tilt": "lateral", "roll": "axial"}[key]
        toward = scene.toward_vector(scene.angle_toward[key], cam)
        bias = DIR_CN[DIR_NAME[toward]]
        axis = cm.axis_on(cam, role)[0]
        sign = senses[{"pan": 0, "tilt": 1, "roll": 2}[key]]
        matrix = cm.rot(role, sign * ang)
        qs.append(_choice(
            f"When {name} is positive, which way does the optical axis deflect in the world frame?",
            SIX, bias, "A1"))
        qs.append(_choice(
            f"Which level frame is the axis of {name}?",
            ["world frame", "camera-1 frame", "camera-2 frame"], _cam_label(cam), "C2"))
        qs.append(_choice(f"Which camera-frame axis does {name} rotate about?", AXIS_ALL, axis, "C1"))
        qs.append(_choice(
            "Is the sense about that axis positive or negative?",
            ["positive", "negative"], "positive" if sign > 0 else "negative", "A1"))
        qs.append(_matrix(f"Write the rotation matrix for {name}.", matrix))

    def _cam_label(cam):
        return "camera-1 frame" if cam is scene.cam1 else "camera-2 frame"

    # Camera-1 attitude, one blank per main-data blank.
    qs.append(_choice(
        "Which conversion brings the camera-1 attitude into the imaging geometry?",
        ["camera-1 frame to world frame", "world frame to camera-1 frame"],
        "camera-1 frame to world frame", "C2"))
    qs.append(_choice("Is the transform active or passive?", ["active", "passive"], "passive", "C2"))
    qs.append(_choice("Does the camera frame use moving axes or fixed axes?", ["moving axes", "fixed axes"], "moving axes", "C2"))
    qs.append(_choice("Does the rotation use intrinsic or extrinsic?", ["intrinsic", "extrinsic"], "intrinsic", "C2"))
    for i in range(3):
        qs.append(_choice(f"Camera-1 attitude order, step {i + 1}?", CAM_EULER, CAM_EULER[i], "A1"))
    angle_process(scene.cam1, cam1_s, psi1, "camera-1 azimuth ψ1", "pan")
    angle_process(scene.cam1, cam1_s, th1, "camera-1 pitch θ1", "tilt")
    angle_process(scene.cam1, cam1_s, ph1, "camera-1 roll φ1", "roll")
    qs.append(_matrix("Write R_ψ1 in R_cg1 = R_ψ1·R_θ1·R_φ1·A1.", cm.rot("normal", cam1_s[0] * psi1)))
    qs.append(_matrix("Write R_θ1 in R_cg1 = R_ψ1·R_θ1·R_φ1·A1.", cm.rot("lateral", cam1_s[1] * th1)))
    qs.append(_matrix("Write R_φ1 in R_cg1 = R_ψ1·R_θ1·R_φ1·A1.", cm.rot("axial", cam1_s[2] * ph1)))
    r_cg1 = scene.camera_to_ground(scene.cam1, ("X1", "Y1", "Z1"), psi1, th1, ph1)
    qs.append(_matrix("Write the fully expanded rotation R_cg1 from camera 1 to the world frame.", r_cg1))

    # Camera-2 attitude.
    angle_process(scene.cam2, cam2_s, psi2, "camera-2 azimuth ψ2", "pan")
    angle_process(scene.cam2, cam2_s, th2, "camera-2 pitch θ2", "tilt")
    angle_process(scene.cam2, cam2_s, ph2, "camera-2 roll φ2", "roll")
    qs.append(_matrix("Write R_ψ2 in R_cg2 = R_ψ2·R_θ2·R_φ2.", cm.rot("normal", cam2_s[0] * psi2)))
    qs.append(_matrix("Write R_θ2 in R_cg2 = R_ψ2·R_θ2·R_φ2.", cm.rot("lateral", cam2_s[1] * th2)))
    qs.append(_matrix("Write R_φ2 in R_cg2 = R_ψ2·R_θ2·R_φ2.", cm.rot("axial", cam2_s[2] * ph2)))
    r_cg2 = scene.camera_to_ground(scene.cam2, ("X2", "Y2", "Z2"), psi2, th2, ph2)
    qs.append(_matrix("Write the fully expanded rotation R_cg2 from camera 2 to the world frame.", r_cg2))

    # Projection, aligned with the main data. The point is the world origin.
    r_cw1 = sp.simplify(r_cg1.T)
    r_cw2 = sp.simplify(r_cg2.T)
    t_cw1 = sp.simplify(-r_cw1 * C1)
    t_cw2 = sp.simplify(-r_cw2 * C2)
    origin = sp.Matrix([0, 0, 0])
    p_c1 = sp.simplify(r_cw1 * origin + t_cw1)
    p_c2 = sp.simplify(r_cw2 * origin + t_cw2)

    def skew(v):
        px, py, pz = v[0], v[1], v[2]
        return sp.Matrix([[0, -pz, py], [pz, 0, -px], [-py, px, 0]])

    skew1 = sp.simplify(skew(p_c1))
    skew2 = sp.simplify(skew(p_c2))
    s1 = cm.camera_layout_permutation(scene.cam1, ("X1", "Y1", "Z1"))
    s2 = cm.camera_layout_permutation(scene.cam2, ("X2", "Y2", "Z2"))
    k1it = sp.simplify((K1 * s1).inv().T)
    k2it = sp.simplify((K2 * s2).inv().T)

    qs.append(_matrix("Write the inverse transpose (K1·S1)^(-T) of the camera-1 effective intrinsic.", k1it))
    qs.append(_matrix("Write the inverse transpose (K2·S2)^(-T) of the camera-2 effective intrinsic.", k2it))
    qs.append(_matrix("Write the pose rotation R_cw1 of camera 1 (world frame to camera-1 frame).", r_cw1))
    qs.append(_matrix("Write the pose rotation R_cw2 of camera 2 (world frame to camera-2 frame).", r_cw2))
    qs.append(_vector("Write the pose translation t_cw1 = −R_cw1·C1 of camera 1.", t_cw1))
    qs.append(_vector("Write the pose translation t_cw2 = −R_cw2·C2 of camera 2.", t_cw2))
    qs.append(_vector("Write the world origin in camera 1, P_c1 = R_cw1·O + t_cw1.", p_c1))
    qs.append(_vector("Write the world origin in camera 2, P_c2 = R_cw2·O + t_cw2.", p_c2))
    qs.append(_matrix("Write the skew matrix [P_c1]_×.", skew1))
    qs.append(_matrix("Write the skew matrix [P_c2]_×.", skew2))

    def line_formula(cam):
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
        return _choice(f"Which formula does the line l in the camera-{c} image satisfy?", options, correct, "A2")

    qs.append(line_formula("1"))
    qs.append(line_formula("2"))

    # extrinsic
    r21 = sp.simplify(r_cw2 * r_cw1.T)
    t21 = sp.simplify(t_cw2 - r21 * t_cw1)
    qs.append(_matrix("Before the product, expand each factor with the symbols just chosen. The expansion of R_cw2 is", r_cw2))
    qs.append(_matrix("The expansion of R_cw1 is", r_cw1))
    qs.append(_matrix("Write the fully expanded rotation R21 from camera 1 to camera 2.", r21))
    qs.append(_vector("Write the fully expanded translation t21 from camera 1 to camera 2.", t21))

    # Epipolar geometry.
    qs.append(_choice("The essential matrix is?",
                      ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
                      "E = skew(t21)*R21", "A2"))
    qs.append(_choice("The fundamental matrix is?",
                      ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)", "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
                      "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2"))
    qs.append(_choice("The epipolar constraint is?",
                      ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
                       "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
                      "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2"))
    qs.append(_choice("In which frame is the triangulated point?", ["world frame", "camera-1 frame", "camera-2 frame"], "world frame", "A1"))
    qs.append(_choice("When does triangulation degenerate?",
                      ["the baseline is zero, the point is at infinity, or the depth is negative",
                       "a longer baseline", "known intrinsics", "more image points"],
                      "the baseline is zero, the point is at infinity, or the depth is negative", "A1"))

    SEGMENT_BOUNDS = [(45, "camera_attitude"), (57, "projection"), (61, "extrinsic"), (66, "epipolar")]
    OFFSET = {"camera_attitude": 12, "projection": 38, "extrinsic": 42, "epipolar": 42}

    def _segment_of(index):
        for bound, seg in SEGMENT_BOUNDS:
            if index <= bound:
                return seg
        return "epipolar"

    for i, item in enumerate(qs, start=1):
        seg = _segment_of(i)
        item["local_id"] = i
        item["id"] = str(i)
        item["process"] = "world to camera"
        item["segment"] = seg
        item["corresponds_to"] = str(i + OFFSET[seg])

    return {
        "sample_id": f"cvctrl-{tier:03d}", "tier": tier, "spec": SPECS[tier],
        "group": "control",
        "declaration": _decl(SPECS[tier], scene), "questions": qs,
    }


def _decl(spec, scene):
    def axes_text(names, frame):
        return "; ".join(
            f"{axis} points {DIR_CN[DIR_NAME[frame[axis]]]}" for axis in names
        )

    hand = {"right": "right-handed", "left": "left-handed"}
    return "\n".join([
        "Fixed conventions",
        "- Coordinate transforms are passive. Camera attitude uses an intrinsic moving-axis sequence. The first angle is the rightmost matrix in the product.",
        "- Camera imaging uses the pinhole model x = (K·S)·[R|t]·X. An image line is l = (K·S)^(-T)·n, where n is the normal of the plane spanned by the optical center and the spatial line.",
        "- The epipolar constraint is x2^T·F·x1 = 0, with F = (K2·S2)^(-T)·E·(K1·S1)^(-1) and E = [t21]_×·R21.",
        "",
        "Absolute directions",
        "The world frame and a camera frame each have three absolute axes. The two sets use the same positive senses: axial positive is forward, normal positive is up, and lateral positive is right. The six absolute directions are the two senses of these three axes.",
        "A coordinate-axis declaration only assigns each axis to an absolute axis and a sense. That sense may agree with the absolute positive direction or oppose it.",
        "The two frames share one orientation rule: forward × right = down. Looking along forward, right is the right-hand side and down is vertically downward. This rule does not change with the definition.",
        "The world's three axes and six directions, relative to the world:",
        "- Axial: the world's fore-aft axis. The positive sense is forward. Rearward is the opposite sense.",
        "- Normal: the world's vertical axis. The positive sense is up, upward is vertically up, and downward is vertically down.",
        "- Lateral: the world's left-right axis. The positive sense is right. Left is the opposite sense.",
        "The camera's three axes and six directions, relative to the camera:",
        "- Axial: the optical axis. The positive sense is forward, along the viewing direction. Rearward is the opposite sense.",
        "- Normal: the vertical direction of the image. The positive sense is up. Down is the opposite sense.",
        "- Lateral: the horizontal direction of the image. The positive sense is right. Left is the opposite sense.",
        "",
        "Scene",
        "Two calibrated, synchronously exposed cameras photograph the same scene from different directions. The optical axis follows the camera axial direction, the image horizontal follows the camera lateral direction, and the image vertical follows the camera normal direction.",
        "One task gives lines through the world origin along the world axes. First determine the camera attitude, then project those lines into both images, and then solve the stereo extrinsics, the epipolar constraint, and triangulation.",
        "",
        "Angles",
        "Camera azimuth ψ is positive when the optical axis deflects to the right of the world.",
        "Camera pitch θ is positive when the optical axis deflects upward in the world.",
        "Camera roll φ is positive when the right edge of the image deflects downward in the camera.",
        "A camera uses the intrinsic order azimuth, pitch, and roll.",
        "",
        "Frames",
        f"World frame ({hand[spec['world'].get('chirality', 'left')]}): {axes_text(('Xw', 'Yw', 'Zw'), scene.wind)}.",
        f"Camera-1 frame ({hand[spec['camera1'].get('chirality', 'left')]}): {axes_text(('X1', 'Y1', 'Z1'), scene.cam1)}.",
        f"Camera-2 frame ({hand[spec['camera2'].get('chirality', 'left')]}): {axes_text(('X2', 'Y2', 'Z2'), scene.cam2)}.",
        "",
        "Given symbols",
        "The camera centers are C1=(x1,y1,z1) and C2=(x2,y2,z2). The intrinsics are K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]] and K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]. K acts in the order [lateral positive (right), normal positive (up), axial positive (forward)]. S reorders that camera's coordinates into this order, including sign flips. The effective intrinsic is K·S.",
        "The world origin is O=(0,0,0).",
    ])


if __name__ == "__main__":
    import json
    for tier in (25, 50, 75, 100):
        sample = build(tier)
        out = __file__.replace("cv_ctrl_generate.py", f"sample_cvctrl_{tier}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"tier {tier}: {len(sample['questions'])} blanks -> {out}")
