# -*- coding: utf-8 -*-
"""Numeric self-check of the CV scene model (wind-tunnel convention: normal=up,
left-handed frames; camera yaw-pitch-roll, body pitch-yaw-roll)."""
import sys

sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

import sympy as sp

from domains.computer_vision import cv_model as cm

STANDARD = {
    "wind": {"chirality": "left", "Xw": {"along": "axial", "points": "forward"},
             "Zw": {"along": "normal", "points": "up"}},
    "body": {"chirality": "left", "Xb": {"along": "axial", "points": "forward"},
             "Zb": {"along": "normal", "points": "up"}},
    "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                "X1": {"along": "lateral", "points": "right"}},
    "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                "X2": {"along": "lateral", "points": "right"}},
}

ALPHA, BETA, GAMMA, L = 0.21, -0.34, 0.12, 2.5
CAM = {"psi1": 0.15, "theta1": -0.25, "phi1": 0.3,
       "psi2": -0.1, "theta2": 0.2, "phi2": -0.15}


def rz(t):
    return sp.Matrix([[sp.cos(t), -sp.sin(t), 0], [sp.sin(t), sp.cos(t), 0], [0, 0, 1]])


def ry(t):
    return sp.Matrix([[sp.cos(t), 0, sp.sin(t)], [0, 1, 0], [-sp.sin(t), 0, sp.cos(t)]])


def rx(t):
    return sp.Matrix([[1, 0, 0], [0, sp.cos(t), -sp.sin(t)], [0, sp.sin(t), sp.cos(t)]])


def close(a, b, tol=1e-9):
    return max(abs(float(sp.N(x - y))) for x, y in zip(a, b)) < tol


def main():
    scene = cm.Scene().declare(STANDARD)
    print("body senses (attack, sideslip, bank):", scene.body_senses())
    print("camera senses (pan, tilt, roll):", scene.camera_senses(scene.cam1))

    b0 = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # [Xb,Yb,Zb] = forward,right,up
    r_wb = scene.body_to_wind().subs({scene.alpha: ALPHA, scene.beta: BETA, scene.gamma: GAMMA})
    expected = ry(ALPHA) * rz(-BETA) * rx(GAMMA) * b0
    print("body_to_wind = Ry(a) Rz(-b) Rx(g) b0:", close(r_wb, expected))

    nose = scene.nose().subs({scene.alpha: ALPHA, scene.beta: BETA,
                              scene.gamma: GAMMA, scene.length: L})
    expected_nose = L * sp.Matrix([sp.cos(ALPHA) * sp.cos(BETA),
                                   -sp.sin(BETA),
                                   -sp.sin(ALPHA) * sp.cos(BETA)])
    print("nose = L(ca cb, -sb, -sa cb):", close(nose, expected_nose))

    zero = scene.camera_to_ground(scene.cam1, ("X1", "Y1", "Z1"), 0, 0, 0)
    print("camera basis at zero (columns X1 Y1 Z1):", zero.tolist())
    quarter = scene.camera_to_ground(scene.cam1, ("X1", "Y1", "Z1"), sp.pi / 2, 0, 0)
    optical = quarter * sp.Matrix([0, 0, 1])
    print("optical axis after pan +90 = right:", close(optical, sp.Matrix([0, 1, 0])))

    subs = {scene.psi1: CAM["psi1"], scene.theta1: CAM["theta1"], scene.phi1: CAM["phi1"],
            scene.psi2: CAM["psi2"], scene.theta2: CAM["theta2"], scene.phi2: CAM["phi2"]}
    r1, c1 = scene.camera_pose(1, (3, -2, 1))
    r2, c2 = scene.camera_pose(2, (3, 2, sp.Rational(6, 5)))
    r1, c1, r2, c2 = (item.subs(subs) for item in (r1, c1, r2, c2))
    r21 = sp.simplify(r2 * r1.T)
    t21 = sp.simplify(c2 - r21 * c1)
    print("R21 orthonormal:", close(r21 * r21.T, sp.eye(3)))
    point = sp.Matrix([1.2, -0.4, 0.7])
    world = r1.T * (point - c1)
    print("extrinsic 1->2 consistent:", close(r2 * world + c2, r21 * point + t21))

    flipped = {key: dict(value) for key, value in STANDARD.items()}
    flipped["camera1"] = {"chirality": "left",
                          "Z1": {"along": "axial", "points": "forward"},
                          "X1": {"along": "lateral", "points": "left"}}
    other = cm.Scene().declare(flipped)
    print("camera X1 left -> axes", other.cam1, "senses", other.camera_senses(other.cam1))

    up = {key: dict(value) for key, value in STANDARD.items()}
    up["body"] = {"chirality": "left", "Xb": {"along": "axial", "points": "forward"},
                  "Zb": {"along": "normal", "points": "down"}}
    body_down = cm.Scene().declare(up)
    print("body normal down -> axes", body_down.body, "senses", body_down.body_senses())


if __name__ == "__main__":
    main()
