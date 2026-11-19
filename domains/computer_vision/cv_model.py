# -*- coding: utf-8 -*-
"""Scene model for the stereo-schlieren attitude task.

Every frame is declared by absolute directions: each coordinate axis names the
absolute role it lies on (axial / normal / lateral) and which way it points.
Chirality fixes the third axis. An Euler angle is bound to an absolute role
axis, and its positive sense follows from the declared axis vector and the
declared positive motion:

    sense = sign( (axis_vector x moving_vector) . toward_vector )

The attitude matrix is the active composition in the embedding (forward, right,
down): yaw about the normal, then pitch about the lateral, then roll about the
axial, with the derived signs. The sign of beta is therefore a result, not a
constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

FORWARD = (1, 0, 0)
RIGHT = (0, 1, 0)
DOWN = (0, 0, 1)
UP = (0, 0, -1)
LEFT = (0, -1, 0)
DIRECTIONS = {
    "forward": FORWARD, "rearward": (-1, 0, 0),
    "right": RIGHT, "left": LEFT,
    "down": DOWN, "up": UP,
}
ROLE_VECTOR = {"axial": FORWARD, "normal": UP, "lateral": RIGHT}


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _neg(v):
    return (-v[0], -v[1], -v[2])


def role_of(vector):
    for role, base in ROLE_VECTOR.items():
        if vector in (base, _neg(base)):
            return role
    raise ValueError(f"{vector} is not one of the six absolute directions")


def build_frame(owner: str, names: tuple[str, str, str], spec: dict) -> dict[str, tuple[int, int, int]]:
    declared: dict[str, tuple[int, int, int]] = {}
    for name in names:
        if name in spec:
            along = spec[name]["along"]
            points = spec[name]["points"]
            vector = DIRECTIONS[points]
            if role_of(vector) != along:
                raise ValueError(f"{owner}.{name} is along {along} but points {points}")
            declared[name] = vector
    if len(declared) < 2:
        raise ValueError(f"{owner} needs two declared axes")
    missing = [name for name in names if name not in declared]
    if len(missing) == 1:
        index = names.index(missing[0])
        first = names[(index + 1) % 3]
        second = names[(index + 2) % 3]
        vector = _cross(declared[first], declared[second])
        if spec.get("chirality", "right") == "left":
            vector = _neg(vector)
        declared[missing[0]] = vector
    return declared


def axis_on(axes: dict[str, tuple], role: str) -> tuple[str, tuple[int, int, int]]:
    for name, vector in axes.items():
        if role_of(vector) == role:
            return name, vector
    raise ValueError(f"no axis on role {role}")


def camera_layout_permutation(cam: dict, names: tuple[str, str, str]) -> sp.Matrix:
    """Signed permutation from declared camera axes to [lateral +right, normal +up, axial +forward].

    K is written for that order. Camera coordinates p satisfy S @ p in the order K expects,
    so the intrinsic that multiplies p is K @ S.
    """
    targets = (RIGHT, UP, FORWARD)
    return sp.Matrix([[_dot(target, cam[name]) for name in names] for target in targets])


def basis(axes: dict[str, tuple], names: tuple[str, str, str]) -> sp.Matrix:
    return sp.Matrix.hstack(*[sp.Matrix([[value] for value in axes[name]]) for name in names])


def rot(role: str, angle):
    """Active rotation about the absolute direction axis of a role."""
    c, s = sp.cos(angle), sp.sin(angle)
    if role == "normal":      # about up
        return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    if role == "lateral":     # about right
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if role == "axial":       # about forward
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    raise ValueError(role)


def sense(axis_vector, moving_vector, toward_vector) -> int:
    value = _dot(_cross(axis_vector, moving_vector), toward_vector)
    if value == 0:
        raise ValueError("declared positive motion is not a rotation about this axis")
    return 1 if value > 0 else -1


def _canonical(axis_vector, role: str) -> int:
    """Sign that turns a rotation about the declared axis into one about the role axis."""
    return 1 if _dot(axis_vector, ROLE_VECTOR[role]) > 0 else -1


DEFAULT_TOWARD = {
    "pan": "wind_right",       # optical axis toward wind right
    "tilt": "wind_up",         # optical axis toward wind up
    "roll": "camera_down",     # camera right edge down
    "attack": "body_up",       # nose toward body up
    "sideslip": "body_left",   # nose toward body left (airspeed on the right)
    "bank": "body_down",       # right wing down
}

ROLE_OF_DIRECTION = {
    "forward": "axial", "rearward": "axial",
    "right": "lateral", "left": "lateral",
    "down": "normal", "up": "normal",
}
NEGATIVE = {"rearward", "left", "down"}


@dataclass
class Scene:
    alpha: sp.Symbol = field(default_factory=lambda: sp.Symbol("α"))
    beta: sp.Symbol = field(default_factory=lambda: sp.Symbol("β"))
    gamma: sp.Symbol = field(default_factory=lambda: sp.Symbol("γ"))
    length: sp.Symbol = field(default_factory=lambda: sp.Symbol("L"))
    psi1: sp.Symbol = field(default_factory=lambda: sp.Symbol("ψ1"))
    theta1: sp.Symbol = field(default_factory=lambda: sp.Symbol("θ1"))
    phi1: sp.Symbol = field(default_factory=lambda: sp.Symbol("φ1"))
    psi2: sp.Symbol = field(default_factory=lambda: sp.Symbol("ψ2"))
    theta2: sp.Symbol = field(default_factory=lambda: sp.Symbol("θ2"))
    phi2: sp.Symbol = field(default_factory=lambda: sp.Symbol("φ2"))

    def declare(self, spec: dict) -> "Scene":
        self.wind = build_frame("wind", ("Xw", "Yw", "Zw"), spec["wind"])
        self.body = build_frame("body", ("Xb", "Yb", "Zb"), spec["body"])
        self.cam1 = build_frame("camera1", ("X1", "Y1", "Z1"), spec["camera1"])
        self.cam2 = build_frame("camera2", ("X2", "Y2", "Z2"), spec["camera2"])
        self.spec = spec
        self.angle_toward = dict(DEFAULT_TOWARD)
        for key, value in (spec.get("angles") or {}).items():
            self.angle_toward[key] = value if isinstance(value, str) else value["toward"]
        return self

    def _frame(self, owner: str, camera: dict | None = None) -> dict:
        if owner == "camera" and camera is not None:
            return camera
        return {"wind": self.wind, "body": self.body,
                "camera1": self.cam1, "camera2": self.cam2}[owner]

    def toward_vector(self, name: str, camera: dict | None = None):
        """Absolute direction name, or <frame>_<direction> resolved on that frame."""
        if name in DIRECTIONS:
            return DIRECTIONS[name]
        owner, direction = name.rsplit("_", 1)
        _, axis = axis_on(self._frame(owner, camera), ROLE_OF_DIRECTION[direction])
        return _neg(axis) if direction in NEGATIVE else axis

    # ---- angle senses -------------------------------------------------
    def camera_senses(self, cam: dict) -> tuple[int, int, int]:
        """Pan, tilt and roll senses; the positive side comes from the declaration."""
        _, c_normal = axis_on(cam, "normal")
        _, c_lateral = axis_on(cam, "lateral")
        _, c_axial = axis_on(cam, "axial")
        pan = (sense(c_normal, c_axial, self.toward_vector(self.angle_toward["pan"], cam))
               * _canonical(c_normal, "normal"))
        tilt = (sense(c_lateral, c_axial, self.toward_vector(self.angle_toward["tilt"], cam))
                * _canonical(c_lateral, "lateral"))
        roll = (sense(c_axial, c_lateral, self.toward_vector(self.angle_toward["roll"], cam))
                * _canonical(c_axial, "axial"))
        return pan, tilt, roll

    def body_senses(self) -> tuple[int, int, int]:
        """Attack, sideslip and bank senses; each positive side is declared."""
        _, b_normal = axis_on(self.body, "normal")
        _, b_lateral = axis_on(self.body, "lateral")
        _, b_axial = axis_on(self.body, "axial")
        attack = (sense(b_lateral, b_axial, self.toward_vector(self.angle_toward["attack"]))
                  * _canonical(b_lateral, "lateral"))
        sideslip = (sense(b_normal, b_axial, self.toward_vector(self.angle_toward["sideslip"]))
                    * _canonical(b_normal, "normal"))
        bank = (sense(b_axial, b_lateral, self.toward_vector(self.angle_toward["bank"]))
                * _canonical(b_axial, "axial"))
        return attack, sideslip, bank

    # ---- rotations ----------------------------------------------------
    def body_to_wind(self) -> sp.Matrix:
        """Wind-tunnel mechanism: pitch about the lateral, then yaw about the
        current normal, then roll about the body axis (pitch-yaw-roll)."""
        attack, sideslip, bank = self.body_senses()
        b0 = basis(self.body, ("Xb", "Yb", "Zb"))
        return sp.simplify(
            rot("lateral", attack * self.alpha)
            * rot("normal", sideslip * self.beta)
            * rot("axial", bank * self.gamma)
            * b0
        )

    def camera_to_ground(self, cam: dict, names: tuple[str, str, str],
                         psi, theta, phi) -> sp.Matrix:
        pan, tilt, roll = self.camera_senses(cam)
        b0 = basis(cam, names)
        return sp.simplify(
            rot("normal", pan * psi)
            * rot("lateral", tilt * theta)
            * rot("axial", roll * phi)
            * b0
        )

    def camera_pose(self, index: int, center) -> tuple[sp.Matrix, sp.Matrix]:
        """Camera pose in the wind frame: X_c = R_cw X_w + t_cw, center in wind coords."""
        if index == 1:
            r_cg = self.camera_to_ground(self.cam1, ("X1", "Y1", "Z1"),
                                         self.psi1, self.theta1, self.phi1)
        else:
            r_cg = self.camera_to_ground(self.cam2, ("X2", "Y2", "Z2"),
                                         self.psi2, self.theta2, self.phi2)
        r_cw = sp.simplify(r_cg.T)
        t_cw = sp.simplify(-r_cw * sp.Matrix(center))
        return r_cw, t_cw

    def nose(self) -> sp.Matrix:
        """Nose position in wind coordinates; the body origin is the nose."""
        return sp.simplify(self.length * self.body_to_wind() * sp.Matrix([1, 0, 0]))

    def extrinsics(self, t1, t2):
        r1, c1 = self.camera_pose(1, t1)
        r2, c2 = self.camera_pose(2, t2)
        r21 = sp.simplify(r2 * r1.T)
        t21 = sp.simplify(c2 - r21 * c1)
        return r1, r2, r21, t21
