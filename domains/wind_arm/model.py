# -*- coding: utf-8 -*-
"""Kinematics for the wind-tunnel spherical wrist.

Seven declarations can change: the three frames, the intrinsic axis order,
the joint positive senses, the zero installation, and the airflow-angle signs.
Joint limits stay on the absolute axes. Axial lock at zero is fixed: body
forward coincides with wind forward. Collision reachability and the inverse
extraction are recomputed for the active declaration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

ABS = {
    "forward": sp.Matrix([1, 0, 0]),
    "up": sp.Matrix([0, 1, 0]),
    "right": sp.Matrix([0, 0, 1]),
    "rearward": sp.Matrix([-1, 0, 0]),
    "down": sp.Matrix([0, -1, 0]),
    "left": sp.Matrix([0, 0, -1]),
}
OPPOSITE = {
    "forward": "rearward",
    "rearward": "forward",
    "up": "down",
    "down": "up",
    "right": "left",
    "left": "right",
}
ROLE_OF = {"forward": "axial", "rearward": "axial", "up": "normal", "down": "normal", "right": "lateral", "left": "lateral"}
ROLE_POSITIVE = {"axial": "forward", "normal": "up", "lateral": "right"}
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
INDEX_AXIS = {0: "X", 1: "Y", 2: "Z"}
LIMITS = {"normal": (-60, 60), "lateral": (-90, 90), "axial": (-90, 90)}
Q = sp.symbols("q1 q2 q3", real=True)


def _cross(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def rot_about(axis: int, theta) -> sp.Matrix:
    c, s = sp.cos(theta), sp.sin(theta)
    if axis == 0:
        return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_matrix(declared: dict[str, str], chirality: str = "right") -> sp.Matrix:
    """Columns are the coordinate axes in the forward-up-right basis."""
    cols: dict[str, sp.Matrix] = {name: ABS[pointing] for name, pointing in declared.items()}
    missing = ({"X", "Y", "Z"} - set(cols)).pop()
    sign = 1 if chirality == "right" else -1
    if missing == "Z":
        cols["Z"] = sign * _cross(cols["X"], cols["Y"])
    elif missing == "X":
        cols["X"] = sign * _cross(cols["Y"], cols["Z"])
    else:
        cols["Y"] = sign * _cross(cols["Z"], cols["X"])
    return sp.Matrix.hstack(cols["X"], cols["Y"], cols["Z"])


def _align(basis: sp.Matrix, direction: str) -> tuple[int, int]:
    target = ABS[direction]
    for index in range(3):
        column = basis[:, index]
        if column == target:
            return index, 1
        if column == -target:
            return index, -1
    raise ValueError(f"no axis along {direction}")


def fmt_expr(expr) -> str:
    value = sp.simplify(expr)
    if value == 0:
        return "0"
    names = {symbol.name for symbol in value.free_symbols}
    if names & {"x_r", "b", "Lambda", "Gamma"}:
        return pretty_expr(sp.expand(value))
    return sp.sstr(value).replace("**", "^")


def pretty_expr(expr) -> str:
    expr = sp.simplify(expr)
    if expr == 0:
        return "0"
    if expr.is_Add:
        parts = []
        for term in sp.Add.make_args(sp.expand(expr)):
            if term.could_extract_minus_sign():
                parts.append(("neg", pretty_expr(sp.simplify(-term))))
            else:
                parts.append(("pos", pretty_expr(term)))
        parts.sort(key=lambda item: (0 if item[1] == "x_r" or item[1].startswith("x_r ") else 1, item[0] == "neg", item[1]))
        text = ""
        for kind, body in parts:
            if not text:
                text = body if kind == "pos" else f"-{body}"
            elif kind == "pos":
                text += f" + {body}"
            else:
                text += f" - {body}"
        return text
    if expr.is_Mul:
        coef, factors = expr.as_coeff_mul()
        factor_text = [pretty_expr(factor) for factor in factors]
        if coef == -1:
            return "-" + "*".join(factor_text)
        if coef == 1:
            return "*".join(factor_text)
        if coef == sp.Rational(1, 2) and factor_text == ["b"]:
            return "b/2"
        if coef == sp.Rational(1, 2):
            return "(b/2)*" + "*".join(part for part in factor_text if part != "b") if "b" in factor_text else f"({pretty_expr(coef)})*" + "*".join(factor_text)
        if coef == sp.Rational(-1, 2) and "b" in factor_text:
            rest = "*".join(part for part in factor_text if part != "b")
            return f"-(b/2)*{rest}" if rest else "-b/2"
        number = pretty_expr(coef)
        return f"{number}*" + "*".join(factor_text)
    if expr.is_Pow:
        base, exp = expr.args
        return f"{pretty_expr(base)}^{pretty_expr(exp)}"
    if expr.func == sp.sqrt:
        return f"sqrt({pretty_expr(expr.args[0])})"
    if expr.func in (sp.sin, sp.cos, sp.tan):
        name = {sp.sin: "sin", sp.cos: "cos", sp.tan: "tan"}[expr.func]
        return f"{name}({pretty_expr(expr.args[0])})"
    if expr.is_Rational or expr.is_Integer:
        return str(expr)
    if expr == sp.Symbol("Lambda"):
        return "Λ"
    if expr == sp.Symbol("Gamma"):
        return "Γ"
    if isinstance(expr, sp.Symbol):
        return expr.name
    text = sp.sstr(expr).replace("**", "^").replace("Lambda", "Λ").replace("Gamma", "Γ")
    return text


def fmt_matrix(matrix: sp.Matrix) -> str:
    rows = []
    for i in range(3):
        rows.append("[" + ", ".join(fmt_expr(matrix[i, j]) for j in range(3)) + "]")
    return "[" + ", ".join(rows) + "]"


def fmt_vector(vector: sp.Matrix) -> str:
    return "[" + ", ".join(fmt_expr(vector[i]) for i in range(3)) + "]"


def _angle_matches(principal, angle) -> bool:
    difference = sp.simplify(principal - angle)
    if difference == 0:
        return True
    value = difference.evalf()
    return value.is_real and abs(float(value)) < 1e-8


def _short_expr(text: str) -> bool:
    return len(text) <= 48


def _signed_ratio(ratio) -> str:
    ratio = sp.simplify(ratio)
    if ratio == 0:
        return "0"
    if ratio.could_extract_minus_sign():
        return f"-{fmt_expr(sp.simplify(-ratio))}"
    return fmt_expr(ratio)


def _signed_arctan(angle, sine=None, cosine=None) -> str:
    angle = sp.simplify(angle)
    if angle == 0:
        return "arctan(0)"
    if sine is None or cosine is None:
        sine = sp.simplify(sp.sin(angle))
        cosine = sp.simplify(sp.cos(angle))
    else:
        sine = sp.simplify(sine)
        cosine = sp.simplify(cosine)
    if cosine != 0:
        ratio = sp.simplify(sine / cosine)
        ratio_text = _signed_ratio(ratio)
        if _short_expr(ratio_text) and _angle_matches(sp.atan(ratio), angle):
            if ratio_text.startswith("-"):
                return f"-arctan({ratio_text[1:]})"
            return f"arctan({ratio_text})"
    return f"atan2({_signed_ratio(sine)}, {_signed_ratio(cosine)})"


def _signed_arcsin(angle, sine=None, cosine=None) -> str:
    angle = sp.simplify(angle)
    if angle == 0:
        return "arcsin(0)"
    if sine is None:
        sine = sp.simplify(sp.sin(angle))
    else:
        sine = sp.simplify(sine)
    sine_text = _signed_ratio(sine)
    if _short_expr(sine_text) and _angle_matches(sp.asin(sine), angle):
        if sine_text.startswith("-"):
            return f"-arcsin({sine_text[1:]})"
        return f"arcsin({sine_text})"
    if cosine is None:
        cosine = sp.simplify(sp.cos(angle))
    else:
        cosine = sp.simplify(cosine)
    return f"atan2({_signed_ratio(sine)}, {_signed_ratio(cosine)})"


@dataclass
class ArmSpec:
    wind_declared: dict[str, str] = field(default_factory=lambda: {"X": "forward", "Y": "up"})
    wind_chirality: str = "right"
    wrist_declared: dict[str, str] = field(default_factory=lambda: {"X": "forward", "Y": "up"})
    wrist_chirality: str = "right"
    body_declared: dict[str, str] = field(default_factory=lambda: {"X": "forward", "Y": "up"})
    body_chirality: str = "right"
    order: tuple[str, str, str] = ("normal", "lateral", "axial")
    joint_sign: dict[str, int] = field(default_factory=lambda: {"normal": 1, "lateral": 1, "axial": 1})
    install_up: str = "right"
    alpha_toward: str = "down"
    beta_toward: str = "right"
    gamma_sense: int = 1
    role_deg: dict[str, int] = field(default_factory=lambda: {"normal": 0, "lateral": 30, "axial": -60})

    def validate(self) -> None:
        if len(set(self.order)) != 3 or set(self.order) != {"normal", "lateral", "axial"}:
            raise ValueError("旋转顺序必须是法向、侧向、轴向的一个排列")
        if self.install_up in {"forward", "rearward"}:
            raise ValueError("零位法向不能与轴向重合")
        if ROLE_OF[self.alpha_toward] != "normal":
            raise ValueError("迎角正方向必须是机体上下")
        if ROLE_OF[self.beta_toward] != "lateral":
            raise ValueError("侧滑角正方向必须是机体左右")


class ArmModel:
    def __init__(self, spec: ArmSpec | None = None):
        self.spec = spec or ArmSpec()
        self.spec.validate()
        self.wind = frame_matrix(self.spec.wind_declared, self.spec.wind_chirality)
        self.wrist = frame_matrix(self.spec.wrist_declared, self.spec.wrist_chirality)
        self.body = frame_matrix(self.spec.body_declared, self.spec.body_chirality)
        self.q_symbols = list(Q)
        self.joint_matrices = [self._joint_matrix(role, Q[index]) for index, role in enumerate(self.spec.order)]
        self.R = sp.simplify(self.joint_matrices[0] * self.joint_matrices[1] * self.joint_matrices[2])
        self.S = self._install()
        self.D_symbolic = sp.simplify(self.R * self.S)
        self.q_degrees = [self.spec.role_deg[role] for role in self.spec.order]
        subs = {Q[i]: sp.rad(self.q_degrees[i]) for i in range(3)}
        self.D = sp.simplify(self.D_symbolic.subs(subs))
        self.H = sp.simplify(self.D * self.S.T)
        self.airspeed_wind = self.wind.T * ABS["forward"]
        self.airspeed_body = sp.simplify(self.D.T * self.airspeed_wind)
        self.airspeed_abs = sp.simplify(self.body * self.airspeed_body)
        self.alpha = self._alpha()
        self.beta = self._beta()
        self.gamma = self._gamma()

    def role_of_q(self, index: int) -> str:
        return self.spec.order[index]

    def limit_of_q(self, index: int) -> tuple[int, int]:
        return LIMITS[self.role_of_q(index)]

    def axis_name(self, frame: str, direction: str) -> str:
        basis = {"wind": self.wind, "wrist": self.wrist, "body": self.body}[frame]
        index, sign = _align(basis, direction)
        prefix = {"wind": "FD", "wrist": "MD", "body": "TZ"}[frame]
        name = f"{INDEX_AXIS[index]}{prefix}"
        return name if sign > 0 else f"-{name}"

    def positive_name(self, frame: str, direction: str) -> str:
        text = self.axis_name(frame, direction)
        return text[1:] if text.startswith("-") else text

    def _joint_matrix(self, role: str, angle) -> sp.Matrix:
        direction = ROLE_POSITIVE[role]
        axis, align = _align(self.wrist, direction)
        epsilon = align * self.spec.joint_sign[role]
        return rot_about(axis, epsilon * angle)

    def _install(self) -> sp.Matrix:
        body_forward = ABS["forward"]
        body_up = ABS[self.spec.install_up]
        body_right = _cross(body_forward, body_up)
        mount = sp.Matrix.hstack(body_forward, body_up, body_right)
        return sp.simplify(self.wrist.T * mount * self.body)

    def turn_of_forward(self, index: int) -> str:
        role = self.role_of_q(index)
        matrix = self._joint_matrix(role, sp.rad(1))
        forward = self.wrist.T * ABS["forward"]
        moved = sp.simplify(self.wrist * matrix * forward)
        delta = sp.simplify(moved - ABS["forward"])
        best = None
        best_score = None
        for name in ("up", "down", "left", "right"):
            score = sp.simplify(delta.dot(ABS[name]))
            if best_score is None or score > best_score:
                best, best_score = name, score
        names = {"up": "上方", "down": "下方", "left": "左侧", "right": "右侧"}
        return names[best]

    def turn_of_up(self, index: int) -> str:
        role = self.role_of_q(index)
        matrix = self._joint_matrix(role, sp.rad(1))
        up = self.wrist.T * ABS["up"]
        moved = sp.simplify(self.wrist * matrix * up)
        delta = sp.simplify(moved - ABS["up"])
        best = None
        best_score = None
        for name in ("up", "down", "left", "right", "forward", "rearward"):
            score = sp.simplify(delta.dot(ABS[name]))
            if best_score is None or score > best_score:
                best, best_score = name, score
        names = {"up": "上方", "down": "下方", "left": "左侧", "right": "右侧", "forward": "前方", "rearward": "后方"}
        return names[best]

    def _alpha(self):
        forward = self.airspeed_abs[0]
        up = self.airspeed_abs[1]
        toward_down = sp.atan2(-up, forward)
        if self.spec.alpha_toward == "down":
            return sp.simplify(toward_down)
        return sp.simplify(-toward_down)

    def _beta(self):
        right = self.airspeed_abs[2]
        angle = sp.asin(sp.simplify(right))
        if self.spec.beta_toward == "right":
            return sp.simplify(angle)
        return sp.simplify(-angle)

    def _gamma(self):
        body_forward = self.body.T * ABS["forward"]
        body_up = self.body.T * ABS["up"]
        t = sp.simplify(self.D * body_forward)
        n = sp.simplify(self.D * body_up)
        wind_up = sp.simplify(self.wind.T * ABS["up"])
        dot = sp.simplify(wind_up.dot(t))
        projected = sp.simplify(wind_up - dot * t)
        norm = sp.simplify(sp.sqrt(projected.dot(projected)))
        u = sp.simplify(projected / norm)
        r = _cross(t, u)
        angle = sp.atan2(sp.simplify(n.dot(r)), sp.simplify(n.dot(u)))
        return sp.simplify(self.spec.gamma_sense * angle)

    def _alpha_pair(self):
        forward = sp.simplify(self.airspeed_abs[0])
        up = sp.simplify(self.airspeed_abs[1])
        if self.spec.alpha_toward == "down":
            return sp.simplify(-up), forward
        return up, forward

    def _beta_pair(self):
        right = sp.simplify(self.airspeed_abs[2])
        sine = right if self.spec.beta_toward == "right" else sp.simplify(-right)
        cosine = sp.simplify(sp.sqrt(1 - sine ** 2))
        return sine, cosine

    def _gamma_pair(self):
        body_forward = self.body.T * ABS["forward"]
        body_up = self.body.T * ABS["up"]
        direction = sp.simplify(self.D * body_forward)
        normal = sp.simplify(self.D * body_up)
        wind_up = sp.simplify(self.wind.T * ABS["up"])
        projected = sp.simplify(wind_up - wind_up.dot(direction) * direction)
        norm = sp.simplify(sp.sqrt(projected.dot(projected)))
        reference = sp.simplify(projected / norm)
        side = _cross(direction, reference)
        sine = sp.simplify(normal.dot(side))
        cosine = sp.simplify(normal.dot(reference))
        if self.spec.gamma_sense < 0:
            sine = sp.simplify(-sine)
        return sine, cosine

    def airflow_text(self) -> str:
        return ", ".join([
            _signed_arctan(self.alpha, *self._alpha_pair()),
            _signed_arcsin(self.beta, *self._beta_pair()),
            _signed_arcsin(self.gamma, *self._gamma_pair()),
        ])

    def vertex_body(self, length) -> sp.Matrix:
        return sp.simplify(length * (self.body.T * ABS["forward"]))

    def tip_body(self) -> sp.Matrix:
        forward = sp.Symbol("x_r") - (sp.Symbol("b") / 2) * sp.tan(sp.Symbol("Lambda"))
        up = (sp.Symbol("b") / 2) * sp.tan(sp.Symbol("Gamma"))
        right = sp.Symbol("b") / 2
        absolute = forward * ABS["forward"] + up * ABS["up"] + right * ABS["right"]
        return sp.simplify(self.body.T * absolute)

    def tip_text(self) -> str:
        return _named(self.tip_body())

    def vertex_text(self, symbol: str = "L + L_测") -> str:
        length = sp.Symbol("ell", positive=True)
        vector = self.vertex_body(length)
        return fmt_vector(vector).replace("ell", f"({symbol})" if " " in symbol else symbol)

    def nose_wind(self):
        ell = sp.Symbol("ell", positive=True)
        q1, q2, q3 = Q
        pose = self.joint_matrices[0] * self.joint_matrices[1] * self.joint_matrices[2] * self.S
        return sp.simplify(pose * self.vertex_body(ell))

    def wrist_pose(self) -> sp.Matrix:
        return sp.simplify(self.joint_matrices[0] * self.joint_matrices[1] * self.joint_matrices[2])

    def extraction(self) -> str:
        """atan2 formula for q1; q2; q3 on the branch where cos(q2) stays positive."""
        pose = self.wrist_pose()
        q1, q2, q3 = Q
        cells = [[sp.simplify(sp.expand_trig(pose[i, j])) for j in range(3)] for i in range(3)]
        sine = _find_cell(cells, [sp.sin(q2), -sp.sin(q2)])
        if sine is None:
            raise ValueError("中间角没有独立的正弦元素")
        row, col, expr = sine
        sign = 1 if expr == sp.sin(q2) else -1
        cos_cells = _cos_factor_cells(cells, q2)
        q1_cells = [item for item in cos_cells if item[2].has(q1) and not item[2].has(q3)]
        q3_cells = [item for item in cos_cells if item[2].has(q3) and not item[2].has(q1)]
        hypot_names = [_h(item[0], item[1]) for item in q1_cells[:2]]
        if len(hypot_names) < 2:
            hypot_names = [_h(item[0], item[1]) for item in cos_cells[:2]]
        hypot = "sqrt(" + " + ".join(f"{name}^2" for name in hypot_names) + ")"
        sine_text = _h(row, col) if sign > 0 else f"-{_h(row, col)}"
        return "; ".join([
            _atan2_pair(q1_cells, q1),
            f"atan2({sine_text}, {hypot})",
            _atan2_pair(q3_cells, q3),
        ])


def _h(row: int, col: int) -> str:
    return f"h{row + 1}{col + 1}"


def _find_cell(cells, targets):
    for row in range(3):
        for col in range(3):
            if cells[row][col] in targets:
                return row, col, cells[row][col]
    return None


def _cos_factor_cells(cells, middle):
    found = []
    for row in range(3):
        for col in range(3):
            expr = cells[row][col]
            if not expr.has(sp.cos(middle)) or expr.has(sp.sin(middle)):
                continue
            ratio = sp.simplify(expr / sp.cos(middle))
            if ratio.has(middle):
                continue
            found.append((row, col, ratio))
    return found


def _atan2_pair(cells, angle) -> str:
    sine = None
    cosine = None
    for row, col, ratio in cells:
        if ratio == sp.sin(angle):
            sine = (row, col, 1)
        elif ratio == -sp.sin(angle):
            sine = (row, col, -1)
        elif ratio == sp.cos(angle):
            cosine = (row, col, 1)
        elif ratio == -sp.cos(angle):
            cosine = (row, col, -1)
    if sine is None or cosine is None:
        raise ValueError(f"无法分离 {angle}")
    sine_text = _h(*sine[:2]) if sine[2] > 0 else f"-{_h(*sine[:2])}"
    cos_text = _h(*cosine[:2]) if cosine[2] > 0 else f"-{_h(*cosine[:2])}"
    return f"atan2({sine_text}, {cos_text})"


def _named(vector: sp.Matrix) -> str:
    names = {
        sp.Symbol("x_r"): "x_r",
        sp.Symbol("b"): "b",
        sp.Symbol("Lambda"): "Λ",
        sp.Symbol("Gamma"): "Γ",
    }
    text = fmt_vector(vector)
    text = text.replace("Lambda", "Λ").replace("Gamma", "Γ")
    return text


def default_model() -> ArmModel:
    return ArmModel(ArmSpec())
