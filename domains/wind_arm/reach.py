# -*- coding: utf-8 -*-
"""Format vectors and recompute nose ranges for one wrist declaration.

Collision answers are read back from the symbolic nose. They are not copied
from the default sample when the order or the installation changes.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from domains.wind_arm.model import Q, ArmModel, _align, pretty_expr, rot_about


ELL = sp.Symbol("ell", positive=True)
WING_RADIUS = "sqrt((L + L_测 + x_r - (b/2)*tan(Λ))^2 + (b/2)^2*(1 + (tan(Γ))^2))"
FINAL_REACH = f"max(L + L_测, {WING_RADIUS}) >= L_风/2"


def format_scalar(expr) -> str:
    expr = sp.simplify(expr)
    if expr == 0:
        return "0"
    if expr == ELL:
        return "L + L_测"
    if expr == -ELL:
        return "-(L + L_测)"
    text = sp.sstr(expr).replace("**", "^")
    text = text.replace("-ell*", "-(L + L_测)*")
    text = text.replace("ell*", "(L + L_测)*")
    if "ell" in text:
        text = text.replace("ell", "(L + L_测)")
    return text


def format_body_component(expr) -> str:
    expr = sp.expand(expr)
    if expr == 0:
        return "0"
    coefficient = expr.coeff(ELL) if expr.has(ELL) else 0
    rest = sp.simplify(sp.expand(expr - coefficient * ELL))
    rest_text = "" if rest == 0 else pretty_expr(rest)
    if coefficient == 0:
        return pretty_expr(expr) if rest.free_symbols else format_scalar(expr)
    if coefficient == 1 and rest == 0:
        return "L + L_测"
    if coefficient == 1 and rest_text.startswith("-"):
        return f"L + L_测 - {rest_text[1:]}"
    if coefficient == 1:
        return f"L + L_测 + {rest_text}"
    if coefficient == -1 and rest == 0:
        return "-(L + L_测)"
    head = format_scalar(coefficient * ELL)
    if not rest_text:
        return head
    if rest_text.startswith("-"):
        return f"{head} - {rest_text[1:]}"
    return f"{head} + {rest_text}"


def format_body_vector(vector: sp.Matrix) -> str:
    return "[" + ", ".join(format_body_component(vector[i]) for i in range(3)) + "]"


def format_wind_vector(vector: sp.Matrix) -> str:
    return "[" + ", ".join(format_scalar(vector[i]) for i in range(3)) + "]"


def principal_rotation(matrix: sp.Matrix):
    for axis in (0, 1, 2):
        for degree in (0, 30, -30, 60, -60, 90, -90, 180, -180):
            if sp.simplify(matrix - rot_about(axis, sp.rad(degree))) == sp.zeros(3):
                return axis, degree
    return None


def joint_axis(model: ArmModel, index: int):
    role = model.role_of_q(index)
    from domains.wind_arm.model import ROLE_POSITIVE

    axis, align = _align(model.wrist, ROLE_POSITIVE[role])
    epsilon = align * model.spec.joint_sign[role]
    return axis, epsilon


def theta_text(model: ArmModel) -> str | None:
    install = principal_rotation(model.S)
    if install is None:
        return None
    axis, epsilon = joint_axis(model, 2)
    if axis != install[0]:
        return None
    degree = install[1]
    if epsilon == 1 and degree == 0:
        return "q3"
    if epsilon == 1 and degree > 0:
        return f"q3+{degree}°"
    if epsilon == 1 and degree < 0:
        return f"q3{degree}°"
    if epsilon == -1 and degree == 0:
        return "-q3"
    if epsilon == -1 and degree > 0:
        return f"-q3+{degree}°"
    if epsilon == -1 and degree < 0:
        return f"-q3{degree}°"
    return None


def nose_ranges(model: ArmModel):
    """Min and max of each wind coordinate, as exact multiples of the nose length."""
    nose = sp.simplify(model.nose_wind())
    found = []
    for index in range(3):
        low, high = _range_ratio(nose[index], model)
        found.append((low, high))
    return found


def _range_ratio(expr, model: ArmModel):
    ratio = sp.simplify(sp.together(expr / ELL))
    if ratio == 0 or ratio.free_symbols == set():
        value = float(ratio)
        return value, value
    free = [symbol for symbol in Q if ratio.has(symbol)]
    function = sp.lambdify(free, ratio, "numpy")
    axes = []
    for symbol in free:
        low, high = model.limit_of_q(Q.index(symbol))
        axes.append(np.deg2rad(np.arange(low, high + 1, 1)))
    grids = np.meshgrid(*axes, indexing="ij")
    values = np.asarray(function(*grids), dtype=float)
    flat = np.ravel(values)
    order = np.argsort(flat)
    low_point = [grids[i].ravel()[order[0]] for i in range(len(free))]
    high_point = [grids[i].ravel()[order[-1]] for i in range(len(free))]
    low = _polish(function, low_point, axes, find_max=False)
    high = _polish(function, high_point, axes, find_max=True)
    return low, high


def _polish(function, start, axes, find_max: bool) -> float:
    point = [float(item) for item in start]
    span = [float(axis[-1] - axis[0]) for axis in axes]
    for _ in range(25):
        for index, axis in enumerate(axes):
            lo = float(axis[0])
            hi = float(axis[-1])
            left, right = point[index], point[index]
            width = max(span[index] / 30, 1e-4)
            left = max(lo, point[index] - width)
            right = min(hi, point[index] + width)
            for _step in range(18):
                cut = (right - left) / 3
                if cut <= 0:
                    break
                first = left + cut
                second = right - cut
                trial_a = point[:]
                trial_b = point[:]
                trial_a[index] = first
                trial_b[index] = second
                value_a = float(function(*trial_a))
                value_b = float(function(*trial_b))
                if find_max:
                    if value_a < value_b:
                        left = first
                    else:
                        right = second
                elif value_a > value_b:
                    left = first
                else:
                    right = second
            point[index] = (left + right) / 2
        span = [item / 3 for item in span]
    return float(function(*point))


def snap_ratio(value: float):
    if abs(value) <= 1e-6:
        return sp.Integer(0)
    atoms = [
        sp.Integer(1), sp.Rational(1, 2), sp.Rational(1, 4),
        sp.sqrt(2) / 2, sp.sqrt(3) / 2, sp.sqrt(3) / 4,
        sp.sqrt(2) / 4, sp.sqrt(6) / 4,
    ]
    candidates = [sp.Integer(0)]
    for item in atoms:
        candidates.extend((item, -item))
    best = min(candidates, key=lambda item: abs(float(item) - value))
    if abs(float(best) - value) <= 1e-5:
        return sp.simplify(best)
    guess = sp.nsimplify(value, constants={sp.sqrt(2), sp.sqrt(3), sp.sqrt(6)}, tolerance=1e-8)
    if abs(float(guess) - value) <= 1e-7:
        return sp.simplify(guess)
    return None


def format_ratio(exact) -> str:
    if exact == 0:
        return "0"
    negative = sp.simplify(exact) < 0
    magnitude = sp.simplify(-exact if negative else exact)
    if magnitude == 1:
        return "-(L + L_测)" if negative else "L + L_测"
    body = sp.sstr(magnitude).replace("**", "^")
    text = f"(L + L_测)*{body}"
    return f"-{text}" if negative else text


def format_interval(low: float, high: float) -> str:
    left = snap_ratio(low)
    right = snap_ratio(high)
    if left is None or right is None:
        raise ValueError(f"鼻尖坐标范围不能化成确切倍数：{low}, {high}")
    return f"[{format_ratio(left)}, {format_ratio(right)}]"


def max_abs_ratio(low: float, high: float) -> float:
    return max(abs(low), abs(high))


def wing_offset(model: ArmModel, lateral: int = 1) -> sp.Matrix:
    total = sp.Matrix(sp.expand(model.vertex_body(ELL) + model.tip_body()))
    if lateral < 0:
        index, _sign = _align(model.body, "right")
        total[index] = sp.expand(-total[index])
    return total


def perpendicular_square(model: ArmModel) -> str:
    total = wing_offset(model, 1)
    index, _sign = _align(model.body, "forward")
    square = 0
    for axis in range(3):
        if axis != index:
            square += sp.simplify(total[axis] ** 2)
    half = sp.Symbol("b") / 2
    gamma = sp.Symbol("Gamma")
    target = sp.simplify(half ** 2 * (1 + sp.tan(gamma) ** 2))
    if sp.simplify(sp.expand(square - target)) == 0:
        return "(b/2)^2*(1 + (tan(Γ))^2)"
    return pretty_expr(sp.factor(square))


def wall_reach(model: ArmModel, step: int = 5) -> float:
    """Smallest wall-normal component attained by some body direction.

    The value is 1 when every sampled direction can be placed on a wall normal
    inside the joint limits. A smaller value means the radius test is only an
    upper bound for that direction.
    """
    pose = model.joint_matrices[0] * model.joint_matrices[1] * model.joint_matrices[2] * model.S
    function = sp.lambdify(Q, pose, "numpy")
    grids = []
    for index in range(3):
        low, high = model.limit_of_q(index)
        grids.append(np.deg2rad(np.arange(low, high + 1, step)))
    mesh = np.meshgrid(*grids, indexing="ij")
    matrices = np.asarray(function(*mesh), dtype=float)
    matrices = np.moveaxis(matrices, [0, 1], [-2, -1]).reshape(-1, 3, 3)
    up_index, up_sign = _align(model.wind, "up")
    right_index, right_sign = _align(model.wind, "right")
    body = np.array(model.body).astype(float)
    count = 48
    index = np.arange(count)
    phi = np.arccos(1 - 2 * (index + 0.5) / count)
    theta = np.pi * (1 + 5 ** 0.5) * index
    directions = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), 1)
    worst = 1.0
    for direction in directions:
        points = matrices @ (body.T @ direction)
        up = np.max(np.abs(points[:, up_index] * up_sign))
        right = np.max(np.abs(points[:, right_index] * right_sign))
        worst = min(worst, max(float(up), float(right)))
    return float(worst)


def recover_joints(model: ArmModel) -> list[float]:
    names = {f"h{row + 1}{col + 1}": model.H[row, col] for row in range(3) for col in range(3)}
    found = []
    for part in model.extraction().split(";"):
        expr = sp.sympify(part.strip().replace("^", "**"), locals=names)
        found.append(float(sp.deg(expr)))
    return found
