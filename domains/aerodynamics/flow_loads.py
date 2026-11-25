# -*- coding: utf-8 -*-
"""Aerodynamic load transform and moment-reference change.

Axes are built from the body components of the unit airspeed. T_{i←j} maps
components from frame j to frame i. Force and moment at one point use the same
matrix. The reference-point change is M_G = M_A - r_AG × F in a right-handed
component basis.
"""

from __future__ import annotations

import sympy as sp

from flight_frame_model import DefinitionError


EYE = sp.eye(3)
DRAG, SIDE, LIFT = sp.symbols("D Y L", real=True)
ROLL_A, PITCH_A, YAW_A = sp.symbols("l_a m_a n_a", real=True)
QBAR, RHO, SPEED, AREA = sp.symbols("q_bar rho V S", positive=True)
SPAN, CHORD = sp.symbols("b c_bar", positive=True)
CLA, CMA, CNA = sp.symbols("C_la C_ma C_na", real=True)
CL_A, CM_A, CN_A = sp.symbols("C_lA C_mA C_nA", real=True)
CX, CYB, CZ = sp.symbols("C_X C_Yb C_Z", real=True)
BODY_X, BODY_Y, BODY_Z = sp.symbols("X Y_b Z", real=True)
BODY_L, BODY_M, BODY_N = sp.symbols("l_A m_A n_A", real=True)
ARM_X, ARM_Y, ARM_Z = sp.symbols("x y z", real=True)
XN, XGE, YGE, ZGE = sp.symbols("X_N x_Ge y_Ge z_Ge", real=True)
LF, LA = sp.symbols("L_f L_A", positive=True)


def _cross(left: sp.Matrix, right: sp.Matrix) -> sp.Matrix:
    return sp.Matrix([
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ])


def _principal(expr, attack, sideslip):
    expr = sp.simplify(expr)
    for angle in (attack, sideslip):
        cosine = sp.cos(angle)
        expr = expr.subs(sp.Abs(cosine), cosine)
        expr = expr.subs(sp.sqrt(cosine**2), cosine)
    return sp.simplify(expr)


def _apply(matrix, attack, sideslip):
    return matrix.applyfunc(lambda item: _principal(item, attack, sideslip))


def _sign(value) -> str:
    number = sp.N(value)
    if abs(number) < sp.Float("1e-8"):
        return "零"
    return "正" if number > 0 else "负"


def build_loads(body_speed: sp.Matrix, attack, sideslip, show, lift_z: int = -1, side_y: int = 1) -> dict:
    """Return displayed load answers derived from the unit airspeed in body axes.

    lift_z is the sign of L along Z_a. side_y is the sign of Y along Y_a.
    The standard declaration uses lift_z=-1 and side_y=+1.
    """
    air_x = sp.Matrix([_principal(item, attack, sideslip) for item in body_speed])
    if sp.simplify(air_x.dot(air_x) - 1) != 0:
        raise DefinitionError("The airflow forward axis is not a unit vector.")
    plane_scale = sp.sqrt(_principal(air_x[0]**2 + air_x[2]**2, attack, sideslip))
    stable_x = sp.Matrix([
        _principal(air_x[0] / plane_scale, attack, sideslip),
        0,
        _principal(air_x[2] / plane_scale, attack, sideslip),
    ])
    stable_y = sp.Matrix([0, 1, 0])
    stable_z = _apply(_cross(stable_x, stable_y), attack, sideslip)
    down_scale = sp.sqrt(_principal(air_x[0]**2 + air_x[2]**2, attack, sideslip))
    air_z = _apply(sp.Matrix([-air_x[2], 0, air_x[0]]) / down_scale, attack, sideslip)
    air_y = _apply(_cross(air_z, air_x), attack, sideslip)
    stable_basis = sp.Matrix.hstack(stable_x, stable_y, stable_z)
    air_basis = sp.Matrix.hstack(air_x, air_y, air_z)
    to_stable = _apply(stable_basis.T, attack, sideslip)
    to_air = _apply(air_basis.T, attack, sideslip)
    stable_to_air = _apply(to_air * to_stable.T, attack, sideslip)
    if sp.simplify(stable_to_air * to_stable - to_air) != sp.zeros(3):
        raise DefinitionError("The two airflow legs do not multiply to the direct transform.")
    for name, basis in (("stability", stable_basis), ("airflow", air_basis)):
        if sp.simplify(basis.T * basis - EYE) != sp.zeros(3) or sp.simplify(basis.det() - 1) != 0:
            raise DefinitionError(f"The {name} basis is not a right-handed orthonormal frame.")

    body_from_air = _apply(to_air.T, attack, sideslip)
    if lift_z not in (-1, 1) or side_y not in (-1, 1):
        raise DefinitionError("Lift and side signs must be -1 or +1.")
    airflow_force = sp.Matrix([-DRAG, side_y * SIDE, lift_z * LIFT])
    body_force = _apply(body_from_air * airflow_force, attack, sideslip)
    airflow_moment = sp.Matrix([ROLL_A, PITCH_A, YAW_A])
    body_moment = _apply(body_from_air * airflow_moment, attack, sideslip)
    if sp.simplify(body_force.dot(body_force) - airflow_force.dot(airflow_force)) != 0:
        raise DefinitionError("Force magnitude changes across the coordinate transform.")
    if sp.simplify(body_moment.dot(body_moment) - airflow_moment.dot(airflow_moment)) != 0:
        raise DefinitionError("Moment magnitude changes across the same-point transform.")

    scale = sp.diag(SPAN, CHORD, SPAN)
    coefficient = sp.Matrix([CLA, CMA, CNA])
    by_dimension = sp.simplify(scale.inv() * body_from_air * scale * coefficient)
    by_force = sp.simplify(scale.inv() * body_from_air * (QBAR * AREA * scale * coefficient) / (QBAR * AREA))
    if sp.simplify(by_dimension - by_force) != sp.zeros(3, 1):
        raise DefinitionError("The coefficient scale does not match the dimensional moment.")

    arm = sp.Matrix([ARM_X, ARM_Y, ARM_Z])
    force = sp.Matrix([BODY_X, BODY_Y, BODY_Z])
    moment_a = sp.Matrix([BODY_L, BODY_M, BODY_N])
    moment_g = sp.simplify(moment_a - _cross(arm, force))
    recovered = sp.simplify(moment_g + _cross(arm, force))
    if sp.simplify(recovered - moment_a) != sp.zeros(3, 1):
        raise DefinitionError("Changing the reference point and changing it back does not recover the moment.")
    _check_order(to_air, body_from_air, attack, sideslip)

    probe_alpha = {attack: sp.Float("0.4"), sideslip: 0}
    probe_beta = {attack: 0, sideslip: sp.Float("0.4")}
    stable_forward = stable_x.subs(probe_alpha)
    air_forward = air_x.subs(probe_beta)
    lift_only = {DRAG: 0, SIDE: 0, LIFT: 1, attack: sp.Float("0.5"), sideslip: 0}
    drag_and_lift = [
        body_force[0].subs({DRAG: drag, SIDE: 0, LIFT: lift, attack: sp.Float("0.5"), sideslip: 0})
        for drag, lift in ((1, sp.Float("0.1")), (sp.Float("0.1"), 1))
    ]
    example = {
        attack: sp.rad(30),
        sideslip: 0,
        DRAG: 2,
        SIDE: 0,
        LIFT: 2 * sp.sqrt(3),
    }
    example_force = sp.Matrix([sp.nsimplify(sp.simplify(item.subs(example))) for item in body_force])
    pitch_axis = air_y.subs({attack: sp.Float("0.2"), sideslip: 0})
    if abs(sp.N(pitch_axis[1]) - 1) < sp.Float("1e-8"):
        pitch_relation = "重合"
    elif abs(sp.N(pitch_axis.dot(sp.Matrix([0, 1, 0])))) < sp.Float("1e-8"):
        pitch_relation = "相互垂直"
    else:
        pitch_relation = "方向无法确定"
    roll_into_pitch = body_moment[1].subs({attack: 0, sideslip: sp.Float("0.4"), ROLL_A: 1, PITCH_A: 0, YAW_A: 0})

    def component(expr, canonical=None) -> str:
        simplified = _principal(sp.expand(expr), attack, sideslip)
        if canonical is not None and sp.simplify(simplified - canonical[0]) == 0:
            return canonical[1]
        return show(simplified)

    cosine_a, sine_a = sp.cos(attack), sp.sin(attack)
    cosine_b, sine_b = sp.cos(sideslip), sp.sin(sideslip)
    axial = -DRAG * cosine_a * cosine_b - SIDE * cosine_a * sine_b + LIFT * sine_a
    lateral = -DRAG * sine_b + SIDE * cosine_b
    normal = -DRAG * sine_a * cosine_b - SIDE * sine_a * sine_b - LIFT * cosine_a
    roll = ROLL_A * cosine_a * cosine_b - PITCH_A * cosine_a * sine_b - YAW_A * sine_a
    pitch = ROLL_A * sine_b + PITCH_A * cosine_b
    yaw = ROLL_A * sine_a * cosine_b - PITCH_A * sine_a * sine_b + YAW_A * cosine_a
    pitch_coeff = (SPAN / CHORD) * CLA * sine_b + CMA * cosine_b

    lift_symbol = "-L" if lift_z < 0 else "L"
    side_symbol = "Y" if side_y > 0 else "-Y"
    coeff_l = "-C_L" if lift_z < 0 else "C_L"
    coeff_y = "C_Ya" if side_y > 0 else "-C_Ya"
    dimensional_l = "-q_bar*S*C_L" if lift_z < 0 else "q_bar*S*C_L"
    dimensional_y = "q_bar*S*C_Ya" if side_y > 0 else "-q_bar*S*C_Ya"
    return {
        "to_stable": _matrix(to_stable, show),
        "stable_to_air": _matrix(stable_to_air, show),
        "to_air": _matrix(to_air, show),
        "body_from_air": _matrix(body_from_air, show),
        "lift_axis": "-Z_a" if lift_z < 0 else "+Z_a",
        "side_axis": "+Y_a" if side_y > 0 else "-Y_a",
        "coeff_symbol": f"[-C_D, {coeff_y}, {coeff_l}]",
        "air_force": f"[-D, {side_symbol}, {lift_symbol}]",
        "air_moment": "[l_a, m_a, n_a]",
        "body_x": component(body_force[0], (axial, "-D*cos(α)*cos(β) - Y*cos(α)*sin(β) + L*sin(α)")),
        "body_y": component(body_force[1], (lateral, "-D*sin(β) + Y*cos(β)")),
        "body_z": component(body_force[2], (normal, "-D*sin(α)*cos(β) - Y*sin(α)*sin(β) - L*cos(α)")),
        "body_l": component(body_moment[0], (roll, "l_a*cos(α)*cos(β) - m_a*cos(α)*sin(β) - n_a*sin(α)")),
        "body_m": component(body_moment[1], (pitch, "l_a*sin(β) + m_a*cos(β)")),
        "body_n": component(body_moment[2], (yaw, "l_a*sin(α)*cos(β) - m_a*sin(α)*sin(β) + n_a*cos(α)")),
        "example_force": _matrix_vector(example_force, show),
        "dynamic": "ρ*V^2/2",
        "coefficient_force": f"[-q_bar*S*C_D, {dimensional_y}, {dimensional_l}]",
        "pitch_coefficient": component(by_dimension[1], (pitch_coeff, "(b/c_bar)*C_la*sin(β) + C_ma*cos(β)")),
        "stable_side": "下方" if stable_forward[2] > 0 else "上方",
        "stable_z": _sign(stable_forward[2]),
        "air_side": "右方" if air_forward[1] > 0 else "左方",
        "air_y": _sign(air_forward[1]),
        "lift_x": _sign(body_force[0].subs(lift_only)),
        "lift_z": _sign(body_force[2].subs(lift_only)),
        "axial_sign": "不能确定" if _sign(drag_and_lift[0]) != _sign(drag_and_lift[1]) else f"一定为{_sign(drag_and_lift[0])}",
        "pitch_relation": pitch_relation,
        "roll_to_pitch": _sign(roll_into_pitch),
        "lever_x": "X_N - x_Ge",
        "arm_from_a": "[X_N - x_Ge, y_Ge, z_Ge]",
        "arm_from_g": "[x_Ge - X_N, -y_Ge, -z_Ge]",
        "nose_distance": "L_f - L_A",
        "roll_g": component(moment_g[0], (BODY_L + ARM_Z * BODY_Y - ARM_Y * BODY_Z, "l_A + z*Y_b - y*Z")),
        "pitch_g": component(moment_g[1], (BODY_M + ARM_X * BODY_Z - ARM_Z * BODY_X, "m_A + x*Z - z*X")),
        "yaw_g": component(moment_g[2], (BODY_N + ARM_Y * BODY_X - ARM_X * BODY_Y, "n_A + y*X - x*Y_b")),
        "pitch_g_engineering": "m_A + (X_N - x_Ge)*Z - z_Ge*X",
        "added": _added_signs(),
        "nose_up": _nose_direction(),
        "roll_coefficient": "C_lA + (z/b)*C_Yb - (y/b)*C_Z",
        "pitch_coefficient_g": "C_mA + (x/c_bar)*C_Z - (z/c_bar)*C_X",
        "yaw_coefficient_g": "C_nA + (y/b)*C_X - (x/b)*C_Yb",
    }


def _matrix(matrix: sp.Matrix, show) -> str:
    rows = []
    for row in range(3):
        cells = [_factor_order(show(matrix[row, column])) for column in range(3)]
        rows.append("[" + ", ".join(cells) + "]")
    return "[" + ", ".join(rows) + "]"


def _factor_order(text: str) -> str:
    """Put cos(α) before sin(β) so equivalent products share one written form."""
    return text.replace("-sin(β)*cos(α)", "-cos(α)*sin(β)").replace("sin(β)*cos(α)", "cos(α)*sin(β)")


def _matrix_vector(vector: sp.Matrix, show) -> str:
    return "[" + ", ".join(show(item) for item in vector) + "]"


def _added_signs() -> dict[tuple[str, str], str]:
    """Sign of one additional moment component for one positive force."""
    arm = sp.Matrix([1, 1, 1])
    names = {"X": sp.Matrix([1, 0, 0]), "Y_b": sp.Matrix([0, 1, 0]), "Z": sp.Matrix([0, 0, 1])}
    moments = ("滚转", "俯仰", "偏航")
    found = {}
    for force_name, force in names.items():
        added = -_cross(arm, force)
        for index, moment_name in enumerate(moments):
            found[(force_name, moment_name)] = _sign(added[index])
    return found


def _nose_direction() -> str:
    """Positive pitch raises the nose when Z points down. Upward force has Z<0."""
    added = -_cross(sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, -1]))
    return "抬头方向" if added[1] > 0 else "低头方向"


def _check_order(to_air: sp.Matrix, body_from_air: sp.Matrix, attack, sideslip) -> None:
    values = {attack: sp.Float("0.3"), sideslip: sp.Float("-0.2")}
    rotation = sp.Matrix(body_from_air.subs(values)).applyfunc(sp.N)
    airflow_force = sp.Matrix([-1.5, 0.4, -2.0])
    airflow_moment = sp.Matrix([0.2, -0.5, 0.7])
    arm_body = sp.Matrix([0.25, -0.1, 0.15])
    arm_air = sp.Matrix(to_air.subs(values)).applyfunc(sp.N) * arm_body
    body_force = rotation * airflow_force
    body_moment = rotation * airflow_moment
    after_rotation = body_moment - _cross(arm_body, body_force)
    before_rotation = rotation * (airflow_moment - _cross(arm_air, airflow_force))
    if max(abs(after_rotation[index] - before_rotation[index]) for index in range(3)) > sp.Float("1e-8"):
        raise DefinitionError("Rotating before the reference change disagrees with the reverse order.")
