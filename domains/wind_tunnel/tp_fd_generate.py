# -*- coding: utf-8 -*-
"""Generate cloze answers for balance elastic angles in the wind frame.

The questions follow TP_FD.md. Axis directions, installation sense, elastic
bends, and increment signs come from BodyFrameModel. Passive rotation is fixed.

The level balance frame is aligned with the wind frame by RP_TP_FD. The
installation roll is applied first, so

    [XFD, YFD, ZFD]^T = R1*R2*[XTP, YTP, ZYP]^T

with R2 the passive installation and R1 = RP_TP_FD. A positive elastic bend
moves the balance axial direction; the small-angle axial vector is then taken
through that product. Delta-alpha and delta-beta are positive when the resulting
nose motion increases the model's own angles.

Run::

    python -m domains.wind_tunnel.tp_fd_generate
    python -m domains.wind_tunnel.tp_fd_generate --spec declaration.json --out tp_fd_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from body_frame_model import (
    ALPHA,
    ALPHA_E,
    ALPHA_V,
    BALANCE_FRONT,
    BALANCE_VOCAB,
    BETA,
    BETA_E,
    BETA_V,
    DefinitionError,
    DELTA_ALPHA,
    DELTA_BETA,
    DELTA_GAMMA,
    FRONT,
    GAMMA,
    GAMMA_E,
    GAMMA_TP,
    NEGATIVE,
    NEGATIVE_ROTATION,
    OPPOSITE,
    PLACE_INVERTED,
    PLACE_UPRIGHT,
    POSITIVE,
    POSITIVE_ROTATION,
    ROTATION_ANGLE,
    SENSE_OPPOSITE,
    SENSE_SAME,
    UNKNOWN,
    WIND_VOCAB,
    BodyFrameModel,
    roll_about_nose_sense,
    _VECTOR,
    _body_direction_in_tunnel,
    _dot,
)


ROLE_ZH = {
    "forward": "前面",
    "rearward": "后面",
    "up": "上方",
    "down": "下方",
    "left": "左侧",
    "right": "右侧",
}
AXIS_ROLE_ZH = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
SIGN_ZH = {POSITIVE: "正", NEGATIVE: "负", UNKNOWN: "不确定"}
DIRECTED = [
    "XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP",
    "-XFD", "-YFD", "-ZFD", "-XTZ", "-YTZ", "-ZTZ", "-XTP", "-YTP", "-ZYP",
]
PLAIN_AXES = ["XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP"]
WIND_COMPONENTS = ["XFD", "YFD", "ZFD"]
FRAMES = ["机体系", "天平系", "风轴系"]
CONVERSION_PAIRS = [f"{source}向{target}" for source in FRAMES for target in FRAMES]
DIRECTIONS = ["上方", "下方", "左侧", "右侧", "前面", "后面"]
SIGNS = ["正", "负", "不确定"]
ROLES = ["轴向", "法向", "侧向"]
TURNS = ["正", "逆"]
CLOCKS = ["逆时针", "顺时针"]
MOUNTS = ["正装", "反装"]
VERTICAL = ["上方", "下方"]
SIDES = ["左侧", "右侧"]
HAND = ["正", "负"]
RELATIONS = ["相同", "相反"]

ALPHA_E_SYM, BETA_E_SYM, GAMMA_TP_SYM = sp.symbols("alpha_e beta_e gamma_tp", real=True)
DISPLAY = {
    "alpha_e": "α_e",
    "beta_e": "β_e",
    "gamma_e": "γ_e",
    "gamma_tp": "γ_tp",
}
ROLE_VECTOR = {"法向": _VECTOR["body_up"], "侧向": _VECTOR["body_right"], "轴向": _VECTOR["body_forward"]}

STANDARD_SPEC = {
    "alpha": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma": {"right_wing_toward": "body_down", "sign": "positive"},
    "alpha_m": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta_m": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma_m": {"right_wing_toward": "body_down", "sign": "positive"},
    "placement": "upright",
    "body_frame": {
        "chirality": "right",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    },
    "wind_frame": {
        "chirality": "right",
        "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
        "YFD": {"along": "tunnel_normal", "points": "tunnel_up"},
    },
    "alpha_e": {"axial_bends": "balance_up", "sign": "positive"},
    "beta_e": {"axial_bends": "balance_left", "sign": "positive"},
    "gamma_e": {"lateral_moves": "balance_down", "sign": "positive"},
    "alpha_v": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta_v": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma_tp": {"sense": "same"},
    "balance_frame": {
        "chirality": "right",
        "XTP": {"along": "balance_axial", "points": "balance_forward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    },
}


def _text(expr) -> str:
    rendered = str(sp.simplify(expr))
    for source, target in DISPLAY.items():
        rendered = rendered.replace(source, target)
    return rendered


def _matrix_text(matrix: sp.Matrix) -> str:
    rows = []
    for row in range(matrix.rows):
        cells = ", ".join(_text(matrix[row, column]) for column in range(matrix.cols))
        rows.append(f"[{cells}]")
    return "[" + ", ".join(rows) + "]"


def _vector_text(values) -> str:
    return "[" + ", ".join(_text(value) for value in values) + "]"


def _axial_hand_sign(right_wing_toward: str) -> str:
    return "正" if roll_about_nose_sense(right_wing_toward) == SENSE_SAME else "负"


def _balance_roll_relation(model: BodyFrameModel) -> str:
    """Whether positive elastic roll matches the right-hand rule about balance forward."""
    toward = model.elastic_angles[GAMMA_E].positive_toward
    sense = model._roll_sense(_VECTOR[BALANCE_FRONT], toward)
    if sense == POSITIVE_ROTATION:
        return "相同"
    if sense == NEGATIVE_ROTATION:
        return "相反"
    raise DefinitionError("Elastic roll has no sense about the balance axial positive direction.")


def _direction_zh(name: str) -> str:
    return ROLE_ZH[name.split("_", 1)[1]]


def _role(absolute_axis: str) -> str:
    return absolute_axis.split("_", 1)[1]


def _directed(frame: dict, target) -> str:
    for name, axis in frame.items():
        alignment = _dot(axis.vector, target)
        if alignment == 1:
            return name
        if alignment == -1:
            return f"-{name}"
    raise DefinitionError("No coordinate axis lies on the requested absolute direction.")


def _coords(vector, frame: dict, names: tuple[str, ...]) -> list[int]:
    return [_dot(vector, frame[name].vector) for name in names]


def _sense_number(rotation: str) -> int:
    if rotation == POSITIVE_ROTATION:
        return 1
    if rotation == NEGATIVE_ROTATION:
        return -1
    raise DefinitionError("The installation or elastic roll has no right-hand sense.")


def _small_sign(expr, positives: tuple) -> str:
    expanded = sp.expand(expr)
    constant = expanded
    coefficients = []
    for symbol in positives:
        coefficients.append(sp.simplify(expanded.coeff(symbol)))
        constant = sp.simplify(constant.subs(symbol, 0))
    if constant != 0:
        if constant > 0:
            return "正"
        if constant < 0:
            return "负"
        return "不确定"
    usable = []
    for coefficient in coefficients:
        if coefficient == 0:
            continue
        if coefficient > 0:
            usable.append(1)
        elif coefficient < 0:
            usable.append(-1)
        else:
            return "不确定"
    if len(set(usable)) != 1:
        return "不确定"
    return "正" if usable[0] > 0 else "负"


def _projection(components, frame: dict, names: tuple[str, ...], target) -> sp.Expr:
    total = 0
    for name, value in zip(names, components):
        total += value * _dot(frame[name].vector, target)
    return sp.simplify(total)


def generate(spec: dict) -> dict:
    """Build the cloze key for one declaration."""
    model = BodyFrameModel().declare(spec)
    if model.placement not in (PLACE_UPRIGHT, PLACE_INVERTED):
        raise DefinitionError("placement must be upright or inverted.")
    for frame, names in (
        (model.wind_frame, WIND_VOCAB.coordinate_axes),
        (model.balance_frame, BALANCE_VOCAB.coordinate_axes),
        (model.body_frame, ("XTZ", "YTZ", "ZTZ")),
    ):
        if any(not frame[name].points_to for name in names):
            raise DefinitionError("Wind, balance, and body frames must be complete.")
    install = model.installation
    if install.rotation_when_positive not in (POSITIVE_ROTATION, NEGATIVE_ROTATION):
        raise DefinitionError("gamma_tp must declare its sense.")
    elastic = [model.elastic_angles[name] for name in (ALPHA_E, BETA_E, GAMMA_E)]
    if any(not record.positive_toward or not record.associated_axis for record in elastic):
        raise DefinitionError("alpha_e, beta_e, and gamma_e must be declared.")
    alpha_e, beta_e, gamma_e = elastic
    if _sense_number(gamma_e.rotation_when_positive) == 0:
        raise DefinitionError("gamma_e has no torsion sense.")

    balance_names = BALANCE_VOCAB.coordinate_axes
    wind_names = WIND_VOCAB.coordinate_axes
    alignment = sp.Matrix([
        [
            _dot(model.wind_frame[wind_name].vector, model.balance_frame[balance_name].vector)
            for balance_name in balance_names
        ]
        for wind_name in wind_names
    ])
    axial = model.balance_frame[install.associated_axis]
    coordinate_sign = _sense_number(axial.installation_rotation_when_positive)
    theta = coordinate_sign * GAMMA_TP_SYM
    roll = sp.simplify(axial.rotation_matrix.subs(ROTATION_ANGLE, theta))

    base = _coords(_VECTOR[BALANCE_FRONT], model.balance_frame, balance_names)
    alpha_shift = _coords(_VECTOR[alpha_e.positive_toward], model.balance_frame, balance_names)
    beta_shift = _coords(_VECTOR[beta_e.positive_toward], model.balance_frame, balance_names)
    deformed = [
        base[index] + alpha_shift[index] * ALPHA_E_SYM + beta_shift[index] * BETA_E_SYM
        for index in range(3)
    ]
    wind_vector = sp.simplify(alignment * roll * sp.Matrix(deformed))
    level_wind = [component.subs(GAMMA_TP_SYM, 0) for component in wind_vector]

    def signed_angle(theory_name: str, airspeed: bool) -> sp.Expr:
        if airspeed:
            toward = model.angles[theory_name].positive_airspeed_toward
        else:
            toward = model.angles[theory_name].right_wing_toward
        if not toward:
            raise DefinitionError(f"{theory_name} must declare its positive direction.")
        nose = _body_direction_in_tunnel(OPPOSITE[toward], model.placement)
        forward = _projection(wind_vector, model.wind_frame, wind_names, _VECTOR[FRONT])
        if forward == 0:
            raise DefinitionError("The deformed axial vector has no tunnel-forward component.")
        return sp.simplify(
            _projection(wind_vector, model.wind_frame, wind_names, nose) / forward
        )

    def component_formula(theory_name: str) -> sp.Expr:
        numerator, denominator = _wind_parts(theory_name)
        return sp.simplify(numerator / denominator)

    def _wind_parts(theory_name: str) -> tuple[sp.Expr, sp.Expr]:
        toward = model.angles[theory_name].positive_airspeed_toward
        nose = _body_direction_in_tunnel(OPPOSITE[toward], model.placement)
        symbols = [sp.Symbol(name) for name in wind_names]
        forward = _projection(symbols, model.wind_frame, wind_names, _VECTOR[FRONT])
        return _projection(symbols, model.wind_frame, wind_names, nose), forward

    value_signs = {
        name: _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM))
        for name, value in zip(WIND_COMPONENTS, level_wind)
    }

    def quotient_parts(theory_name: str) -> dict:
        delta_name = DELTA_ALPHA if theory_name == ALPHA else DELTA_BETA
        outer = SIGN_ZH.get(model.increments[delta_name].increment_sign, "")
        return _positive_quotient(*_wind_parts(theory_name), value_signs, outer)

    delta_alpha = signed_angle(ALPHA, True)
    delta_beta = signed_angle(BETA, True)

    def flow_sign(flow_name: str, theory_name: str) -> str:
        flow = model.flow_angles[flow_name].positive_airspeed_toward
        theory = model.angles[theory_name].positive_airspeed_toward
        if not flow or not theory:
            raise DefinitionError(f"{flow_name} and {theory_name} must both be declared.")
        return "正" if flow == theory else "负"

    items: list[dict] = []

    def add(step: int, label: str, prompt: str, answer, options=None) -> None:
        items.append({
            "id": len(items) + 1,
            "manuscript_id": str(len(items) + 1),
            "step": step,
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer if isinstance(answer, str) else _text(answer),
        })

    add(1, "C2", "弹性角增量的坐标系转换方向", "天平系向风轴系", CONVERSION_PAIRS)
    add(1, "C1", "风洞绝对轴向正方向（前方）", _directed(model.wind_frame, ROLE_VECTOR["轴向"]), DIRECTED)
    add(1, "C1", "天平绝对轴向正方向（前方）", _directed(model.balance_frame, ROLE_VECTOR["轴向"]), DIRECTED)
    for role_name in ("法向", "侧向", "轴向"):
        add(1, "C1", f"风轴系绝对{role_name}正方向对应的有向坐标轴", _directed(model.wind_frame, ROLE_VECTOR[role_name]), DIRECTED)
    for role_name in ("法向", "侧向", "轴向"):
        add(
            1, "C1", f"天平系绝对{role_name}正方向对应的有向坐标轴",
            _directed(model.balance_frame, ROLE_VECTOR[role_name]), DIRECTED,
        )
    add(1, "A2", "平置后天平坐标到风轴坐标的矩阵 RP_TP_FD", _matrix_text(alignment))
    add(1, "C2", "RP_TP_FD 所对齐的坐标系", "天平系", FRAMES)
    add(1, "C1", "与天平系坐标轴方向相同的坐标系", "风轴系", FRAMES)

    add(2, "A1", "正安装角绕天平的绝对方向轴", AXIS_ROLE_ZH[_role(axial.absolute_axis)], ROLES)
    add(2, "C1", "该旋转对应的坐标轴", axial.name, PLAIN_AXES)
    add(2, "A1", "相对这根坐标轴正方向的转向", "正" if coordinate_sign == 1 else "逆", TURNS)
    add(2, "A2", "旋转矩阵 R_γ_tp", _matrix_text(roll))

    factor_options = ["RP_TP_FD", "R_γ_tp"]
    add(3, "A2", "连乘式中的 R1", "RP_TP_FD", factor_options)
    add(3, "A2", "连乘式中的 R2", "R_γ_tp", factor_options)

    add(4, "A1", "未变形时沿天平绝对轴向正方向（前方）的单位向量", _vector_text(base))
    add(4, "C1", "正弹性迎角使天平轴向弯向", _direction_zh(alpha_e.positive_toward), DIRECTIONS)
    add(4, "C1", "正弹性侧滑角使天平轴向弯向", _direction_zh(beta_e.positive_toward), DIRECTIONS)
    add(
        4, "A1", "沿天平绝对轴向正方向（前方）看去，正弹性滚转的扭转方向",
        "顺时针" if _balance_roll_relation(model) == "相同" else "逆时针",
        CLOCKS,
    )
    for record, angle_name in ((alpha_e, "弹性迎角"), (beta_e, "弹性侧滑角"), (gamma_e, "弹性滚转角")):
        add(4, "C1", f"{angle_name}的旋转轴", record.associated_axis, PLAIN_AXES)
    for name, value in zip(("XTP_e", "YTP_e", "ZTP_e"), deformed):
        add(4, "A1", f"正弹性角下 {name} 的符号", _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM)), SIGNS)
    add(4, "A1", "小角度近似后的天平轴向单位向量", _vector_text(deformed))

    for name, value in zip(WIND_COMPONENTS, wind_vector):
        add(5, "A2", f"变形后天平轴向在风轴系的 {name}", _text(value))

    add(6, "A1", "正理论迎角的空速方向确定后，Δα_e 应使用的风轴分量", _component_name(component_formula(ALPHA)), WIND_COMPONENTS)
    add(6, "A1", "正理论侧滑的空速方向确定后，Δβ_e 应使用的风轴分量", _component_name(component_formula(BETA)), WIND_COMPONENTS)

    add(7, "C1", "模型安装方式", "正装" if model.placement == PLACE_UPRIGHT else "反装", MOUNTS)
    add(7, "A1", "正弹性迎角使空速位于模型的", _direction_zh(model.increments[DELTA_ALPHA].effect_toward), VERTICAL)
    add(7, "A1", "正弹性侧滑角使空速位于模型的", _direction_zh(model.increments[DELTA_BETA].effect_toward), SIDES)
    add(7, "A1", "正弹性滚转绕天平绝对轴向正方向（前方），与右手定则的关系", _balance_roll_relation(model), RELATIONS)
    add(7, "C1", "正理论迎角时，空速指向机体的", _direction_zh(model.angles[ALPHA].positive_airspeed_toward), DIRECTIONS)
    add(7, "C1", "正理论侧滑时，空速指向机体的", _direction_zh(model.angles[BETA].positive_airspeed_toward), DIRECTIONS)
    add(7, "C1", "理论滚转角绕机体绝对轴向正方向（前方）的右手定则", _axial_hand_sign(model.angles[GAMMA].right_wing_toward), HAND)
    for delta_name, prompt in (
        (DELTA_ALPHA, "天平弹性角产生的 Δα_e"),
        (DELTA_BETA, "天平弹性角产生的 Δβ_e"),
        (DELTA_GAMMA, "天平弹性角产生的 Δγ_e"),
    ):
        sign = model.increments[delta_name].increment_sign
        add(7, "A1", prompt, SIGN_ZH.get(sign, "不确定"), SIGNS)
    for name, value in zip(WIND_COMPONENTS, level_wind):
        add(7, "A1", f"γ_tp = 0 且弹性角为正时 {name} 的符号", _small_sign(value, (ALPHA_E_SYM, BETA_E_SYM)), SIGNS)

    inside = "要求分子和分母都必须为正，角度正负只能写在分式外面"
    for symbol, theory_name in (("Δα_e", ALPHA), ("Δβ_e", BETA)):
        parts = quotient_parts(theory_name)
        add(8, "A1", f"写在分式外面的 {symbol} 符号。{inside}", parts["outer"], SIGNS)
        add(8, "A1", f"根据所选风轴分量的符号，{symbol} 分子的符号。{inside}", parts["num_sign"], SIGNS)
        add(8, "A1", f"{symbol} 的分子应使用的风轴分量。{inside}", parts["num_axis"], WIND_COMPONENTS)
        add(8, "A1", f"根据所选风轴分量的符号，{symbol} 分母的符号。{inside}", parts["den_sign"], SIGNS)
        add(8, "A1", f"{symbol} 的分母应使用的风轴分量。{inside}", parts["den_axis"], WIND_COMPONENTS)
    for symbol, theory_name in (("Δα_e", ALPHA), ("Δβ_e", BETA)):
        parts = quotient_parts(theory_name)
        prompt = f"代入后的 {symbol}。分子和分母都为正，负号写在整个分式之前"
        if symbol == "Δα_e":
            prompt += "。这个空只能填风轴分量 XFD、YFD、ZFD 的分式，不能直接写含 α_e、β_e 或 γ_e 的弹性角式"
        add(8, "A2", prompt, parts["formula"])
    add(8, "A2", "用弹性角表示的 Δα_e", _text(delta_alpha))
    add(8, "A2", "用弹性角表示的 Δβ_e", _text(delta_beta))
    roll_sign = model.increments[DELTA_GAMMA].increment_sign
    if roll_sign == POSITIVE:
        delta_gamma = "γ_e"
    elif roll_sign == NEGATIVE:
        delta_gamma = "-γ_e"
    else:
        raise DefinitionError("Δγ_e has no sign.")
    add(8, "A1", "弹性滚转产生的 Δγ_e。不随安装角变化，符号由弹性滚转与理论滚转绕绝对轴向正方向（前方）是否同向决定", delta_gamma)

    add(9, "A1", "气流偏心使实际迎角增大时 Δα_V 的符号", flow_sign(ALPHA_V, ALPHA), SIGNS)
    add(9, "A1", "气流偏心使实际侧滑角增大时 Δβ_V 的符号", flow_sign(BETA_V, BETA), SIGNS)

    return {
        "task": "balance_elastic_angle_to_wind_frame",
        "declaration": spec,
        "fixed": {
            "transform": "passive",
            "product": "[XFD, YFD, ZFD]^T = R1*R2*[XTP, YTP, ZYP]^T",
            "small_angle": "sin(theta) = theta, cos(theta) = 1",
        },
        "items": items,
    }


def _single_axis_term(expr) -> tuple[int, str]:
    expanded = sp.expand(expr)
    found = None
    for name in WIND_COMPONENTS:
        coefficient = sp.simplify(expanded.coeff(sp.Symbol(name)))
        if coefficient == 0:
            continue
        if found is not None or coefficient not in (1, -1):
            raise DefinitionError(f"A wind-angle part must be one signed axis, got {expr}.")
        found = (int(coefficient), name)
    remainder = sp.simplify(expanded.subs({sp.Symbol(name): 0 for name in WIND_COMPONENTS}))
    if found is None or remainder != 0:
        raise DefinitionError(f"A wind-angle part must be one signed axis, got {expr}.")
    return found


def _sign_number(sign: str) -> int:
    if sign == "正":
        return 1
    if sign == "负":
        return -1
    raise DefinitionError(f"A wind component sign must be 正 or 负, got {sign}.")


def _signed_factor(sign: str, axis: str) -> str:
    if sign == "正":
        return f"({axis})"
    return f"(-{axis})"


def _positive_quotient(numerator, denominator, value_signs: dict, outer: str) -> dict:
    """Keep each written part positive when the component itself may be negative.

    A negative component is written with a minus inside the numerator or denominator.
    The angle sign stays outside and follows the sign of the angle, not that minus.
    """
    num_coefficient, num_axis = _single_axis_term(numerator)
    den_coefficient, den_axis = _single_axis_term(denominator)
    num_sign = value_signs[num_axis]
    den_sign = value_signs[den_axis]
    if outer not in ("正", "负"):
        raise DefinitionError(f"The wind-angle sign must be 正 or 负, got {outer}.")
    geometric = num_coefficient * _sign_number(num_sign) * den_coefficient * _sign_number(den_sign)
    if geometric != _sign_number(outer):
        raise DefinitionError("The outside angle sign does not match the positive wind-component quotient.")
    formula = f"{_signed_factor(num_sign, num_axis)}/{_signed_factor(den_sign, den_axis)}"
    if outer == "负":
        formula = f"-{formula}"
    return {
        "outer": outer,
        "num_sign": num_sign,
        "num_axis": num_axis,
        "den_sign": den_sign,
        "den_axis": den_axis,
        "formula": formula,
    }


def _component_name(formula) -> str:
    """The wind component in the numerator of the small-angle formula."""
    numerator, _denominator = sp.fraction(sp.together(formula))
    measuring = [name for name in WIND_COMPONENTS if sp.Symbol(name) in numerator.free_symbols]
    if len(measuring) != 1:
        raise DefinitionError(f"Cannot tell which wind component represents {formula}.")
    return measuring[0]


def _render(payload: dict) -> str:
    lines = ["# 天平系到风轴系评测答案", ""]
    step = None
    for item in payload["items"]:
        if item["step"] != step:
            step = item["step"]
            lines.append(f"## 步骤 {step}")
            lines.append("")
        options = ""
        if item["options"]:
            options = "（" + "/".join(item["options"]) + "）"
        lines.append(
            f"- ID {item['manuscript_id']} [{item['label']}] {item['prompt']}{options}：{item['answer']}"
        )
    lines.append("")
    return "\n".join(lines)


def _check_standard() -> None:
    payload = generate(STANDARD_SPEC)
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    assert len(payload["items"]) == 64
    assert by_id["1"] == "天平系向风轴系"
    assert by_id["2"] == "XFD"
    assert by_id["3"] == "XTP"
    assert [by_id[str(index)] for index in range(4, 10)] == ["YFD", "ZFD", "XFD", "YTP", "ZYP", "XTP"]
    assert by_id["10"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert by_id["11"] == "天平系"
    assert by_id["12"] == "风轴系"
    assert by_id["13"] == "轴向"
    assert by_id["14"] == "XTP"
    assert by_id["15"] == "正"
    assert [by_id["17"], by_id["18"]] == ["RP_TP_FD", "R_γ_tp"]
    assert by_id["19"] == "[1, 0, 0]"
    assert by_id["20"] == "上方"
    assert by_id["21"] == "左侧"
    assert by_id["22"] == "顺时针"
    assert by_id["23"] == "ZYP"
    assert by_id["24"] == "YTP"
    assert by_id["25"] == "XTP"
    assert by_id["26"] == "正"
    assert by_id["27"] == "正"
    assert by_id["28"] == "负"
    assert by_id["29"] == "[1, α_e, -β_e]"
    assert by_id["30"] == "1"
    assert by_id["33"] == "YFD"
    assert by_id["34"] == "ZFD"
    assert by_id["35"] == "正装"
    assert by_id["36"] == "下方"
    assert by_id["37"] == "右侧"
    assert by_id["38"] == "相同"
    assert by_id["39"] == "下方"
    assert by_id["40"] == "右侧"
    assert by_id["41"] == "正"
    assert by_id["42"] == "正"
    assert by_id["43"] == "正"
    assert by_id["44"] == "正"
    assert by_id["45"] == "正"
    assert by_id["46"] == "正"
    assert by_id["47"] == "负"
    assert [by_id[str(index)] for index in range(48, 53)] == ["正", "正", "YFD", "正", "XFD"]
    assert [by_id[str(index)] for index in range(53, 58)] == ["正", "负", "ZFD", "正", "XFD"]
    assert by_id["58"] == "(YFD)/(XFD)"
    assert by_id["59"] == "(-ZFD)/(XFD)"
    assert by_id["60"] == "α_e*cos(γ_tp) - β_e*sin(γ_tp)"
    assert by_id["61"] == "α_e*sin(γ_tp) + β_e*cos(γ_tp)"
    assert by_id["62"] == "γ_e"
    assert by_id["63"] == "正"
    assert by_id["64"] == "正"

    inverted = json.loads(json.dumps(STANDARD_SPEC))
    inverted["placement"] = PLACE_INVERTED
    flipped = {item["manuscript_id"]: item["answer"] for item in generate(inverted)["items"]}
    assert flipped["10"] == by_id["10"]
    assert flipped["35"] == "反装"
    assert flipped["36"] == "上方"
    assert flipped["37"] == "左侧"
    assert flipped["38"] == "相同"
    assert flipped["41"] == "正"
    assert flipped["42"] == "负"
    assert flipped["43"] == "负"
    assert flipped["44"] == "正"
    assert flipped["48"] == "负"
    assert flipped["50"] == "YFD"
    assert flipped["58"] == "-(YFD)/(XFD)"
    assert flipped["60"] == "-α_e*cos(γ_tp) + β_e*sin(γ_tp)"
    assert flipped["62"] == "γ_e"

    opposite = json.loads(json.dumps(STANDARD_SPEC))
    opposite["gamma_tp"] = {"sense": SENSE_OPPOSITE}
    turned = {item["manuscript_id"]: item["answer"] for item in generate(opposite)["items"]}
    assert turned["15"] == "逆"
    assert turned["60"] == "α_e*cos(γ_tp) + β_e*sin(γ_tp)"

    opposed = json.loads(json.dumps(STANDARD_SPEC))
    opposed["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    opposed_flow = {item["manuscript_id"]: item["answer"] for item in generate(opposed)["items"]}
    assert opposed_flow["63"] == "负"

    rearward_balance = json.loads(json.dumps(STANDARD_SPEC))
    rearward_balance["balance_frame"] = {
        "chirality": "left",
        "XTP": {"along": "balance_axial", "points": "balance_rearward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    }
    rearward_balance_answers = {
        item["manuscript_id"]: item["answer"] for item in generate(rearward_balance)["items"]
    }
    assert rearward_balance_answers["3"] == "-XTP"
    assert rearward_balance_answers["9"] == "-XTP"
    assert rearward_balance_answers["15"] == "逆"
    assert rearward_balance_answers["22"] == "顺时针"
    assert rearward_balance_answers["38"] == "相同"
    assert rearward_balance_answers["44"] == "正"
    assert rearward_balance_answers["62"] == "γ_e"

    roll_up = json.loads(json.dumps(STANDARD_SPEC))
    roll_up["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    roll_up["gamma"] = {"right_wing_toward": "body_up", "sign": "positive"}
    roll_up_answers = {item["manuscript_id"]: item["answer"] for item in generate(roll_up)["items"]}
    assert roll_up_answers["38"] == "相反"
    assert roll_up_answers["41"] == "负"
    assert roll_up_answers["62"] == "γ_e"

    elastic_up = json.loads(json.dumps(STANDARD_SPEC))
    elastic_up["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    elastic_up_answers = {item["manuscript_id"]: item["answer"] for item in generate(elastic_up)["items"]}
    assert elastic_up_answers["44"] == "负"
    assert elastic_up_answers["62"] == "-γ_e"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the balance-to-wind elastic-angle cloze answer key.")
    parser.add_argument("--spec", type=Path, help="JSON declaration. Defaults to the standard upright frame.")
    parser.add_argument("--out", type=Path, default=Path("tp_fd_eval.json"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else STANDARD_SPEC
    if args.spec is None:
        _check_standard()
    payload = generate(spec)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = args.out.with_suffix(".md")
    markdown.write_text(_render(payload), encoding="utf-8")
    counts = {}
    for item in payload["items"]:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    print(f"Wrote {len(payload['items'])} blanks to {args.out}")
    print("Labels: " + ", ".join(f"{label} {counts[label]}" for label in ("C1", "C2", "A1", "A2")))


if __name__ == "__main__":
    main()
