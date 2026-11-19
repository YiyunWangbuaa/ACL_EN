# -*- coding: utf-8 -*-
"""Flight-mechanics spatial-transform cloze.

The manuscript is the standard coordinate-conversion flow: ground to body,
ground to the unbanked track frame, airflow roll about the track, body through
stability to airflow, then the airspeed closure. Elementary matrices are the passive right-handed matrices
already used by FlightFrameModel. A parent-from-child attitude is the transpose
of the child-from-parent product, so the factors below are computed, not copied
from a hand formula.

Run from the repository root::

    python -m domains.aerodynamics.flow_generate
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

import sympy as sp

from cloze_score import answer_letter, format_choices
from domain_layout import DOMAINS
from domains.aerodynamics.flow_loads import build_loads
from flight_frame_model import (
    ALPHA,
    BETA,
    GAMMA,
    MU,
    NEGATIVE,
    NEGATIVE_ROTATION,
    PHI,
    POSITIVE,
    POSITIVE_ROTATION,
    PSI,
    STANDARD_SPEC,
    THETA,
    VARPHI,
    DefinitionError,
    FlightFrameModel,
)


BLANK_COUNT = 131
OUT_DIR = DOMAINS["aerodynamics"]["flow"]["review"]
EYE = sp.eye(3)
E1 = sp.Matrix([1, 0, 0])

YAW, PITCH, ROLL, AZIMUTH, INCLINE, BANK, ATTACK, SIDESLIP = sp.symbols(
    "yaw pitch roll azimuth incline bank attack sideslip", real=True
)
VX, VY, VZ = sp.symbols("vx vy vz", real=True)
SYMBOL = {
    PSI: YAW,
    THETA: PITCH,
    PHI: ROLL,
    VARPHI: AZIMUTH,
    MU: INCLINE,
    GAMMA: BANK,
    ALPHA: ATTACK,
    BETA: SIDESLIP,
}
NAME = {
    YAW: "ψ",
    PITCH: "θ",
    ROLL: "Φ",
    AZIMUTH: "φ",
    INCLINE: "μ",
    BANK: "γ",
    ATTACK: "α",
    SIDESLIP: "β",
}
SLOT_LINE = re.compile(r"^(\d+)\. .+\{ID:(\d+), .+\}。$")


def _passive(slot: str, angle) -> sp.Matrix:
    c = sp.cos(angle)
    s = sp.sin(angle)
    if slot == "X":
        return sp.Matrix([[1, 0, 0], [0, c, s], [0, -s, c]])
    if slot == "Y":
        return sp.Matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    if slot == "Z":
        return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    raise DefinitionError(slot)


def _binding(model: FlightFrameModel, origin: str, destination: str, angle: str):
    found = [item for item in model.conversion(origin, destination) if item.angle == angle]
    if len(found) != 1 or not found[0].axis:
        raise DefinitionError(f"{angle} is not bound from {origin} to {destination}.")
    return found[0]


def _argument(model: FlightFrameModel, origin: str, destination: str, angle: str):
    binding = _binding(model, origin, destination, angle)
    symbol = SYMBOL[angle]
    sense = binding.rotation_when_positive
    if sense == POSITIVE_ROTATION:
        return binding, symbol
    if sense == NEGATIVE_ROTATION:
        return binding, -symbol
    raise DefinitionError(f"{angle} has no rotation sense.")


def _product(model: FlightFrameModel, steps: tuple[tuple[str, str, str], ...]) -> tuple[sp.Matrix, str]:
    """Parent-from-child matrix and its factor text.

    ``steps`` run from the first intrinsic rotation to the last. The
    child-from-parent product therefore has the first factor on the right.
    """
    factors = []
    for origin, destination, angle in steps:
        binding, argument = _argument(model, origin, destination, angle)
        slot = binding.axis[0]
        shown = sp.simplify(-argument)
        factors.append(f"C{slot.lower()}({_argument_text(shown, SYMBOL[angle])})")
    text = "*".join(factors)
    matrix = EYE
    for part in text.split("*"):
        matrix *= _parse_factor(part)
    return sp.simplify(matrix), text


def _argument_text(expr, symbol) -> str:
    expr = sp.simplify(expr)
    if expr == symbol:
        return NAME[symbol]
    if expr == -symbol:
        return f"-{NAME[symbol]}"
    raise DefinitionError(f"Cannot show rotation argument {expr}.")


def _parse_factor(text: str) -> sp.Matrix:
    matched = re.fullmatch(r"C([xyz])\((.+)\)", text)
    if not matched:
        raise DefinitionError(f"Bad factor {text}.")
    slot = matched.group(1).upper()
    body = matched.group(2)
    sign = -1 if body.startswith("-") else 1
    name = body[1:] if body.startswith("-") else body
    symbol = next(key for key, value in NAME.items() if value == name)
    return _passive(slot, sign * symbol)


def _flip_factor(text: str) -> str:
    matched = re.fullmatch(r"C([xyz])\((.+)\)", text)
    if not matched:
        raise DefinitionError(f"Bad factor {text}.")
    body = matched.group(2)
    flipped = body[1:] if body.startswith("-") else f"-{body}"
    return f"C{matched.group(1)}({flipped})"


def _transpose_product(text: str) -> str:
    """Factor string of the transpose. Passive C(η)^T equals C(-η)."""
    return "*".join(_flip_factor(part) for part in reversed(text.split("*")))


def _track_wrongs(correct: str, bank_factor: str) -> list[str]:
    parts = correct.split("*")
    candidates = [
        "*".join(_flip_factor(part) for part in parts),
        "*".join(reversed(parts)),
        "*".join(_flip_factor(part) if index == len(parts) - 1 else part for index, part in enumerate(parts)),
        f"{correct}*{bank_factor}",
    ]
    wrongs = []
    for item in candidates:
        if item != correct and item not in wrongs:
            wrongs.append(item)
    if len(wrongs) < 2:
        raise DefinitionError(f"Not enough distinct distractors for {correct}.")
    return wrongs


def _show(expr) -> str:
    text = str(sp.simplify(expr))
    text = text.replace("**", "^")
    for symbol, name in NAME.items():
        text = text.replace(str(symbol), name)
    return text


def _vector(matrix_times_column) -> str:
    cells = ", ".join(_show(item) for item in matrix_times_column)
    return f"[{cells}]"


def _at_degrees(expr, values: dict) -> str:
    symbols = {key: sp.Symbol(f"deg{amount}") for key, amount in values.items()}
    shown = _show(sp.simplify(expr.subs(symbols)))
    for amount in values.values():
        shown = shown.replace(f"deg{amount}", f"{amount}°")
    return shown


def _component_word(expr, values: dict) -> str:
    number = sp.N(expr.subs({key: sp.rad(amount) for key, amount in values.items()}))
    if abs(number) < sp.Float("1e-10"):
        raise DefinitionError("The vertical component is zero, so its direction is undefined.")
    return "下方" if number > 0 else "上方"


def _recovers(component_expr, target, candidates: list) -> sp.Expr:
    for candidate in candidates:
        good = True
        for amount in (-0.6, -0.2, 0.15, 0.55):
            other = 0.25 if target == SIDESLIP else -0.2
            values = {ATTACK: other, SIDESLIP: amount} if target == SIDESLIP else {ATTACK: amount, SIDESLIP: other}
            predicted = candidate.subs({
                VX: component_expr[0].subs(values),
                VY: component_expr[1].subs(values),
                VZ: component_expr[2].subs(values),
            })
            if abs(sp.N(predicted - values[target])) > sp.Float("1e-8"):
                good = False
                break
        if good:
            return candidate
    raise DefinitionError(f"No recovery formula matches {target}.")


def _formula(expr) -> str:
    text = str(expr).replace("**", "^").replace(" ", "")
    return text.replace(",", ", ")


def _sign_word(model: FlightFrameModel, angle: str, words: dict[str, str]) -> str:
    toward = model.binding(angle).positive_toward
    if toward not in words:
        raise DefinitionError(f"{angle} points {toward}, which this question does not name.")
    return words[toward]


def _choose(correct: str, wrongs: list[str], salt: int) -> tuple[list[str], str]:
    options = []
    for item in (correct, *wrongs):
        if item not in options:
            options.append(item)
    if len(options) < 2 or correct not in options:
        raise DefinitionError(f"Choices for {correct} are not usable.")
    shift = salt % len(options)
    return options[shift:] + options[:shift], correct


def _choice(number: int, label: str, stem: str, correct: str, wrongs: list[str]) -> dict:
    options, answer = _choose(correct, wrongs, number)
    return {"id": number, "label": label, "stem": stem, "options": options, "answer": answer, "format": "无"}


def _open(number: int, label: str, stem: str, answer: str, fmt: str = "向量") -> dict:
    return {"id": number, "label": label, "stem": stem, "options": None, "answer": answer, "format": fmt}


def _load_questions(loads: dict) -> list[dict]:
    added = loads["added"]
    rows = [
        _choice(60, "C2", "本子过程的已知气动力分量位于气流系，输出要求采用机体系，因此所需转换方向是", "气流系向机体系", ["机体系向气流系", "气流系向稳定性系后停止"]),
        _choice(61, "C2", "本子过程转换前后，力矩参考点", "始终为A", ["由A变为G", "由G变为A"]),
        _choice(62, "C1", "机体系绝对轴向正方向对应", "X_b", ["Y_b", "Z_b"]),
        _choice(63, "C1", "机体系绝对侧向正方向对应", "Y_b", ["X_b", "Z_b"]),
        _choice(64, "C1", "机体系绝对法向正方向对应", "Z_b", ["X_b", "Y_b"]),
        _choice(65, "C1", "气流系中的阻力方向为", "-X_a", ["+X_a", "-Z_a"]),
        _choice(66, "C1", "气流系中的正升力方向为", "-Z_a", ["+Z_a", "+Y_a"]),
        _choice(67, "C1", "气流系中的正侧力方向为", "+Y_a", ["-Y_a", "+Z_a"]),
        _open(68, "A2", "按照 X_a、Y_a、Z_a 的顺序，气流系气动力向量为", loads["air_force"]),
        _choice(69, "C1", "气流系滚转力矩 l_a 的正方向绕", "+X_a", ["+Y_a", "+Z_a"]),
        _choice(70, "C1", "气流系俯仰力矩 m_a 的正方向绕", "+Y_a", ["+X_a", "+Z_a"]),
        _open(71, "A2", "按照 X_a、Y_a、Z_a 的顺序，气流系力矩向量为", loads["air_moment"]),
        _choice(72, "A1", "仅有正迎角时，稳定性系前方轴相对机体系前方轴偏向机体", loads["stable_side"], ["上方", "下方", "右方"]),
        _choice(73, "A1", "因此该前方轴在机体系中的 Z_b 分量为", loads["stable_z"], ["正", "负", "零"]),
        _open(74, "A2", "由此构造机体系到稳定性系的被动变换矩阵 T_s←b 为", loads["to_stable"], "矩阵"),
        _choice(75, "A1", "仅有正侧滑角时，气流系前方轴相对稳定性系前方轴偏向稳定性系", loads["air_side"], ["左方", "右方", "下方"]),
        _choice(76, "A1", "因此该前方轴在稳定性系中的 Y_s 分量为", loads["air_y"], ["正", "负", "零"]),
        _open(77, "A2", "由此构造稳定性系到气流系的被动变换矩阵 T_a←s 为", loads["stable_to_air"], "矩阵"),
        _choice(78, "C2", "机体系到气流系的整体变换 T_a←b 应写为", "T_a←s*T_s←b", ["T_s←b*T_a←s"]),
        _open(79, "A2", "T_a←s的展开式为", loads["stable_to_air"], "矩阵"),
        _open(80, "A2", "T_s←b的展开式为", loads["to_stable"], "矩阵"),
        _open(81, "A2", "展开该乘积得到 T_a←b 为", loads["to_air"], "矩阵"),
        _choice(82, "C2", "将气流系气动力换成机体系气动力，应使用", "T_a←b 的转置", ["T_a←b"]),
        _open(83, "A2", "记该反向矩阵为 B，展开 B 得到", loads["body_from_air"], "矩阵"),
        _open(84, "A2", "机体系轴向气动力 X 为", loads["body_x"], "表达式"),
        _open(85, "A2", "机体系侧向气动力 Y_b 为", loads["body_y"], "表达式"),
        _open(86, "A2", "机体系法向气动力 Z 为", loads["body_z"], "表达式"),
        _choice(87, "A1", "升力产生的机体系轴向分量为", loads["lift_x"], ["正", "负", "零"]),
        _choice(88, "A1", "升力产生的机体系法向分量为", loads["lift_z"], ["正", "负", "零"]),
        _choice(89, "A1", "当 β=0、Y=0 且 D>0、L>0 时，仅凭这些条件，机体系轴向合力 X 的符号", loads["axial_sign"], ["一定为正", "一定为负", "不能确定"]),
        _open(90, "A2", "取 α=30°、β=0°、D=2、Y=0、L=2*sqrt(3)，机体系气动力向量为", loads["example_force"]),
        _choice(91, "A1", "坐标转换后，气动力的欧氏长度与转换前", "相同", ["一般增大", "一般减小"]),
        _choice(92, "C2", "同一参考点 A 处的气动力矩从气流系变到机体系时，应使用", "同一个矩阵 B", ["B 的负矩阵", "额外加入力臂平移"]),
        _open(93, "A2", "机体系滚转力矩 l_A 为", loads["body_l"], "表达式"),
        _open(94, "A2", "机体系俯仰力矩 m_A 为", loads["body_m"], "表达式"),
        _open(95, "A2", "机体系偏航力矩 n_A 为", loads["body_n"], "表达式"),
        _choice(96, "A1", "当 β=0 时，气流系正俯仰力矩的旋转轴与机体系侧向轴", loads["pitch_relation"], ["重合", "相互垂直", "方向无法确定"]),
        _choice(97, "A1", "当 α=0、0°<β<90°、l_a>0 且 m_a=n_a=0 时，机体系俯仰力矩 m_A 为", loads["roll_to_pitch"], ["正", "负", "零"]),
        _open(98, "A2", "由 ρ 和 V 计算动压 q_bar 得到", loads["dynamic"], "表达式"),
        _open(99, "A2", "将气动力系数换成气流系有量纲力向量，结果为", loads["coefficient_force"]),
        _choice(100, "C2", "三个力分量采用相同归一化尺度，因此有符号分量系数向量 [-C_D, C_Ya, -C_L] 可以", "使用同一矩阵 B 直接转换", ["必须分别乘不同参考长度后才能转换"]),
        _choice(101, "C2", "三个力矩系数采用 b、c_bar、b 三个参考长度，当 b 不等于 c_bar 时，应当", "先恢复有量纲力矩，旋转后按输出轴参考长度归一化", ["直接将 B 乘 [C_la, C_ma, C_na]"]),
        _open(102, "A2", "按照当前归一化定义，机体系俯仰力矩系数 C_mA 为", loads["pitch_coefficient"], "表达式"),
        _choice(103, "C1", "换算前的力矩参考点是", "A", ["G", "N"]),
        _choice(104, "C1", "换算后的力矩参考点是", "G", ["A", "N"]),
        _open(105, "A2", "由共线距离关系，A 到机头参考点 N 的距离 X_N 为", loads["nose_distance"], "表达式"),
        _choice(106, "C1", "工程轴向位置 x_Ge 的正方向是机体", "后方", ["前方"]),
        _choice(107, "A1", "该方向与机体系 +X_b 方向", "相反", ["相同"]),
        _open(108, "A2", "因此质心 G 在以 A 为原点的机体系中的轴向坐标为", loads["lever_x"], "表达式"),
        _open(109, "A2", "按 X_b、Y_b、Z_b 的顺序，从 A 指向 G 的向量 r_AG 为", loads["arm_from_a"]),
        _open(110, "A2", "从 G 指向 A 的向量 r_GA 为", loads["arm_from_g"]),
        _choice(111, "C2", "参考点由 A 换到 G 后，同一气动力系的合力", "保持不变", ["必须取反", "必须增加一个力臂项"]),
        _choice(112, "C2", "为把关于 A 的力矩换成关于 G 的力矩，附加力矩可用", "r_GA×F_b", ["r_AG×F_b", "r_GA×M_A"]),
        _choice(113, "A2", "若统一使用从 A 指向 G 的向量 r_AG，换算关系应写为", "M_G=M_A-r_AG×F_b", ["M_G=M_A+r_AG×F_b", "M_G=F_b-r_AG×M_A"]),
        _choice(114, "A1", "仅有正轴向力 X 时，该力对 G 产生的附加俯仰力矩为", added[("X", "俯仰")], ["正", "负", "零"]),
        _choice(115, "A1", "仅有正轴向力 X 时，该力对 G 产生的附加偏航力矩为", added[("X", "偏航")], ["正", "负", "零"]),
        _choice(116, "A1", "仅有正侧向力 Y_b 时，该力对 G 产生的附加滚转力矩为", added[("Y_b", "滚转")], ["正", "负", "零"]),
        _choice(117, "A1", "仅有正侧向力 Y_b 时，该力对 G 产生的附加偏航力矩为", added[("Y_b", "偏航")], ["正", "负", "零"]),
        _choice(118, "A1", "仅有正法向力 Z 时，该力对 G 产生的附加滚转力矩为", added[("Z", "滚转")], ["正", "负", "零"]),
        _choice(119, "A1", "仅有正法向力 Z 时，该力对 G 产生的附加俯仰力矩为", added[("Z", "俯仰")], ["正", "负", "零"]),
        _open(120, "A2", "换算到质心后的滚转力矩 l_G 为", loads["roll_g"], "表达式"),
        _open(121, "A2", "换算到质心后的俯仰力矩 m_G 为", loads["pitch_g"], "表达式"),
        _open(122, "A2", "换算到质心后的偏航力矩 n_G 为", loads["yaw_g"], "表达式"),
        _open(123, "A2", "代入工程位置定义，质心处的俯仰力矩 m_G 为", loads["pitch_g_engineering"], "表达式"),
        _choice(124, "A1", "若 G 相对 A 仅沿机体前方偏移，即 x>0、y=z=0，且合力仅沿机体上方，那么附加俯仰力矩为", loads["nose_up"], ["抬头方向", "低头方向", "零"]),
        _choice(125, "A1", "若 r_AG 与 F_b 平行，参考点换算产生的附加力矩为", "零", ["一定为正", "一定为负"]),
        _choice(126, "A1", "若 G 相对 A 仅沿机体轴向移动，滚转力矩 l_G 与 l_A", "相同", ["必定相反", "不能建立关系"]),
        _open(127, "A2", "若所有参考长度、参考面积和动压均保持不变，定义机体系力系数为 [X, Y_b, Z]=q_bar*S*[C_X, C_Yb, C_Z]，则质心处的滚转力矩系数 C_lG 为", loads["roll_coefficient"], "表达式"),
        _open(128, "A2", "采用相同约定，质心处的俯仰力矩系数 C_mG 为", loads["pitch_coefficient_g"], "表达式"),
        _open(129, "A2", "采用相同约定，质心处的偏航力矩系数 C_nG 为", loads["yaw_coefficient_g"], "表达式"),
        _choice(130, "C2", "若同一气动力系从气流系转换到机体系并同时从 A 换算到 G，允许先在气流系完成参考点换算，但必须把同一个力臂向量也表达在", "气流系", ["机体系而不作转换", "任意坐标系"]),
        _choice(131, "A2", "在本稿右手正交坐标系条件下，若两条路径的力、力矩和力臂均保持一致，则先旋转后换参考点与先换参考点后旋转的最终机体系质心力矩", "相同", ["互为相反数", "一般没有关系"]),
    ]
    return rows


def build(spec: dict) -> dict:
    model = FlightFrameModel().declare(spec)
    for parent, child in (
        ("ground", "body"),
        ("ground", "track"),
        ("body", "stability"),
        ("stability", "airflow"),
    ):
        if model.alignment(parent, child) != EYE:
            raise DefinitionError(f"{parent} and {child} are not aligned at zero angles.")

    ground_body, ground_body_text = _product(model, (("ground", "body", PSI), ("ground", "body", THETA), ("ground", "body", PHI)))
    body_from_ground = _transpose_product(ground_body_text)
    if sp.simplify(_parse_product(body_from_ground) - ground_body.T) != sp.zeros(3):
        raise DefinitionError("bRg is not the transpose of gRb.")
    if "ψ" not in body_from_ground.split("*")[-1]:
        raise DefinitionError("The yaw factor is not the rightmost factor of bRg.")
    ground_track, ground_track_text = _product(model, (("ground", "track", VARPHI), ("ground", "track", MU)))
    track_air, track_air_text = _product(model, (("track", "airflow", GAMMA),))
    if sp.simplify(track_air * E1 - E1) != sp.zeros(3, 1):
        raise DefinitionError("Track roll changes the airspeed axis.")
    body_stability, body_stability_text = _product(model, (("body", "stability", ALPHA),))
    stability_air, stability_air_text = _product(model, (("stability", "airflow", BETA),))
    body_air = sp.simplify(body_stability * stability_air)
    body_air_text = f"{body_stability_text}*{stability_air_text}"
    if sp.simplify(body_air - _parse_product(body_air_text)) != sp.zeros(3):
        raise DefinitionError("The airflow product does not match its factors.")

    nose = sp.simplify(ground_body * E1)
    track_speed = sp.simplify(ground_track * E1)
    body_speed = sp.simplify(body_air * E1)
    beta_formula = _recovers(body_speed, SIDESLIP, [
        sp.atan2(VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(-VY, sp.sqrt(VX**2 + VZ**2)),
        sp.atan2(VZ, sp.sqrt(VX**2 + VY**2)),
        sp.atan2(-VZ, sp.sqrt(VX**2 + VY**2)),
    ])
    alpha_formula = _recovers(body_speed, ATTACK, [
        sp.atan2(VZ, VX),
        sp.atan2(-VZ, VX),
        sp.atan2(VY, VX),
        sp.atan2(-VY, VX),
    ])
    nose_example = _vector(nose.subs({YAW: sp.Symbol("deg30"), PITCH: sp.Symbol("deg20"), ROLL: sp.Symbol("deg15")})).replace("deg30", "30°").replace("deg20", "20°").replace("deg15", "15°")
    track_z = _at_degrees(track_speed[2], {AZIMUTH: 40, INCLINE: 10})
    nose_word = _component_word(nose[2], {YAW: 30, PITCH: 20, ROLL: 15})
    speed_word = _component_word(track_speed[2], {AZIMUTH: 40, INCLINE: 10, BANK: 0})
    zero_incline = sp.simplify(track_speed[2].subs({INCLINE: 0, BANK: 0}))
    if zero_incline != 0:
        raise DefinitionError(f"Zero inclination still leaves a normal component {zero_incline}.")
    positive_beta = sp.N(body_speed[1].subs({ATTACK: 0.2, SIDESLIP: 0.3}))
    positive_alpha = sp.N(body_speed[2].subs({ATTACK: 0.3, SIDESLIP: 0.1}))
    _check_closure(ground_body, body_air, ground_track, track_air)
    _check_rotation(ground_body)
    _check_rotation(ground_track)
    _check_rotation(body_air)
    loads = build_loads(body_speed, ATTACK, SIDESLIP, _show)

    yaw_side = _sign_word(model, PSI, {"ground_right": "地面右方", "ground_left": "地面左方"})
    pitch_side = _sign_word(model, THETA, {"ground_up": "上方", "ground_down": "下方"})
    incline_side = _sign_word(model, MU, {"ground_up": "上方", "ground_down": "下方"})
    attack_side = _sign_word(model, ALPHA, {"body_down": "下方", "body_up": "上方"})
    sideslip_side = _sign_word(model, BETA, {"body_right": "右方", "body_left": "左方"})
    angles = {name: model.binding(name).definition for name in SYMBOL}
    if model.binding(PHI).sign == POSITIVE:
        angles[PHI] += "。在前、右、下坐标下，右翼向下为正"

    questions = _questions(
        ground_body_text=ground_body_text,
        body_from_ground=body_from_ground,
        ground_track_text=ground_track_text,
        track_air_text=track_air_text,
        body_stability_text=body_stability_text,
        stability_air_text=stability_air_text,
        body_air_text=body_air_text,
        nose_example=nose_example,
        nose_word=nose_word,
        track_speed=_vector(track_speed),
        track_z=track_z,
        speed_word=speed_word,
        body_speed=_vector(body_speed),
        beta_text=_formula(beta_formula),
        alpha_text=_formula(alpha_formula),
        beta_sign="正" if positive_beta > 0 else "负",
        alpha_sign="正" if positive_alpha > 0 else "负",
        yaw_side=yaw_side,
        pitch_side=pitch_side,
        incline_side=incline_side,
        attack_side=attack_side,
        sideslip_side=sideslip_side,
    )
    questions.extend(_load_questions(loads))
    if [item["id"] for item in questions] != list(range(1, BLANK_COUNT + 1)):
        raise DefinitionError("Question numbers are not 1 through 131.")
    return {
        "model": model,
        "angles": angles,
        "ground_body": ground_body_text,
        "body_from_ground": body_from_ground,
        "ground_track": ground_track_text,
        "track_air": track_air_text,
        "body_stability": body_stability_text,
        "stability_air": stability_air_text,
        "body_air": body_air_text,
        "track_speed": _vector(track_speed),
        "body_speed": _vector(body_speed),
        "questions": questions,
    }


def _parse_product(text: str) -> sp.Matrix:
    matrix = EYE
    for part in text.split("*"):
        matrix *= _parse_factor(part)
    return matrix


def _check_rotation(matrix: sp.Matrix) -> None:
    if sp.simplify(matrix.T * matrix - EYE) != sp.zeros(3):
        raise DefinitionError("A pose matrix is not orthogonal.")
    if sp.simplify(matrix.det() - 1) != 0:
        raise DefinitionError("A pose matrix does not have determinant +1.")


def _check_closure(ground_body: sp.Matrix, body_air: sp.Matrix, ground_track: sp.Matrix, track_air: sp.Matrix) -> None:
    """The angles describe one state: airspeed axes meet, and one bank matches the frames."""
    values = {YAW: 0.4, PITCH: -0.25, ROLL: 0.15, ATTACK: 0.3, SIDESLIP: -0.2}
    target = sp.N((ground_body * body_air).subs(values))
    speed = sp.simplify(ground_track * E1)
    equations = [speed[index] - target[index, 0] for index in (0, 1)]
    for guess in ((0.2, -0.1), (0.2, 0.4), (-0.3, 0.5), (1.0, -0.6), (-1.0, 0.6)):
        try:
            solution = sp.nsolve(equations, [AZIMUTH, INCLINE], guess)
        except (ValueError, TypeError, AttributeError):
            continue
        pose = sp.N(ground_track.subs({AZIMUTH: solution[0], INCLINE: solution[1]}))
        predicted = pose * E1
        if max(abs(predicted[index] - target[index, 0]) for index in range(3)) > sp.Float("1e-8"):
            continue
        relative = pose.T * target
        bank = sp.atan2(relative[2, 1], relative[1, 1])
        combined = sp.N((ground_track * track_air).subs({AZIMUTH: solution[0], INCLINE: solution[1], BANK: bank}))
        if max(abs(combined[index, column] - target[index, column]) for index in range(3) for column in range(3)) <= sp.Float("1e-8"):
            return
    raise DefinitionError("The track path and the body path do not describe the same airflow attitude.")


def _questions(**text) -> list[dict]:
    g = text["ground_body_text"]
    p = text["ground_track_text"]
    bs = text["body_stability_text"]
    sa = text["stability_air_text"]
    ba = text["body_air_text"]
    rows = [
        _choice(1, "C1", "地面系、机体系和航迹系的法向正方向分别指向", "下方", ["上方"]),
        _choice(2, "C1", "地面系、机体系和航迹系的侧向正方向分别指向", "右方", ["左方"]),
        _choice(3, "C1", "机体前方、机体右方、机体下方构成", "右手系", ["左手系"]),
        _choice(4, "C1", "偏航角为正时，机体前方在地面水平面上的投影向", f"{text['yaw_side']}偏转", ["地面右方偏转", "地面左方偏转"]),
        _choice(5, "C1", "俯仰角为正时，机头指向地面", text["pitch_side"], ["上方", "下方"]),
        _choice(6, "C1", "滚转角 Φ 绕机体的", "轴向", ["侧向", "法向"]),
        _choice(7, "C2", "本题坐标转换采用", "被动", ["主动"]),
        _choice(8, "C2", "姿态角采用", "动轴内旋", ["定轴外旋"]),
        _choice(9, "C2", f"按偏航、俯仰、滚转构造姿态后，将地面系分量映射到机体系的矩阵写成 bRg={text['body_from_ground']}，其中偏航矩阵位于连乘式的", "最右侧", ["最左侧"]),
        _choice(10, "C2", "将机体系坐标转换到地面系时，使用矩阵 gRb，该矩阵的物理映射方向是", "机体系到地面系", ["地面系到机体系"]),
        _choice(11, "C2", "将地面系向量转换到机体系时，应使用 gRb 的", "转置", ["本身"]),
        _choice(12, "C2", "按定义，gRb 的复合形式为", g, ["Cx(Φ)*Cy(θ)*Cz(ψ)", "Cz(ψ)*Cy(θ)*Cx(Φ)", "Cy(θ)*Cz(ψ)*Cx(Φ)"]),
        _open(13, "C1", "机体前方单位向量在机体系中的坐标为", "[1, 0, 0]"),
        _choice(14, "C2", "将机体前方轴映射到地面系，应计算", "gRb*[1, 0, 0]", ["gRb^T*[1, 0, 0]"]),
        _open(15, "A2", "在 ψ=30°、θ=20°、Φ=15° 时，机体前方轴在地面系中的三个分量为", text["nose_example"]),
        _choice(16, "A1", "由该地面法向分量的符号判断，机头指向地面", text["nose_word"], ["上方", "下方"]),
        _choice(17, "C2", "给定完整姿态矩阵时，机体右方轴在地面系中的方向由矩阵的第", "二列", ["一列", "三列"]),
        _choice(18, "C2", "机体下方轴在地面系中的方向由矩阵的第", "三列", ["一列", "二列"]),
        _choice(19, "C1", "正交姿态矩阵的转置等于其", "逆矩阵", ["负矩阵", "行列式"]),
        _choice(20, "C1", "任意合法三维旋转矩阵的行列式为", "+1", ["-1", "0"]),
        _choice(21, "C1", "航迹方位角 φ 描述空速水平投影相对", "Xg", ["Yg", "Zg"]),
        _choice(22, "C1", "航迹倾斜角 μ 为正表示飞行器向地面", text["incline_side"], ["上方", "下方"]),
        _choice(23, "C1", "航迹滚转角 γ 使气流系相对航迹系绕共同的空速轴转动，该轴是航迹系的", "Xp", ["Yp", "Zp"]),
        _choice(24, "C1", "地面系看到的航迹系姿态矩阵记作", "gRp", ["pRg"]),
        _choice(25, "C2", "按当前定义，不含滚转的 gRp 的复合形式为", p, _track_wrongs(p, text["track_air_text"])),
        _open(26, "C1", "航迹系单位空速向量为", "[1, 0, 0]"),
        _choice(27, "C2", "航迹系单位空速向量转换到地面系，应使用", "gRp", ["gRp^T"]),
        _choice(28, "A2", "空速的地面系法向分量为", text["track_z"], [text["track_z"].replace("-", "") if text["track_z"].startswith("-") else f"-{text['track_z']}", "cos(10°)", "sin(10°)"]),
        _choice(29, "A1", "由该地面法向分量的符号判断，空速指向地面", text["speed_word"], ["上方", "下方"]),
        _choice(30, "A2", "当航迹倾斜角为零时，空速地面系法向分量为", "0", ["+1", "-1"]),
        _choice(31, "A1", "在 μ=0 时，航迹方位角改变会使空速水平投影在 Xg、Yg 平面内", "转动", ["保持不变"]),
        _choice(32, "A1", "航迹滚转角 γ 改变时，空速方向本身", "不改变", ["改变"]),
        _choice(33, "A1", "航迹滚转会改变", "气流系 Ya、Za 相对航迹系的方向", ["航迹系 Yp、Zp 轴", "空速大小"]),
        _open(34, "A2", "地面系空速向量可由航迹方向写成", text["track_speed"]),
        _choice(35, "C1", "稳定性系 Xs 轴沿空速在飞行器", "对称面内的投影", ["机翼展向投影"]),
        _choice(36, "C1", "迎角 α 为正时，空速在机体对称面内偏向机体", text["attack_side"], ["上方", "下方"]),
        _choice(37, "C1", "侧滑角 β 为正时，空速偏向机体", text["sideslip_side"], ["左方", "右方"]),
        _choice(38, "C2", "机体系到稳定性系的姿态差异只由", "迎角", ["侧滑角", "航迹方位角"]),
        _choice(39, "C2", "稳定性系到气流系的姿态差异只由", "侧滑角", ["迎角", "偏航角"]),
        _choice(40, "C2", "按定义，稳定性系到机体系的矩阵为", bs, ["Cy(-α)", "Cy(α)", "Cz(-β)"]),
        _choice(41, "C2", "气流系到稳定性系的矩阵为", sa, ["Cz(-β)", "Cz(β)", "Cy(-α)"]),
        _choice(42, "C2", "气流系到机体系的复合矩阵为", ba, ["Cy(-α)*Cz(-β)", "Cz(-β)*Cy(-α)", "Cy(α)*Cz(β)"]),
        _open(43, "C1", "空速在气流系中的单位向量为", "[1, 0, 0]"),
        _choice(44, "C2", "空速从气流系变换到机体系，应使用", "bRa", ["bRa^T"]),
        _open(45, "A2", "空速在机体系中的单位向量为", text["body_speed"]),
        _choice(46, "A1", "机体系空速的 yb 分量为正时，空速指向机体", "右方", ["左方"]),
        _choice(47, "A1", "机体系空速的 zb 分量为正时，空速指向机体", "下方", ["上方"]),
        _choice(48, "A2", "仅由机体系单位空速向量 [vx, vy, vz] 恢复侧滑角时，可用", text["beta_text"], [
            "atan2(vy, sqrt(vx^2+vz^2))",
            "atan2(-vy, sqrt(vx^2+vz^2))",
            "atan2(vz, sqrt(vx^2+vy^2))",
            "atan2(vx, vy)",
        ]),
        _choice(49, "A1", "在 vx>0 且 vy>0 时，侧滑角符号为", text["beta_sign"], ["正", "负"]),
        _choice(50, "A1", "在 vx>0 且 vz>0 时，迎角符号为", text["alpha_sign"], ["正", "负"]),
        _choice(51, "A2", "迎角的稳定表达式为", text["alpha_text"], ["atan2(vz, vx)", "atan2(-vz, vx)", "atan2(vy, vx)"]),
        _choice(52, "C1", "gRb 和 gRp 都把各自子坐标系分量映射到", "地面系", ["气流系"]),
        _choice(53, "C2", "由航迹角得到地面系空速后，转换到机体系应计算", "gRb^T*vg", ["gRb*vg"]),
        _choice(54, "C2", "由迎角、侧滑角得到机体系空速时，γ 和 Φ", "不直接进入机体系空速分量公式", ["必须加入"]),
        _open(55, "A2", "由航迹角构造的地面系空速为", text["track_speed"]),
        _choice(56, "A1", "将上述空速转换到机体系后，其结果应与由 α、β 直接构造的机体系空速", "相同", ["相反", "一般不相关"]),
        _choice(57, "A2", "在输入角度描述同一飞行状态且忽略数值舍入的条件下，两条路径结果不一致，说明计算或约定使用中", "至少存在一处错误", ["必定不存在错误"]),
        _choice(58, "C2", "γ 不改变空速单位向量的地面系坐标，比较空速向量只能验证共同的空速轴而不能验证绕该轴的滚转，因此验证空速方向时可以", "只比较空速向量", ["忽略空速轴向", "只比较滚转角"]),
        _choice(59, "A2", "对两条路径得到的单位向量作闭合验证，要求其差向量的范数为", "0", ["1", "2"]),
    ]
    return rows


def _slot(item: dict) -> str:
    if item["options"] is None:
        choices = "无"
        answer = item["answer"]
    else:
        choices = format_choices(item["options"])
        answer = answer_letter(item["answer"], item["options"])
        if answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise DefinitionError(f"ID {item['id']} answer is not a letter.")
        chosen = item["options"][ord(answer) - ord("A")]
        if chosen != item["answer"]:
            raise DefinitionError(f"ID {item['id']} letter {answer} points at {chosen}, not {item['answer']}.")
    return f"{{ID:{item['id']}, 选项:{choices}, 输出格式:{item['format']}, 标签:{item['label']}, 答案:{answer}}}"


def _line(item: dict) -> str:
    line = f"{item['id']}. {item['stem']}{_slot(item)}。"
    matched = SLOT_LINE.fullmatch(line)
    if not matched or matched.group(1) != matched.group(2) or line.count("。") != 1 or line.count("{ID:") != 1:
        raise DefinitionError(f"ID {item['id']} is not one sentence with one slot.")
    return line


def _load_definition(angles: dict) -> list[str]:
    return [
        "## 气动力与参考点定义",
        "",
        "这一节只送给气动力子过程和参考点子过程。",
        "",
        "固定约定",
        "- 向量是列向量，矩阵左乘列向量。",
        "- T_a←b 把同一物理向量的机体系分量映成气流系分量。反向变换用它的转置。",
        "- T_s←b 是机体系到稳定性系，T_a←s 是稳定性系到气流系。定义不给出这两个矩阵的元素。",
        "- 三个坐标系都是右手系。分量按 X、Y、Z 书写。",
        "- 力是作用在飞行器上的气动力。气动力和气动力矩按有符号分量处理。",
        "- 同一点上改变坐标表达，和改变力矩参考点，是两件事。力和力矩的坐标转换都从气流系到机体系，参考点保持为 A。",
        "- 角度进入三角函数时使用同一角度单位。本稿取 -90°<α<90°、-90°<β<90°，所以 cos(α)>0、cos(β)>0。",
        "",
        "方向和坐标轴",
        "- 前方、右方、下方分别是轴向、侧向、法向的正方向。前方 × 右方 = 下方。",
        "- X_b 指向机体前方，Y_b 指向机体右方，Z_b 指向机体下方。机体系原点在气动参考点 A。",
        "- X_s 是空速在机体对称面内的投影，Y_s 与 Y_b 同向，Z_s 由 X_s×Y_s 确定。原点与 A 重合。",
        "- X_a 沿飞行器相对空气的速度。Z_a 位于机体对称面内并垂直于 X_a，取与 Z_b 点积为正的分支。Y_a 由 Z_a×X_a 确定。原点与 A 重合。",
        "- 空速是飞行器相对空气的速度。空气相对飞行器的方向与它相反。",
        "",
        "气流角",
        f"- 迎角 α：{angles[ALPHA]}。",
        f"- 侧滑角 β：{angles[BETA]}。",
        "- 只分析迎角时令 β=0。只分析侧滑时令 α=0。",
        "",
        "气动力与力矩",
        "- D 是阻力大小，D≥0，方向沿 -X_a。Y 是侧力的有符号大小，正方向沿 +Y_a。L 是升力的有符号大小，正方向沿 -Z_a。",
        "- l_a、m_a、n_a 分别绕 +X_a、+Y_a、+Z_a 按右手定则为正。",
        "- X、Y_b、Z 分别沿 +X_b、+Y_b、+Z_b 为正。Y 是气流系侧力，Y_b 是机体系侧向力。",
        "- l_A、m_A、n_A 是关于 A 的力矩，分别绕 +X_b、+Y_b、+Z_b 按右手定则为正。",
        "- 在前、右、下机体系中，绕 +Y_b 的正俯仰力矩对应抬头，负值对应低头。",
        "",
        "动压与系数",
        "- ρ>0，V>0，动压 q_bar=ρ*V^2/2。S>0 是参考面积，b>0 是参考翼展，c_bar>0 是参考平均气动弦长。",
        "- D=q_bar*S*C_D，Y=q_bar*S*C_Ya，L=q_bar*S*C_L。",
        "- l_a=q_bar*S*b*C_la，m_a=q_bar*S*c_bar*C_ma，n_a=q_bar*S*b*C_na。",
        "- l_A=q_bar*S*b*C_lA，m_A=q_bar*S*c_bar*C_mA，n_A=q_bar*S*b*C_nA。",
        "",
        "参考点",
        "- A 是原气动数据的力矩参考点，G 是质心，也是换算后的力矩参考点。换算时机体系轴方向不变。",
        "- 机头参考点是 N。N、A 和一个机尾参考点沿机体轴向共线。L_f 是 N 到机尾参考点的距离，L_A 是 A 到机尾参考点的距离，0<L_A<L_f。",
        "- X_N 是 A 到 N 的距离，由过程计算。",
        "- x_Ge 从 N 沿机体后方量到 G 的轴向投影，后方为正。y_Ge、z_Ge 从 A 沿 +Y_b、+Z_b 量取。",
        "- x_Ge 与 +X_b 相反，y_Ge、z_Ge 与对应的机体系分量同向。定义不直接给出 G 相对 A 的坐标，也不直接给出参考点换算公式。",
        "",
    ]


def render(sample: dict, record: dict) -> str:
    angles = record["angles"]
    questions = {item["id"]: item for item in record["questions"]}
    blocks = [
        ("一", "地面系到机体系", range(1, 21)),
        ("二", "地面系到航迹系", range(21, 35)),
        ("三", "机体系、稳定性系与气流系", range(35, 52)),
        ("四", "地面系与机体系闭环", range(52, 60)),
        ("五", "气动力和气动力矩从气流系转换到机体系", range(60, 103)),
        ("六", "力矩从气动参考点换算到质心", range(103, 132)),
    ]
    before = {
        28: [
            "取 φ=40°、μ=10° 计算地面系空速。",
        ],
        52: [
            "vg 是由航迹方位角和航迹倾斜角得到的地面系单位空速向量。",
            "gRb、gRp 分别把机体系、航迹系分量映到地面系。bRa 把气流系分量映到机体系，pRa 把气流系分量映到航迹系。",
            "gRp 不含航迹滚转。pRa 只含气流系相对航迹系绕共同空速轴的滚转。",
            "本子过程的姿态角、航迹角、迎角和侧滑角描述同一飞行状态，并忽略数值舍入。这些角不能各自独立取值。",
            "同一状态下，机体系单位空速经 gRb 映到地面系，等于航迹系单位空速经 gRp 映到地面系。",
            "完整姿态还满足 gRb*bRa=gRp*pRa。空速向量相等只验证共同的空速轴，不能验证绕该轴的滚转。",
        ],
        79: [
            "在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。",
        ],
        85: [
            "下面只保留升力，取 β=0、Y=0、D=0、L>0，且 0°<α<90°。",
        ],
        101: [
            "本子过程直接给出机体系中的合力 F_b=[X, Y_b, Z]，以及关于 A 点的力矩 M_A=[l_A, m_A, n_A]。",
            "所有分量都已经在同一个机体系里。本子过程只改变力矩参考点，不再改变坐标轴方向。",
        ],
        112: [
            "以下将 r_AG 简记为 [x, y, z]。",
            "本段只考察附加力矩的方向，令 x>0、y>0、z>0，并逐次单独保留一个正力分量。未提到的力分量都是零。",
        ],
        118: [
            "下面恢复所有有符号力分量，不再假设它们为正。",
        ],
    }
    body = []
    for key, heading, numbers in blocks:
        if key == "五":
            body.extend(_load_definition(angles))
        body.append(f"### 子过程{key}：{heading}")
        body.append("")
        for number in numbers:
            if number in before:
                body.extend(before[number])
                body.append("")
            body.append(_line(questions[number]))
        body.append("")
    return "\n".join([
        f"# 数据 {sample['id']} {sample['title']}",
        "",
        "本文件是飞行力学领域审阅稿。编号 1 到 131。前四个子过程做地面、航迹、机体和气流之间的坐标转换。后两个子过程做气动力六分量转换和力矩参考点换算。气动力子过程与参考点子过程单独注入：参考点子过程直接给出机体系中的力和力矩，不使用气动力子过程的回答。这里不做飞行包线判断，不做误差补偿。",
        "",
        f"本条定义：{sample['change']}",
        "",
        "审阅说明、标签、答案和生成注记不送入模型。输出格式规范送入。前四个子过程使用前面的定义声明和物理常识。后两个子过程使用气动力与参考点定义，不携带前四个子过程的正文。送入模型的还有当前子过程名，以及从该子过程开头直到当前句号的正文。当前句之前的空写成答案原文。每句只有一个待填槽。",
        "",
        "## 输出格式规范",
        "",
        "- 无：选择题。值只写选项字母。示例：A",
        "- 向量：一行三个分量，形如 [a, b, c]。逗号后恰好一个空格。顺序是 X、Y、Z。示例：[1, 0, 0]",
        "- 矩阵：一行 3×3，形如 [[a, b, c], [d, e, f], [g, h, i]]。逗号后恰好一个空格。示例：[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "- 表达式：一行。乘号用 *。示例：a*cos(α)+b",
        "",
        "## 定义声明",
        "",
        "固定约定",
        "- 坐标变换采用被动变换。姿态角、航迹角和气流角采用动轴内旋。",
        "- 向量是列向量，变换矩阵左乘列向量。坐标分量变换矩阵从右向左作用。",
        "- gRb 表示机体系相对地面系的姿态，把机体系分量映射到地面系。逆向变换使用其转置。",
        f"- 对偏航、俯仰、滚转这一动轴姿态构造，地面系到机体系的分量变换为 bRg={record['body_from_ground']}，逆向变换为其转置。其他变换按明确的起始系、目标系和中间系确定连乘顺序。",
        "- 分量按各系的 X、Y、Z 顺序书写。",
        "- 取 -90°<α、β、θ、μ<90°，使迎角投影、航迹水平投影及所用姿态角分支不发生本稿未处理的退化。方位角和滚转角采用主值区间 [-180°, 180°)。",
        "",
        "数学规则",
        "- 正交矩阵的转置等于它的逆矩阵。",
        "- 任意合法三维旋转矩阵的行列式为 +1。",
        "",
        "绝对方向与手性",
        "- 地面系、机体系、稳定性系、航迹系和气流系都是右手系。",
        "- 轴向正方向是前方，侧向正方向是右方，法向正方向是下方。前方 × 右方 = 下方。",
        "- 轴向、侧向、法向是物理方向，不是固定的坐标轴名。各系坐标轴的局部正方向不随姿态角改变。",
        "",
        "坐标系",
        "- 地面系 {g}：Xg 是水平面前方，Yg 是水平面右方，Zg 铅垂向下。地面系是固定参考系。",
        "- 机体系 {b}：xb 沿机头，yb 指向机体右方，zb 指向机体下方。机体系固定于飞行器。机体对称面是 xb-zb 平面。",
        "- 稳定性系 {s}：Xs 是空速在机体对称面上的投影并指向前方，Ys 垂直于对称面指向右方，Zs 在对称面内垂直于 Xs 并指向下方。",
        "- 航迹系 {p}：Xp 沿空速，Zp 位于过空速的铅垂面内、垂直于空速并取向下分支，Yp=Zp×Xp。航迹系不含滚转。这里的航迹使用空速方向，不是地速。",
        "- 气流系 {a}：Xa 沿空速。Za 位于机体对称面内、垂直于 Xa，并取与 Zb 点积为正的分支。Ya=Za×Xa。",
        "- 气流系相对航迹系绕共同的空速轴滚转。γ 按 +Xp 的右手定则，再按本条角度声明取正负。",
        "",
        "角度",
        f"- 偏航角 ψ：{angles[PSI]}。",
        f"- 俯仰角 θ：{angles[THETA]}。",
        f"- 滚转角 Φ：{angles[PHI]}。",
        f"- 航迹方位角 φ：{angles[VARPHI]}。",
        f"- 航迹倾斜角 μ：{angles[MU]}。",
        f"- 航迹滚转角 γ：{angles[GAMMA]}。它是气流系相对航迹系的滚转，不改变航迹系本身。",
        f"- 迎角 α：{angles[ALPHA]}。",
        f"- 侧滑角 β：{angles[BETA]}。",
        "",
        "基本旋转矩阵",
        "令 cη=cos(η)，sη=sin(η)。下列矩阵是绕当前坐标轴的被动变换，正角由该轴的右手定则确定。",
        "- Cx(η)=[[1, 0, 0], [0, cη, sη], [0, -sη, cη]]，绕轴向 X。",
        "- Cy(η)=[[cη, 0, -sη], [0, 1, 0], [sη, 0, cη]]，绕侧向 Y。",
        "- Cz(η)=[[cη, sη, 0], [-sη, cη, 0], [0, 0, 1]]，绕法向 Z。",
        "",
        "空速方向",
        "- 空速指飞行器相对空气的速度，大小 V>0。本稿不使用地速，也不引入风速。",
        "- 航迹系 Xp 和气流系 Xa 都沿空速，所以两系中的单位空速都是 [1, 0, 0]。",
        "",
        "## 物理常识",
        "",
        "地面系 Zg 指向下方。地面法向分量为正表示指向下方，为负表示指向上方。各角绕定义声明里的那根绝对轴，正方向以定义声明为准。坐标变换采用被动变换，姿态、航迹和气流角采用动轴内旋。",
        "",
        "## 完整过程",
        "",
        *body,
        "# 生成注记",
        "",
        "这一节不送入模型。基本旋转矩阵使用被动形式。gRb 把机体系分量映到地面系，等于 bRg 的转置。本条 gRb="
        + record["ground_body"]
        + "，bRg="
        + record["body_from_ground"]
        + "。航迹系不含滚转，gRp="
        + record["ground_track"]
        + "，pRa="
        + record["track_air"]
        + "，gRa=gRp*pRa。bRs="
        + record["body_stability"]
        + "，bRa="
        + record["body_air"]
        + "。角度符号改变后，这些因子、空速分量和气动力矩阵一起重算。定义声明不写入 gRb、gRp、bRs、bRa 和单位空速分量，避免直接抄写派生结果。",
        "",
        "气动力用同一套正交基。T_i←j 把 j 系分量映到 i 系，它的行是 i 系坐标轴在 j 系中的分量。定义不写 T_s←b 和 T_a←s 的元素。标准迎角下，机体系法向力里的侧力项是 -Y*sin(α)*sin(β)。参考点换算在右手分量下使用 M_G=M_A-r_AG×F_b，定义里不写这个公式。",
        "",
        "校验包括：同一飞行状态下空速轴闭合，并且存在一个航迹滚转使 gRb*bRa=gRp*pRa；气流基正交且为右手系；两段变换乘积等于整体变换；力和同点力矩的长度不变；力矩系数的参考长度尺度与有量纲力矩一致；参考点换算可以换回；先旋转后换参考点与先换点后旋转一致。",
        "",
    ])


def samples() -> list[dict]:
    def flipped(angle: str) -> dict:
        spec = deepcopy(STANDARD_SPEC)
        spec[angle] = {"sign": NEGATIVE}
        return spec

    return [
        {"id": "01", "title": "统一默认", "change": "五个坐标系和八个角度都取标准定义。", "spec": deepcopy(STANDARD_SPEC)},
        {"id": "02", "title": "偏航取负", "change": "偏航角改为机头向地面右方偏时为负。", "spec": flipped(PSI)},
        {"id": "03", "title": "俯仰取负", "change": "俯仰角改为机头向地面上方偏时为负。", "spec": flipped(THETA)},
        {"id": "04", "title": "滚转取负", "change": "滚转角改为向机体右侧滚转时为负。", "spec": flipped(PHI)},
        {"id": "05", "title": "航迹倾斜取负", "change": "航迹倾斜角改为向地面上方飞时为负。", "spec": flipped(MU)},
        {"id": "06", "title": "迎角取负", "change": "迎角改为空速投影处于机体轴向下方时为负。", "spec": flipped(ALPHA)},
        {"id": "07", "title": "侧滑取负", "change": "侧滑角改为空速处于飞行器右方时为负。", "spec": flipped(BETA)},
    ]


def _check_standard(record: dict) -> None:
    by_id = {item["id"]: item["answer"] for item in record["questions"]}
    expected = {
        4: "地面右方偏转",
        5: "上方",
        12: "Cz(-ψ)*Cy(-θ)*Cx(-Φ)",
        15: "[cos(20°)*cos(30°), sin(30°)*cos(20°), -sin(20°)]",
        16: "上方",
        22: "上方",
        9: "最右侧",
        25: "Cz(-φ)*Cy(-μ)",
        33: "气流系 Ya、Za 相对航迹系的方向",
        57: "至少存在一处错误",
        28: "-sin(10°)",
        29: "上方",
        34: "[cos(φ)*cos(μ), sin(φ)*cos(μ), -sin(μ)]",
        36: "下方",
        37: "右方",
        40: "Cy(α)",
        41: "Cz(-β)",
        42: "Cy(α)*Cz(-β)",
        45: "[cos(α)*cos(β), sin(β), sin(α)*cos(β)]",
        48: "atan2(vy, sqrt(vx^2+vz^2))",
        49: "正",
        50: "正",
        51: "atan2(vz, vx)",
        99: "[-q_bar*S*C_D, q_bar*S*C_Ya, -q_bar*S*C_L]",
        103: "A",
        68: "[-D, Y, -L]",
        72: "下方",
        73: "正",
        74: "[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]",
        75: "右方",
        76: "正",
        77: "[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]",
        79: "[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]",
        80: "[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]",
        81: "[[cos(α)*cos(β), sin(β), sin(α)*cos(β)], [-cos(α)*sin(β), cos(β), -sin(α)*sin(β)], [-sin(α), 0, cos(α)]]",
        83: "[[cos(α)*cos(β), -cos(α)*sin(β), -sin(α)], [sin(β), cos(β), 0], [sin(α)*cos(β), -sin(α)*sin(β), cos(α)]]",
        84: "-D*cos(α)*cos(β) - Y*cos(α)*sin(β) + L*sin(α)",
        86: "-D*sin(α)*cos(β) - Y*sin(α)*sin(β) - L*cos(α)",
        90: "[0, 0, -4]",
        93: "l_a*cos(α)*cos(β) - m_a*cos(α)*sin(β) - n_a*sin(α)",
        102: "(b/c_bar)*C_la*sin(β) + C_ma*cos(β)",
        105: "L_f - L_A",
        109: "[X_N - x_Ge, y_Ge, z_Ge]",
        114: "负",
        119: "正",
        120: "l_A + z*Y_b - y*Z",
        124: "低头方向",
        127: "C_lA + (z/b)*C_Yb - (y/b)*C_Z",
    }
    mismatches = {number: by_id[number] for number, answer in expected.items() if by_id[number] != answer}
    if mismatches:
        raise AssertionError(mismatches)


def _index(rows: list[dict]) -> str:
    lines = [
        "# 飞行力学空间坐标转换",
        "",
        "每条 131 个空，编号 1 到 131。前四个子过程做坐标转换。后两个子过程做气动力六分量和力矩参考点换算，并且单独注入。",
        "",
    ]
    for row in rows:
        lines.append(f"- [数据 {row['id']} {row['title']}](sample_{row['id']}.md)：{row['change']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    index_rows = []
    for sample in samples():
        record = build(sample["spec"])
        if sample["id"] == "01":
            _check_standard(record)
        document = render(sample, record)
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(document, encoding="utf-8")
        payload.append({
            "id": sample["id"],
            "title": sample["title"],
            "change": sample["change"],
            "domain": "飞行力学",
            "process": "空间坐标转换",
            "declaration_products": {
                "gRb": record["ground_body"],
                "gRp": record["ground_track"],
                "pRa": record["track_air"],
                "bRg": record["body_from_ground"],
                "bRs": record["body_stability"],
                "sRa": record["stability_air"],
                "bRa": record["body_air"],
            },
            "questions": [
                {
                    "id": str(item["id"]),
                    "label": item["label"],
                    "answer": item["answer"] if item["options"] is None else answer_letter(item["answer"], item["options"]),
                    "options": item["options"] or [],
                    "format": item["format"],
                }
                for item in record["questions"]
            ],
        })
        index_rows.append(sample)
        print(f"{sample['id']} {sample['title']}: {len(record['questions'])} blanks")
    (OUT_DIR / "samples.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "00_索引.md").write_text(_index(index_rows), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
