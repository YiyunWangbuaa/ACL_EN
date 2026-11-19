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
ROLE_CN = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
POINT_CN = {
    "forward": "前方", "rearward": "后方", "right": "右侧",
    "left": "左侧", "down": "下方", "up": "上方",
}
HAND_CN = {"left": "左手系", "right": "右手系"}
BASIC_ANGLES = ("pan", "tilt", "roll")
ANGLE_CHANGE = {
    "pan": "相机方位角改为光轴偏向相机1左侧为正",
    "tilt": "相机俯仰角改为光轴偏向相机1下方为正",
    "roll": "相机滚转角改为像面右缘偏向相机上方为正",
}
FRAME_TITLE = {"wind": "世界坐标系", "camera1": "相机1系", "camera2": "相机2系"}
FORBIDDEN = (
    "物体系", "机体", "风轴", "风洞", "天平", "激波", "空速", "机头",
    "迎角", "侧滑", "机构", "正装", "反装", "参考方向", "Xb", "Yb", "Zb", "地面",
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
        "相机2系绝对轴向", "相机2系绝对法向", "相机2系绝对侧向",
        "相机1系绝对轴向", "相机1系绝对法向", "相机1系绝对侧向",
    ]
    role_gold = {"pan": "相机2系绝对法向", "tilt": "相机2系绝对侧向", "roll": "相机2系绝对轴向"}

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        "首先确定相机1系到相机2系的转换。转换方向是？",
        ["相机2系向相机1系", "相机1系向相机2系", "相机2图像向相机1系", "相机1图像向相机2系"],
        "相机1系向相机2系", "C2",
    ), 1, "首先确定相机1系到相机2系的转换矩阵。转换方向是", "。")
    put(_choice("所有坐标系的变换采用主动还是被动？", ["主动", "被动"], "被动", "C2"),
        1, "所有坐标系的变换采用", "，")
    put(_choice("该变换视为动轴还是静轴？", ["动轴", "静轴"], "动轴", "C2"),
        1, "视为", "，")
    put(_choice("旋转使用外旋还是内旋？", ["外旋", "内旋"], "内旋", "C2"),
        1, "使用", "。")
    role_word = {"forward": "轴向", "up": "法向", "right": "侧向"}
    target_leads = [
        "由于相机2系的绝对轴向正方向（前方）、绝对法向正方向（上方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    source_leads = [
        "。相机1系的绝对轴向正方向（前方）、绝对法向正方向（上方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    for lead, direction in zip(target_leads, ("forward", "up", "right")):
        put(_choice(
            f"相机2系的绝对{role_word[direction]}对应哪个坐标轴？",
            axis_options, _axis_pointing(target, direction), "C1",
        ), 1, lead)
    for lead, direction in zip(source_leads, ("forward", "up", "right")):
        put(_choice(
            f"相机1系的绝对{role_word[direction]}对应哪个坐标轴？",
            axis_options, _axis_pointing(source, direction), "C1",
        ), 1, lead)
    put(_write_matrix("写出把相机1系坐标轴方向变到与相机2系相同的前置矩阵 RP1。", rp1),
        1, "。首先使用一个前置矩阵RP1将相机1系的坐标轴方向转换到与相机2系相同，形成相机2系平正状态的中间坐标系。RP1为")
    euler_leads = ["。基于该中间坐标系，欧拉角顺序第一个是", "，第二个是", "，第三个是"]
    for lead, name in zip(euler_leads, CAM_EULER):
        tail = "。" if name == "滚转角" else ""
        put(_choice("相机2 欧拉角顺序中的这一项是？", CAM_EULER, name, "A1"), 1, lead, tail)

    specs = (
        ("pan", "方位角", symbols[0], angles[0], "axial", 2),
        ("tilt", "俯仰角", symbols[1], angles[1], "axial", 3),
        ("roll", "滚转角", symbols[2], angles[2], "lateral", 4),
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
        subject = "像面右缘" if key == "roll" else "光轴"
        put(_choice(
            f"当相机2{angle_name} {symbol_name}>0 时，{subject}向相机1的哪一方偏？",
            SIX, side, "A1",
        ), paragraph, f"当相机2{angle_name} {symbol_name}>0 时，{subject}向相机1的", "偏。")
        put(_choice(
            f"相机2{angle_name} {symbol_name} 相当于绕哪个绝对方向轴旋转？",
            role_options, role_gold[key], "C2",
        ), paragraph, "这相当于绕", "旋转。")
        phrase, direction = _ROLE_POINTING[role]
        put(_choice(
            f"相机2系的绝对{phrase}对应哪个坐标轴？",
            axis_options, _axis_pointing(target, direction), "C1",
        ), paragraph, f"相机2系的绝对{phrase}对应", "。")
        put(_choice(
            f"相机2{angle_name} {symbol_name} 绕相机2系的哪个坐标轴旋转？",
            AXIS_NAMES, axis, "C1",
        ), paragraph, "绕相机2系的坐标轴", "，")
        put(_choice("绕该轴是正转还是反转？", THREE_SIGN, _bias(sign), "A1"),
            paragraph, "取", "转。")
        put(_write_matrix(f"写出相机2{angle_name} {symbol_name} 对应的旋转矩阵。", matrix),
            paragraph, f"对应的旋转矩阵R_{symbol_name}为", "。")

    factors = [f"R_{symbols[0]}", f"R_{symbols[1]}", f"R_{symbols[2]}", "RP1"]
    ordered = [factors[2], factors[1], factors[0], "RP1"]
    factor_leads = [
        "从相机1系到相机2系的坐标变换矩阵为R21=R1·R2·R3·R4。其中R1是",
        "。R2是", "。R3是", "。R4是",
    ]
    for lead, name in zip(factor_leads, ordered):
        tail = "。" if name == "RP1" else ""
        put(_choice("R21=R1·R2·R3·R4 中的这一项是？", factors, name, "A2"), 5, lead, tail)
    total = sp.simplify(matrices["roll"] * matrices["tilt"] * matrices["pan"] * rp1)
    for number, matrix in enumerate((matrices["roll"], matrices["tilt"], matrices["pan"], rp1), start=1):
        lead = (
            "在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R1的展开式为"
            if number == 1 else f"R{number}的展开式为"
        )
        put(_write_matrix(f"总矩阵前 R{number} 的展开式。", matrix), 5, lead, "。")
    put(_write_matrix("写出从相机1系到相机2系的旋转 R21（全展开）。", total),
        5, "写出R21（全展开），R21=", "。")
    if len(questions) != 41:
        raise RuntimeError(f"相机2过程应为 41 空，实际 {len(questions)}。")
    return questions, total


def build(spec: dict) -> dict:
    scene = cm.Scene().declare(spec)
    questions: list[dict] = []

    # 角度正方向里的“风轴右/上”改按相机1的轴读取：世界系就是相机1系。
    scene.wind = {"Xw": scene.cam1["X1"], "Yw": scene.cam1["Y1"], "Zw": scene.cam1["Z1"]}
    cam2_questions, r21 = _camera2_from_camera1(scene)
    questions.extend(cam2_questions)

    d_w = _coord_unit(scene.cam1, CAM_AXES[1], "forward")
    point = sp.simplify(scene.length * _coord_unit(scene.cam1, CAM_AXES[1], "up"))
    t21 = sp.simplify(-r21 * C2)
    p_c1 = sp.simplify(point)
    p_c2 = sp.simplify(r21 * point + t21)
    if sp.simplify(_skew(p_c1) * d_w) == sp.zeros(3, 1):
        raise RuntimeError("直线经过相机1光心，图像直线退化为零。")
    s1 = cm.camera_layout_permutation(scene.cam1, CAM_AXES[1])
    s2 = cm.camera_layout_permutation(scene.cam2, CAM_AXES[2])
    k1it = sp.simplify((K1 * s1).inv().T)
    k2it = sp.simplify((K2 * s2).inv().T)

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        "相机1图像中的直线 l 应满足的公式是？其中 d_w 是这条空间直线在相机1系中的方向向量。",
        [
            "l = (K1·S1)^(-T) · R21 · [P_c1]_× · d_w",
            "l = (K1·S1)^(-1) · [P_c1]_× · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · d_w + t21",
            "l = [P_c1]_× · (K1·S1)^(-T) · d_w",
            "l = (K1·S1)^(-T) · [P_c1]_× · R21^T · d_w",
        ],
        "l = (K1·S1)^(-T) · [P_c1]_× · d_w", "A2",
    ), 1, "相机1图像中的直线 l 应满足的公式是", "，其中 d_w 是这条空间直线在相机1系中的方向向量。")
    put(_choice(
        "相机2图像中的直线 l 应满足的公式是？其中 d_w 是这条空间直线在相机1系中的方向向量。",
        [
            "l = (K2·S2)^(-T) · R21 · [P_c2]_× · d_w",
            "l = (K2·S2)^(-1) · [P_c2]_× · R21 · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · R21 · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · R21^T · d_w",
            "l = (K2·S2)^(-T) · [P_c2]_× · d_w",
            "l = [P_c2]_× · (K2·S2)^(-T) · R21 · d_w",
        ],
        "l = (K2·S2)^(-T) · [P_c2]_× · R21 · d_w", "A2",
    ), 1, "相机2图像中的直线 l 应满足的公式是", "，其中 d_w 是这条空间直线在相机1系中的方向向量。")
    put(_write_matrix("写出相机1 的有效内参逆转置 (K1·S1)^(-T)。", k1it),
        2, "写出相机1 的有效内参逆转置 (K1·S1)^(-T)=", "。")
    put(_write_matrix("写出相机2 的有效内参逆转置 (K2·S2)^(-T)。", k2it),
        2, "写出相机2 的有效内参逆转置 (K2·S2)^(-T)=", "。")
    put(_choice(
        "相机2 相对相机1 的平移向量 t21 的公式形式是？",
        ["t21 = R21·C2", "t21 = −R21^T·C2", "t21 = −R21·C2", "t21 = C2 − R21"],
        "t21 = −R21·C2", "A2",
    ), 3, "相机2 相对相机1 的平移向量 t21 的公式形式是", "。")
    put(_write_vector("写出相机2 相对相机1 的平移向量 t21。", t21),
        3, "写出相机2 相对相机1 的平移向量t21，t21=", "。")
    put(_choice(
        "直线上的点 P_w 的公式是？",
        ["P_w = L·e_n", "P_w = L·d_w", "P_w = L·[0, 1, 0]^T", "P_w = e_n"],
        "P_w = L·e_n", "A2",
    ), 4, "这条直线平行于相机1光轴正方向，经过点 P_w。e_n 为相机1法向正方向在相机1系中的单位向量，P_w 到光心的距离为 L。P_w 的公式是", "。")
    put(_write_vector("写出该点在相机1系中的坐标 P_w。", point),
        4, "写出该点在相机1系中的坐标 P_w，P_w=", "。")
    put(_choice(
        "该点在相机1 系中的坐标 P_c1 的公式形式是？",
        ["P_c1 = R21·P_w + t21", "P_c1 = P_w", "P_c1 = R21^T·P_w + t21", "P_c1 = P_w + t21"],
        "P_c1 = P_w", "A2",
    ), 5, "该点在相机1 系中的坐标 P_c1 的公式形式是", "。")
    put(_write_vector("写出该点在相机1 系中的坐标 P_c1。", p_c1),
        5, "写出该点在相机1 系中的坐标 P_c1，P_c1=", "。")
    put(_choice(
        "该点在相机2 系中的坐标 P_c2 的公式形式是？",
        ["P_c2 = R21·P_w − t21", "P_c2 = R21^T·P_w + t21", "P_c2 = R21·P_w + t21", "P_c2 = P_w + t21"],
        "P_c2 = R21·P_w + t21", "A2",
    ), 5, "该点在相机2 系中的坐标 P_c2 的公式形式是", "。")
    put(_write_vector("写出该点在相机2 系中的坐标 P_c2。", p_c2),
        5, "写出该点在相机2 系中的坐标 P_c2，P_c2=", "。")
    put(_write_matrix("写出 P_c1 的反对称矩阵 [P_c1]_×。", sp.simplify(_skew(p_c1))),
        6, "写出 P_c1 的反对称矩阵 [P_c1]_×=", "。")
    put(_write_matrix("写出 P_c2 的反对称矩阵 [P_c2]_×。", sp.simplify(_skew(p_c2))),
        6, "写出 P_c2 的反对称矩阵 [P_c2]_×=", "。")

    put(_write_vector("写出这条直线的方向向量 d_w。", d_w),
        1,
        "直线平行于相机1光轴正方向，方向向量记为 d_w。"
        "相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·d_w，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R21·d_w。其中d_w=",
        "。")

    put(_choice(
        "本质矩阵 E 的表达式是？",
        ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
        "E = skew(t21)*R21", "A2",
    ), 1, "本质矩阵 E 的表达式是", "。")
    put(_choice(
        "基础矩阵 F 的表达式是？",
        ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)",
         "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
        "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2",
    ), 2, "基础矩阵 F 的表达式是", "。")
    put(_choice(
        "极线约束方程是？",
        ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
         "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
        "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2",
    ), 3, "极线约束方程是", "。")
    d1 = sp.simplify((K1 * s1).inv() * sp.Matrix([u1, v1, 1]))
    d2 = sp.simplify((K2 * s2).inv() * sp.Matrix([u2, v2, 1]))
    put(_write_vector(
        "相机1图像目标点的齐次坐标为 [u1,v1,1]，相机1系射线方向 d1 = (K1·S1)^(-1)·[u1,v1,1]^T。", d1),
        4, "相机1图像目标点的齐次坐标为 [u1,v1,1]。相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，其中d1=", "。")
    put(_write_vector(
        "相机2图像目标点的齐次坐标为 [u2,v2,1]，相机2系射线方向 d2 = (K2·S2)^(-1)·[u2,v2,1]^T。", d2),
        5, "相机2图像目标点的齐次坐标为 [u2,v2,1]。相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T，其中d2=", "。")
    put(_choice(
        "目标点在相机1系中的坐标沿射线方向，X_c1 = ?",
        ["X_c1 = λ1·d1", "X_c1 = d1 + λ1", "X_c1 = λ1 + d1", "X_c1 = d1 − λ1"],
        "X_c1 = λ1·d1", "A2",
    ), 6, "目标点在相机1系中的坐标 X_c1 沿射线方向 d1，其中 λ1 为深度。X_c1=", "。")
    put(_choice(
        "两个相机看到同一目标点，约束方程为 ?",
        ["λ2·d2 = R21·λ1·d1 + t21", "λ2·d2 = R21·λ1·d1 − t21",
         "λ2·d2 = R21^T·λ1·d1 + t21", "λ1·d1 = R21·λ2·d2 + t21"],
        "λ2·d2 = R21·λ1·d1 + t21", "A2",
    ), 7, "两个相机看到同一目标点，相机1系到相机2系的变换为 X_c2=R21·X_c1+t21，故约束方程为", "。")
    put(_choice(
        "在约束方程两侧叉乘哪个向量可消去 λ2 ?",
        ["d2", "d1", "R21·d1", "t21"],
        "d2", "A2",
    ), 8, "在约束方程两侧叉乘", "，消去λ2，得 d2×(λ1·R21·d1+t21)=0。")
    put(_choice(
        "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
        "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1。λ1 = ?",
        ["λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = −(d2×t21) / (d2×R21·d1)",
         "λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²"],
        "λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²", "A2",
    ), 9,
        "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
        "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1，λ1=",
        "。")
    put(_choice(
        "目标点在相机1系中的坐标 X_c1 已知，它在世界系中的坐标 X_w = ?",
        ["X_w = X_c1", "X_w = R21^T·(X_c1 − t21)", "X_w = R21·X_c1 + t21", "X_w = X_c1 + t21"],
        "X_w = X_c1", "A2",
    ), 10, "目标点在相机1系中的坐标 X_c1 已知。相机1系就是世界系，所以它在世界系中的坐标 X_w =", "。")

    bounds = [
        (41, "camera_attitude"),
        (55, "projection"),
        (56, "line_direction"),
        (66, "epipolar"),
    ]
    if len(questions) != bounds[-1][0]:
        raise RuntimeError(f"基础理论题应为 {bounds[-1][0]} 空，实际 {len(questions)}。")

    def segment_of(index: int) -> str:
        for bound, name in bounds:
            if index <= bound:
                return name
        return "epipolar"

    for index, item in enumerate(questions, start=1):
        item["local_id"] = index
        item["id"] = str(index)
        item["process"] = "机器视觉基础理论"
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
        f"{title}改为{HAND_CN[spec['chirality']]}，"
        f"{x_name}沿{ROLE_CN[spec[x_name]['along']]}指向{POINT_CN[spec[x_name]['points']]}，"
        f"{y_name}沿{ROLE_CN[spec[y_name]['along']]}指向{POINT_CN[spec[y_name]['points']]}"
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
        return "统一默认。相机1、相机2都取默认坐标轴，相机俯仰、方位、滚转都取默认正方向。"
    origin = f"来自{source}。" if source else ""
    return origin + "相对统一默认，只保留两台相机的差异：" + "；".join(phrases) + "。"


def _declaration(spec: dict, scene: cm.Scene) -> str:
    chirality = {"right": "右", "left": "左"}

    def frame_block(title: str, names: tuple, frame: dict, frame_spec: dict) -> str:
        lines = [f"{title}（{chirality[frame_spec.get('chirality', 'left')]}手系）："]
        for name in names:
            vector = frame[name]
            role = cm.role_of(vector)
            pointing = DIR_CN[DIR_NAME[vector]]
            inferred = "声明给出" if name in frame_spec else "由手性推断"
            lines.append(f"- {name}：沿{ROLE_CN[role]}，指向{pointing}（{inferred}）。")
        return "\n".join(lines)

    pitch = "正" if scene.angle_toward["tilt"] == "wind_up" else "负"
    yaw = "正" if scene.angle_toward["pan"] == "wind_right" else "负"
    roll = "正" if scene.angle_toward["roll"] == "camera_down" else "负"
    return "\n".join([
        "【定义声明】",
        "",
        "固定约定",
        "- 坐标系转换一律采用被动变换。姿态旋转一律为动轴内旋。",
        "- 相机成像采用针孔模型 x = (K·S)·[R|t]·X。图像直线 l = (K·S)^(-T)·n，n 为光心与空间直线张成平面的法向。",
        "- 极线约束 x2^T·F·x1 = 0，F = (K2·S2)^(-T)·E·(K1·S1)^(-1)，E = [t21]_×·R21。",
        "- 世界坐标系取为相机1系：相机1光心为世界原点，相机1的三根坐标轴就是世界系的三根坐标轴。相机1相对世界系的旋转为单位阵、平移为零。只需求相机1系到相机2系的旋转 R21 和平移 t21。",
        "- 相机1系中有一条直线。它平行于相机1光轴正方向。d_w 是该正方向在相机1系中的单位向量。e_n 是相机1法向正方向在相机1系中的单位向量。直线经过点 P_w = L·e_n，不经过光心。",
        "- 所有量均为符号：L、ψ2、θ2、φ2、u1、v1、u2、v2、λ1、相机2光心在相机1系（世界系）中的坐标为 C2=(x2,y2,z2)。",
        "- 相机内参 K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]]，K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]。K 按 [侧向正方向（右）, 法向正方向（上）, 轴向正方向（前）] 作用。S 把该相机坐标变到这个顺序，含取反。有效内参为 K·S。",
        "",
        "绝对方向",
        "世界、相机各有三根绝对轴。绝对轴的正方向定义相同：轴向正方向是前方，法向正方向是上方，侧向正方向是右侧。六个绝对方向是这三根轴上的两个指向。",
        "下文凡写轴向正方向、法向正方向、侧向正方向，以及世界轴向、相机轴向的正方向，都是这三根绝对轴的正方向，不是坐标轴的正方向。坐标轴声明只是指定每根坐标轴沿哪根绝对轴、指向哪一侧，可以与绝对正方向相同，也可以相反。",
        "两个系共用同一套指向关系：前方 × 右侧 = 下方。沿前方看去，右侧在右手侧，下方铅垂向下。这个关系不随定义改变。",
        "世界的三轴六方向：",
        "- 轴向：世界前后轴。正方向朝前。后方是其反方向。",
        "- 法向：世界铅垂轴。正方向朝上，上方铅垂向上。下方铅垂向下。",
        "- 侧向：世界水平横轴。正方向朝右。左侧是其反方向。",
        "相机的三轴六方向相对相机：",
        "- 轴向：光轴。正方向朝前，指向拍摄方向。后方是其反方向。",
        "- 法向：像面纵向。正方向朝上。下方是其反方向。",
        "- 侧向：像面横向。正方向朝右。左侧是其反方向。",
        "",
        "场景",
        "两台已标定、同步曝光的相机观测一条固定直线。相机1为参考相机，其坐标系即为世界坐标系。相机光轴沿相机轴向，像面横向沿相机侧向，像面纵向沿相机法向。",
        "任务有两个。其一是把这条直线映射到两台相机的图像中，写出图像直线。其二是给定两台相机图像上的同一目标点，按双目模型求出它在相机1系（世界系）中的坐标。",
        "子任务包括：由相机2与相机1系的关系，写出相机1系到相机2系的旋转 R21 和平移 t21；由该直线的方向和直线上的一点，计算它在两幅图像中的表达式；由两幅图像上的目标点，经射线、极线约束和深度，得到该点的相机1系坐标。相机1系就是世界系，所以这个坐标就是世界系坐标。",
        "",
        "角度",
        "相机俯仰角 θ2：相机2纵轴与相机1纵轴水平面的夹角，"
        f"相机2纵轴抬起时（相机2纵轴向相机1的上方偏时）为{pitch}。",
        "相机方位角 ψ2：相机2纵轴在水平面上的投影，与相机1系中沿轴向、指向前方的参考轴间的夹角。"
        f"相机2纵轴向相机1的右偏为{yaw}。",
        "相机滚转角 φ2：相机2法向与通过相机2纵轴的铅垂面的夹角，"
        f"向相机2的右侧滚转时为{roll}。",
        "相机2按方位角、俯仰角、滚转角的顺序内旋，记为 ψ2、θ2、φ2。",
        "",
        "坐标系",
        "沿轴向、指向前方的那根坐标轴是参考轴。另外两根坐标轴在法向与侧向之间绑定，绑定结果必须符合该系的手性。",
        frame_block("世界坐标系（=相机1系）", CAM_AXES[1], scene.cam1, spec["camera1"]),
        "",
        frame_block("相机2系", CAM_AXES[2], scene.cam2, spec["camera2"]),
        "",
        "物理常识",
        "轴向正方向、法向正方向、侧向正方向分别是前方、上方、右侧。相机方位角绕绝对法向轴，相机俯仰角绕绝对侧向轴，相机滚转角绕绝对轴向轴。各角的正方向以定义声明为准。坐标系转换采用被动变换。相机姿态采用动轴内旋。",
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
            "source": item["source"] or "标准",
            "title": "机器视觉基础理论" if item["source"] else "机器视觉基础理论·统一默认",
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
        raise RuntimeError(f"{record['id']} 仍有非基础理论用语：{bad}")
    for item in record["questions"]:
        if item["options"] and item["answer"] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            raise RuntimeError(f"{record['id']} 空 {item['id']} 的答案不是选项字母。")


def _titles() -> dict:
    return {
        "frame": "绝对方向对应坐标轴",
        "camera_attitude": "相机姿态",
        "projection": "成像与直线上的点",
        "line_direction": "图像直线的方向向量",
        "epipolar": "极线与三角化",
    }


def _export(rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    intro = (
        "本文件是机器视觉基础理论的审阅稿。世界坐标系就是相机1系，没有物体系。"
        "直线平行于相机1光轴正方向，经过法向正方向上距离光心为 L 的点，不经过光心。"
        "相机2相对相机1的位姿记为 R21、t21。图像上的目标点按双目射线反解回相机1系。"
        "槽里的标签和标准答案只给审阅用，模型看到的待填槽不含这两项。"
    )
    titles = _titles()
    for index, row in enumerate(rows, start=1):
        (OUT_DIR / f"sample_{index:02d}.md").write_text(
            rx._sample_markdown(row, index, intro=intro, titles=titles), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    lines = [
        "# 机器视觉基础理论",
        "",
        "这份数据从纯机器视觉题化简而来，只考察针孔成像、相机位姿、直线投影、双目外参和图像点反解。",
        "没有物体系，也没有机构、安装和姿态角。世界坐标系就是相机1系。相机2相对相机1的位姿记为 R21、t21。",
        "几何对象是相机1系中的一条直线：平行于相机1光轴正方向，经过法向正方向上距离光心为 L 的点，不经过光心。",
        "另一个任务是给出两台相机图像上的同一目标点，按双目射线求出它在相机1系中的坐标。",
        "",
        f"每条 {rows[0]['n_blanks']} 个空。正式输入格式见 00_注入格式.md。端到端提问见 [端到端](端到端.md)。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [数据 {index:02d}](sample_{index:02d}.md)：{row['id']}，{row['change']}")
    lines.append("")
    (OUT_DIR / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "00_注入格式.md").write_text(rx._format_document(example), encoding="utf-8")
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
