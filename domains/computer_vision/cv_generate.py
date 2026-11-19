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

DIR_CN = {"forward": "前方", "rearward": "后方", "right": "右侧",
          "left": "左侧", "down": "下方", "up": "上方"}
DIR_NAME = {value: key for key, value in cm.DIRECTIONS.items()}
SIX = ["前方", "后方", "右侧", "左侧", "下方", "上方"]
THREE_SIGN = ["正", "负"]
EULER = ["迎角", "侧滑角", "滚转角"]
CAM_EULER = ["方位角", "俯仰角", "滚转角"]
AXIS_ALL = ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb",
            "X1", "Y1", "Z1", "X2", "Y2", "Z2"]
AXIS_OPTIONS = AXIS_ALL + ["-" + name for name in AXIS_ALL]
ABS_DIRS = [("前方", "forward"), ("上方", "up"), ("右侧", "right")]

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
    "normal": ("法向正方向（上方）", "up"),
    "lateral": ("侧向正方向（右侧）", "right"),
    "axial": ("轴向正方向（前方）", "forward"),
}
SUBJECT = {"pan": "光轴", "tilt": "光轴", "roll": "像面右缘",
           "attack": "机头", "sideslip": "机头", "bank": "右翼"}
INDEX = {"pan": 0, "tilt": 1, "roll": 2, "attack": 0, "sideslip": 1, "bank": 2}

FRAME_NAMES = {
    "wind": ("Xw", "Yw", "Zw"),
    "body": ("Xb", "Yb", "Zb"),
    "camera1": ("X1", "Y1", "Z1"),
    "camera2": ("X2", "Y2", "Z2"),
}
FRAME_TITLE = {"wind": "风轴系", "body": "机体系", "camera1": "相机1系", "camera2": "相机2系"}
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
    "pan": "相机方位角改为光轴偏向风轴左侧为正",
    "tilt": "相机俯仰角改为光轴偏向风轴下方为正",
    "roll": "相机滚转角改为像面右缘偏向相机上方为正",
    "attack": "机构迎角改为机体轴向偏向风洞下方为正",
    "sideslip": "机构侧滑角改为机体轴向偏向风洞右侧为正",
    "bank": "机构滚转角改为与右手定则方向相反",
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
    return {"prompt": prompt, "label": label, "format": "无",
            "options": list(options), "answer": letters[options.index(correct)]}


def _write_matrix(prompt: str, matrix: sp.Matrix, label: str = "A2") -> dict:
    return {"prompt": prompt, "label": label, "format": "矩阵",
            "options": [], "answer": _m(matrix)}


def _v(matrix: sp.Matrix) -> str:
    cells = [sp.sstr(matrix[index, 0]) for index in range(matrix.rows)]
    return "[" + ", ".join(cells) + "]"


def _write_vector(prompt: str, matrix: sp.Matrix, label: str = "A2") -> dict:
    return {"prompt": prompt, "label": label, "format": "向量",
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
    return "正" if sign > 0 else "负"


_BODY_AXES = ("Xb", "Yb", "Zb")
_WIND_AXES = ("Xw", "Yw", "Zw")
_TUNNEL_SIDE = {
    "body_up": "上方", "body_down": "下方",
    "body_left": "左侧", "body_right": "右侧",
}
_FRAME_PAIR = ["机体系", "风轴系"]
_CONVERSION = [
    "机体系向机体系", "机体系向风轴系",
    "风轴系向机体系", "风轴系向风轴系",
]
_DIRECTED = (
    ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb"]
    + ["-Xw", "-Yw", "-Zw", "-Xb", "-Yb", "-Zb"]
)
_BODY_AXIS_OPTIONS = ["Xw", "Yw", "Zw", "Xb", "Yb", "Zb"]
_FACTORS = ["RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M"]


def _mechanism_definition(scene: cm.Scene) -> list[str]:
    """机构结构写出正方向，不写出欧拉顺序，顺序要由机构结构推断。"""
    attack = _TUNNEL_SIDE[scene.angle_toward["attack"]]
    sideslip = _TUNNEL_SIDE[scene.angle_toward["sideslip"]]
    hand = "相同" if scene.angle_toward["bank"] == "body_down" else "相反"
    return [
        "机构迎角 α_M：功角支撑杆在铅垂面上运动，可视为绕风轴系的侧向轴运动，"
        "为支撑杆（机体轴向）与水平面的夹角。"
        f"机体轴向偏向风洞{attack}时为正。",
        "机构侧滑角 β_M：侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，"
        "即可视为侧滑角机构基于功角机构在作动。"
        "β_M是侧滑角支撑杆与铅垂面的夹角。"
        f"使机体轴向偏向风洞{sideslip}时为正。",
        "机构滚转角 γ_M：机体绕侧滑角支撑杆的转角。"
        "角度为正时，以模型绝对轴向正方向（前方）旋转，与右手定则方向"
        f"{hand}。",
        "模型只考虑正装。",
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
        raise RuntimeError(f"{axis_name} 的正转没有转到声明的方向。")
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
        raise RuntimeError(f"{axis_name} 的被动矩阵与右手螺旋的转角不一致（点积 {dot}）。")


def _orientation(frame: dict, names: tuple[str, ...]) -> int:
    """+1 when the named axes are right-handed in the embedding, otherwise -1."""
    triple = cm.basis(frame, names).det()
    if triple == 0:
        raise RuntimeError(f"{names} 共面，无法判断手性。")
    return 1 if triple > 0 else -1


def signed_rotation(frame: dict, names: tuple[str, ...], axis_name: str, moving, toward, symbol):
    """Right-hand sign about the positive axis, and the passive matrix of that signed angle.

    正转 writes the textbook passive matrix of a positive angle. 负转 writes the
    same matrix with the angle negated. A left-handed basis does not get a
    second sign flip inside the matrix.
    """
    component = _algebraic_sign_named(frame, names, axis_name, moving, toward)
    handed = cm.sense(frame[axis_name], moving, toward)
    if component != _orientation(frame, names) * handed:
        raise RuntimeError(f"{axis_name} 的分量正转与右手螺旋不一致。")
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
    """风洞「风轴系到体轴系」前八段。体轴系写成机体系。不出现天平，安装只考虑正装。"""
    questions: list[dict] = []
    wind_basis = cm.basis(scene.wind, _WIND_AXES)
    body_basis = cm.basis(scene.body, _BODY_AXES)
    rp1 = sp.simplify(body_basis.T * wind_basis)
    rp2 = sp.eye(3)

    questions.extend([
        _placed(_choice(
            "方便起见，可依据机构角正向从哪种转换推导坐标转换矩阵？",
            _CONVERSION, "风轴系向机体系", "C2",
        ), 1, "初始状态，机体保持平正，动作机构产生了α_M、β_M、γ_M三个机构角使得模型在风轴系中的姿态发生了改变。方便起见，可依据机构角正向从"),
        _placed(_choice(
            "推导得到风轴系向机体系的矩阵后，应进行哪种运算才能得到机体系向风轴系的转换矩阵？",
            ["转置", "求逆", "取负", "不变换"], "转置", "C2",
        ), 1, "推导坐标转换矩阵，之后将矩阵进行"),
        _placed(_choice(
            "将风轴系向机体系的矩阵转置后，得到的是哪种转换？",
            _CONVERSION, "机体系向风轴系", "C2",
        ), 1, "即可得到从", "的转换矩阵，以求解机体两轴在风轴系中的表达式。"),
    ])

    role_of = {"up": "法向正方向（上方）", "forward": "轴向正方向（前方）", "right": "侧向正方向（右侧）"}
    wind_leads = [
        "由于风轴系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    body_leads = [
        "。机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    for lead, direction in zip(wind_leads, ("up", "forward", "right")):
        questions.append(_placed(_choice(
            f"风轴系的绝对{role_of[direction]}对应哪个坐标轴？",
            _DIRECTED, _axis_pointing(scene.wind, direction), "C1",
        ), 2, lead))
    for lead, direction in zip(body_leads, ("up", "forward", "right")):
        questions.append(_placed(_choice(
            f"机体系的绝对{role_of[direction]}对应哪个坐标轴？",
            _DIRECTED, _axis_pointing(scene.body, direction), "C1",
        ), 2, lead))
    questions.extend([
        _placed(_write_matrix("写出风轴系向机体系平正位置处的前置变换矩阵 RP1。", rp1),
                2, "。所以有风轴系向机体系平正位置处的前置变换矩阵RP1为"),
        _placed(_choice("RP1 使哪个坐标系的坐标轴与自身绝对方向的关系，与对照坐标系一致？", _FRAME_PAIR, "机体系", "C2"),
                2, "。该矩阵能够使"),
        _placed(_choice("RP1 对照的是哪个坐标系？", _FRAME_PAIR, "风轴系", "C2"),
                2, "的坐标轴与自身绝对方向的关系与"),
        _placed(_choice("模型安装方式为？", ["正装", "反装"], "正装", "C1"),
                2, "的坐标轴与自身绝对方向的关系一致。此外，模型安装方式为"),
        _placed(_choice("模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞的绝对法向正方向和绝对侧向正方向是否相同？",
                        ["相同", "相反"], "相同", "A1"),
                2, "。此时模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞的绝对法向正方向和绝对侧向正方向"),
        _placed(_write_matrix("写出安装引起的前置变换矩阵 RP2。", rp2),
                2, "。所以还需要另一个前置变换矩阵RP2为"),
        _placed(_choice("RP2 使哪个坐标系的坐标轴方向被对齐？", _FRAME_PAIR, "机体系", "C2"),
                2, "。该矩阵使得"),
        _placed(_choice("RP2 对照的坐标方向属于哪个坐标系？", _FRAME_PAIR, "风轴系", "C2"),
                2, "的坐标轴方向与", "的坐标方向相同。"),
    ])

    questions.extend([
        _placed(_choice("根据机构结构，第一个旋转角是？", EULER, "迎角", "A1"),
                3, "根据机构结构，推断第一个旋转角为"),
        _placed(_choice("第二个旋转角是？", EULER, "侧滑角", "A1"),
                3, "。第二个旋转角为"),
        _placed(_choice("第三个旋转角是？", EULER, "滚转角", "A1"),
                3, "。第三个旋转角为", "。"),
    ])

    specs = (
        ("迎角", "α_M", "attack", "lateral", cm.FORWARD, 4),
        ("侧滑角", "β_M", "sideslip", "normal", cm.FORWARD, 5),
        ("滚转角", "γ_M", "bank", "axial", cm.RIGHT, 6),
    )
    matrices = []
    for _title, symbol_name, key, role, moving, paragraph in specs:
        toward_name = scene.angle_toward[key]
        if key == "bank":
            toward_vec = cm.UP if toward_name == "body_up" else cm.DOWN
            hand = "相反" if toward_name == "body_up" else "相同"
        else:
            toward_vec = cm.DIRECTIONS[_TOWARD_KEY[toward_name]]
            nose_side = _TUNNEL_SIDE[toward_name]
        axis_name, _axis_vec = cm.axis_on(scene.body, role)
        symbol = sp.Symbol(symbol_name)
        sign, matrix = signed_rotation(scene.body, _BODY_AXES, axis_name, moving, toward_vec, symbol)
        matrices.append(matrix)
        if key == "bank":
            questions.append(
                _placed(_choice("当γ_M>0时，以模型绝对轴向正方向（前方）为旋转轴，与右手定则相同还是相反？",
                                ["相同", "相反"], hand, "C1"),
                        paragraph, "当γ_M>0时，以模型绝对轴向正方向（前方）为旋转轴，与右手定则"),
            )
            axis_lead = "绕坐标轴"
            matrix_lead = "，旋转γ_M角度。对应的旋转矩阵R_γ_M为"
        else:
            questions.append(
                _placed(_choice(f"当{symbol_name}>0时，模型纵轴向风洞{nose_side}偏。模型采用哪种安装？",
                                ["正装", "倒装"], "正装", "C1"),
                        paragraph, f"当{symbol_name}>0时，模型纵轴向风洞{nose_side}偏。模型采用"),
            )
            axis_lead = "平正时，绕"
            matrix_lead = f"，旋转{symbol_name}角度。对应的旋转矩阵R_{symbol_name}为"
        questions.append(
            _placed(_choice(f"这相当于哪个坐标系平正时绕轴旋转{symbol_name}？", _FRAME_PAIR, "机体系", "C2"),
                    paragraph, "。这相当于", "平正时。" if key == "bank" else ""),
        )
        if key == "bank":
            questions.append(_placed(_choice(
                "机体系的绝对轴向正方向（前方）对应哪个坐标轴？",
                _DIRECTED, _axis_pointing(scene.body, "forward"), "C1",
            ), paragraph, "机体系的绝对轴向正方向（前方）对应", "。"))
        questions.extend([
            _placed(_choice(f"{symbol_name} 绕机体系的哪个坐标轴旋转？", _BODY_AXIS_OPTIONS, axis_name, "C1"),
                    paragraph, axis_lead),
            _placed(_choice("绕该轴旋转方向为？", ["正", "负", "不确定"], _bias(sign), "A1"),
                    paragraph, "旋转。绕该轴旋转方向为"),
            _placed(_write_matrix(f"写出对应的旋转矩阵 R_{symbol_name}。", matrix),
                    paragraph, matrix_lead, "。"),
        ])

    questions.extend([
        _placed(_choice("此过程的坐标系转换采用主动还是被动？", ["主动", "被动"], "被动", "C2"),
                7, "此过程涉及到坐标系转换，应当采用坐标系的"),
        _placed(_choice("机体系是动轴还是定轴？", ["动轴", "定轴"], "动轴", "C2"),
                7, "。此外由于机体系是"),
        _placed(_choice("旋转采用内旋还是外旋？", ["内旋", "外旋"], "内旋", "C2"),
                7, "，故应当在旋转时采用"),
    ])
    order = ["R_γ_M", "R_β_M", "R_α_M", "RP2", "RP1"]
    factor_leads = [
        "。必须先变换坐标轴对应关系，之后再变换姿态。写出[Xb,Yb,Zb]^T=R1*R2*R3*R4*R5*[Xw,Yw,Zw]^T。R1到R5从前面得到的矩阵代号中选择。公式中对应的R1矩阵为",
        "。对应的R2矩阵为",
        "。对应的R3矩阵为",
        "。对应的R4矩阵为",
        "。对应的R5矩阵为",
    ]
    for lead, name in zip(factor_leads, order):
        tail = "。" if name == "RP1" else ""
        questions.append(_placed(_choice(
            f"公式中对应的{name}在 [Xb,Yb,Zb]^T=R1*R2*R3*R4*R5*[Xw,Yw,Zw]^T 里是 R 的哪一项？",
            _FACTORS, name, "A2",
        ), 7, lead, tail))
    total = sp.simplify(matrices[2] * matrices[1] * matrices[0])
    for number, matrix in enumerate((matrices[2], matrices[1], matrices[0]), start=1):
        lead = (
            "在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R1的展开式为"
            if number == 1 else f"R{number}的展开式为"
        )
        questions.append(_placed(
            _write_matrix(f"总矩阵前 R{number} 的展开式。", matrix),
            8, lead, "。",
        ))
    questions.append(_placed(
        _write_matrix("写出 R1*R2*R3 符号化简后的总矩阵 R_t。", total),
        8, "写出R1*R2*R3符号化简后的总矩阵R_t，R_t=", "。",
    ))
    questions.append(_placed(
        _write_matrix("写出从机体系向风轴系的转换矩阵 R_t'。", sp.simplify(total.T)),
        9, "因此从机体系向风轴系的转换矩阵R_t'为", "。",
    ))
    if len(questions) != 49:
        raise RuntimeError(f"风轴系向机体系过程应为 49 空，实际 {len(questions)}。")
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
    title = f"相机{index}"
    frame = f"相机{index}系"
    questions: list[dict] = []
    wind_basis = cm.basis(scene.wind, wind_names)
    cam_basis = cm.basis(cam, names)
    rp1 = sp.simplify(cam_basis.T * wind_basis)
    axis_options = _signed(wind_names + names)
    role_options = [
        f"{frame}绝对轴向", f"{frame}绝对法向", f"{frame}绝对侧向",
        "风轴绝对轴向", "风轴绝对法向", "风轴绝对侧向",
    ]
    role_gold = {"pan": f"{frame}绝对法向", "tilt": f"{frame}绝对侧向", "roll": f"{frame}绝对轴向"}
    paragraph_base = 0 if index == 1 else 6

    def put(item, paragraph, lead, tail=""):
        questions.append(_placed(item, paragraph, lead, tail))

    put(_choice(
        f"首先确定风轴系到{frame}的转换。转换方向是？",
        ["相机1系向风轴系", "风轴系向相机1系", "风轴系向机体系", "机体系向风轴系"] if index == 1
        else ["相机2系向风轴系", "风轴系向相机2系", "风轴系向机体系", "机体系向风轴系"],
        f"风轴系向{frame}", "C2",
    ), 1 + paragraph_base, f"首先确定风轴系到{frame}的转换矩阵。转换方向是", "。")
    put(_choice("所有坐标系的变换采用主动还是被动？", ["主动", "被动"], "被动", "C2"),
        1 + paragraph_base, "所有坐标系的变换采用", "，")
    put(_choice("该变换视为动轴还是静轴？", ["动轴", "静轴"], "动轴", "C2"),
        1 + paragraph_base, "视为", "，")
    put(_choice("旋转使用外旋还是内旋？", ["外旋", "内旋"], "内旋", "C2"),
        1 + paragraph_base, "使用", "。")
    cam_leads = [
        f"由于{frame}的绝对轴向正方向（前方）、绝对法向正方向（上方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    wind_leads = [
        "。风轴系的绝对轴向正方向（前方）、绝对法向正方向（上方）、绝对侧向正方向（右侧）分别对应",
        "、", "、",
    ]
    for lead, direction in zip(cam_leads, ("forward", "up", "right")):
        put(_choice(f"{frame}的绝对{ {'forward': '轴向', 'up': '法向', 'right': '侧向'}[direction] }对应哪个坐标轴？",
                    axis_options, _axis_pointing(cam, direction), "C1"), 1 + paragraph_base, lead)
    for lead, direction in zip(wind_leads, ("forward", "up", "right")):
        put(_choice(f"风轴系的绝对{ {'forward': '轴向', 'up': '法向', 'right': '侧向'}[direction] }对应哪个坐标轴？",
                    axis_options, _axis_pointing(scene.wind, direction), "C1"), 1 + paragraph_base, lead)
    put(_write_matrix(f"写出把风轴系坐标轴方向变到与{frame}相同的前置矩阵 RP1。", rp1),
        1 + paragraph_base, f"。首先使用一个前置矩阵RP1将风轴系的坐标轴方向转换到与{frame}相同，形成{frame}平正状态的中间坐标系。RP1为")
    euler_leads = [
        "。基于该中间相机坐标系，姿态转换与一般飞行力学中的欧拉角姿态变化原理相同。欧拉角顺序第一个是",
        "，第二个是",
        "，第三个是",
    ]
    for lead, name in zip(euler_leads, CAM_EULER):
        tail = "。" if name == "滚转角" else ""
        put(_choice(f"{title} 欧拉角顺序中的这一项是？", CAM_EULER, name, "A1"), 1 + paragraph_base, lead, tail)

    specs = (
        ("pan", "方位角", symbols[0], angles[0], "axial", 2 + paragraph_base),
        ("tilt", "俯仰角", symbols[1], angles[1], "axial", 3 + paragraph_base),
        ("roll", "滚转角", symbols[2], angles[2], "lateral", 4 + paragraph_base),
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
        subject = SUBJECT[key]
        put(_choice(f"当{title}{angle_name} {symbol_name}>0 时，{subject}向风轴的哪一方偏？",
                    SIX, side, "A1"),
            paragraph, f"当{title}{angle_name} {symbol_name}>0 时，{subject}向风轴的", "偏。")
        put(_choice(f"{title}{angle_name} {symbol_name} 相当于绕哪个绝对方向轴旋转？",
                    role_options, role_gold[key], "C2"),
            paragraph, "这相当于绕", "旋转。")
        phrase, direction = _ROLE_POINTING[role]
        put(_choice(f"{frame}的绝对{phrase}对应哪个坐标轴？",
                    axis_options, _axis_pointing(cam, direction), "C1"),
            paragraph, f"{frame}的绝对{phrase}对应", "。")
        put(_choice(f"{title}{angle_name} {symbol_name} 绕{frame}的哪个坐标轴旋转？",
                    _nine_axes(), axis, "C1"),
            paragraph, f"绕{frame}的坐标轴", "，")
        put(_choice("绕该轴是正转还是反转？", THREE_SIGN, _bias(sign), "A1"),
            paragraph, "取", "转。")
        put(_write_matrix(f"写出{title}{angle_name} {symbol_name} 对应的旋转矩阵。", matrix),
            paragraph, f"对应的旋转矩阵R_{symbol_name}为", "。")

    factors = [f"R_{symbols[0]}", f"R_{symbols[1]}", f"R_{symbols[2]}", "RP1"]
    # Intrinsic: first angle is rightmost; RP1 is applied before the angles.
    ordered = [factors[2], factors[1], factors[0], "RP1"]
    factor_leads = [
        f"从风轴系到{frame}的坐标变换矩阵为R_cw{index}=R1·R2·R3·R4。其中R1是",
        "。R2是", "。R3是", "。R4是",
    ]
    for lead, name in zip(factor_leads, ordered):
        tail = "。" if name == "RP1" else ""
        put(_choice(f"R_cw{index}=R1·R2·R3·R4 中的这一项是？", factors, name, "A2"),
            5 + paragraph_base, lead, tail)
    total = sp.simplify(matrices["roll"] * matrices["tilt"] * matrices["pan"] * rp1)
    for number, matrix in enumerate((matrices["roll"], matrices["tilt"], matrices["pan"], rp1), start=1):
        lead = (
            "在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R1的展开式为"
            if number == 1 else f"R{number}的展开式为"
        )
        put(_write_matrix(f"总矩阵前 R{number} 的展开式。", matrix), 5 + paragraph_base, lead, "。")
    put(_write_matrix(
        f"根据 ψ{index}、θ{index}、φ{index} 和当前坐标轴定义，推导风轴系到{frame}的旋转矩阵。给出 R_cw{index}。",
        total),
        5 + paragraph_base, f"给出风轴系到{frame}的旋转矩阵 R_cw{index}，R_cw{index}=", "。")
    center = C1 if index == 1 else C2
    t_cw = sp.simplify(-total * center)
    correct = f"t_cw{index} = −R_cw{index}·C{index}"
    put(_choice(
        f"相机{index} 的位姿平移向量 t_cw{index} 的公式形式是？",
        [
            f"t_cw{index} = R_cw{index}·C{index}",
            f"t_cw{index} = −R_cw{index}^T·C{index}",
            correct,
            f"t_cw{index} = C{index} − R_cw{index}",
        ],
        correct, "A2"),
        6 + paragraph_base, f"相机{index} 的位姿平移向量 t_cw{index} 的公式形式是", "。")
    put(_write_vector(f"写出相机{index} 的位姿平移向量 t_cw{index}。", t_cw),
        6 + paragraph_base, f"写出相机{index} 的位姿平移向量t_cw{index}，t_cw{index}=", "。")
    if len(questions) != 43:
        raise RuntimeError(f"{frame}过程应为 43 空，实际 {len(questions)}。")
    return questions, total


def build(spec: dict) -> dict:
    scene = cm.Scene().declare(spec)
    wind_x = scene.wind["Xw"]
    if cm.role_of(wind_x) != "axial" or DIR_NAME[wind_x] != "forward":
        raise RuntimeError("世界坐标系的 Xw 必须沿轴向指向前方。")
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

    def _line_symbols(c: str) -> str:
        return (
            f"l 是相机{c}图像中的直线。"
            f"K{c} 是相机{c}内参。"
            f"S{c} 把相机{c}坐标变到侧向正方向（右）、法向正方向（上）、轴向正方向（前）这个顺序，含取反。"
            f"(K{c}·S{c})^(-T) 是有效内参 K{c}·S{c} 的逆转置。"
            f"[P_c{c}]_× 是机头在相机{c}系中的坐标 P_c{c} 的反对称矩阵。"
            f"R_cw{c} 是风轴系到相机{c}系的旋转矩阵。"
            f"d_w 是这条空间直线在风轴系中的方向向量。"
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
        return _choice(
            _line_symbols(c) + f"相机{c}图像中的直线 l 应满足的公式是？",
            options, correct, "A2")

    def _nose_formula() -> dict:
        correct = "p_nose = L·RP1^T·R_t'·e"
        options = [
            "p_nose = L·R_t'·e",
            correct,
            "p_nose = RP1^T·R_t'·e",
            "p_nose = L·[1, 0, 0]^T",
        ]
        return _choice(
            "e 为机体纵轴在机体系的单位向量。机头在风轴系中的坐标 p_nose 的公式是？",
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
        return _choice(f"机头在相机{c} 系中的坐标 P_c{c} 的公式形式是？", options, correct, "A2")

    _put(_line_formula("1"), 1, _line_symbols("1") + "相机1图像中的直线 l 应满足的公式是", "。")
    _put(_line_formula("2"), 1, _line_symbols("2") + "相机2图像中的直线 l 应满足的公式是", "。")
    _put(_write_matrix(
        "给出直线公式中 (K1·S1)^(-T) 的展开式。不要与 [P_c1]_×、R_cw1、d_w 相乘。", k1it),
         2, "直线公式中的 (K1·S1)^(-T) 单独展开，不要与其他符号相乘。(K1·S1)^(-T)=", "。")
    _put(_write_matrix(
        "给出直线公式中 (K2·S2)^(-T) 的展开式。不要与 [P_c2]_×、R_cw2、d_w 相乘。", k2it),
         2, "直线公式中的 (K2·S2)^(-T) 单独展开，不要与其他符号相乘。(K2·S2)^(-T)=", "。")
    _put(_nose_formula(), 3, "e 为机体纵轴在机体系的单位向量。机头在风轴系中的坐标 p_nose 的公式是", "。")
    _put(_write_vector("写出机头在风轴系中的坐标 p_nose。", nose),
         3, "写出机头在风轴系中的坐标 p_nose，p_nose=", "。")
    _put(_point_formula("1"), 4, "机头在相机1 系中的坐标 P_c1 的公式形式是", "。")
    _put(_write_vector("写出机头在相机1 系中的坐标 P_c1。", p_c1),
         4, "写出机头在相机1 系中的坐标 P_c1，P_c1=", "。")
    _put(_point_formula("2"), 4, "机头在相机2 系中的坐标 P_c2 的公式形式是", "。")
    _put(_write_vector("写出机头在相机2 系中的坐标 P_c2。", p_c2),
         4, "写出机头在相机2 系中的坐标 P_c2，P_c2=", "。")
    _put(_write_matrix(
        "给出直线公式中 [P_c1]_× 的展开式。不要与 (K1·S1)^(-T)、R_cw1、d_w 相乘。", skew1),
         5, "直线公式中的 [P_c1]_× 单独展开，不要与其他符号相乘。[P_c1]_×=", "。")
    _put(_write_matrix(
        "给出直线公式中 [P_c2]_× 的展开式。不要与 (K2·S2)^(-T)、R_cw2、d_w 相乘。", skew2),
         5, "直线公式中的 [P_c2]_× 单独展开，不要与其他符号相乘。[P_c2]_×=", "。")

    _put(_write_vector(
        "空速在风轴系中的方向向量d_wV。"
        "空速在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV。写出 d_wV。",
        air),
        1,
        "空速在风轴系中的方向向量d_wV=",
        "。空速在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV。")
    _put(_write_vector(
        "机体纵轴在机体系的单位向量为 e。它在风轴系中的方向向量为 d_wL。"
        "机体纵轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL。写出 d_wL。",
        bx),
        1,
        "机体纵轴在机体系的单位向量为 e。它在风轴系中的方向向量为 d_wL = ",
        "。机体纵轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL。")
    _put(_write_vector(
        "机体法向轴在机体系的单位向量为 e。它在风轴系中的方向向量为d_wN。"
        "机体法向轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN。写出 d_wN。",
        bz),
        1,
        "机体法向轴在机体系的单位向量为 e。它在风轴系中的方向向量为d_wN=",
        "。机体法向轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN。")

    r21 = sp.simplify(r_cw2 * r_cw1.T)
    t21 = sp.simplify(t_cw2 - r21 * t_cw1)
    _put(_write_matrix("总矩阵前 R_cw2 的展开式。", r_cw2),
         1, "在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R_cw2的展开式为", "。")
    _put(_write_matrix("总矩阵前 R_cw1 的展开式。", r_cw1),
         1, "R_cw1的展开式为", "。")
    _put(_write_matrix(
        "根据 C1=(x1,y1,z1)、C2=(x2,y2,z2) 和姿态角 ψ1、θ1、φ1、ψ2、θ2、φ2，"
        "推导相机1系到相机2系的旋转 R21。给出 R21。",
        r21),
         1,
         "相机1、相机2的光心在风轴系中的坐标分别为 C1=(x1,y1,z1)、C2=(x2,y2,z2)，"
         "姿态角为 ψ1、θ1、φ1 与 ψ2、θ2、φ2。"
         "现在需要推导相机1系到相机2系的旋转 R21 和平移 t21。其中 R21=",
         "")
    _put(_write_vector(
        "根据 C1=(x1,y1,z1)、C2=(x2,y2,z2) 和姿态角 ψ1、θ1、φ1、ψ2、θ2、φ2，"
        "推导相机1系到相机2系的平移 t21。给出 t21。",
        t21),
         1, "，t21=", "。")
    _put(_choice("本质矩阵 E 的表达式是？",
                 ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
                 "E = skew(t21)*R21", "A2"),
         1, "本质矩阵 E 的表达式是", "。")
    _put(_choice("基础矩阵 F 的表达式是？",
                 ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)", "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
                 "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2"),
         2, "基础矩阵 F 的表达式是", "。")
    _put(_choice("极线约束方程是？",
                 ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
                  "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
                 "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2"),
         3, "极线约束方程是", "。")
    # —— 激波点反解：双相机三角化（公式建立）——
    inv1 = sp.simplify((K1 * s1).inv())
    inv2 = sp.simplify((K2 * s2).inv())
    d1 = sp.simplify(inv1 * sp.Matrix([u1, v1, 1]))
    d2 = sp.simplify(inv2 * sp.Matrix([u2, v2, 1]))

    _put(_write_vector(
        "相机1、相机2图像上激波点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
        "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
        "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
        "现在需要给出这两个射线方向各自的展开式,写成向量形式。写出 d1。",
        d1),
         4,
         "相机1、相机2图像上激波点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
         "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
         "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
         "现在需要给出这两个射线方向各自的展开式,写成向量形式。其中 d1=",
         "")
    _put(_write_vector(
        "相机1、相机2图像上激波点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
        "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
        "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
        "现在需要给出这两个射线方向各自的展开式,写成向量形式。写出 d2。",
        d2),
         4, "，d2=", "。")
    _put(_choice("激波点在相机1系中的坐标沿射线方向，X_c1 = ?",
                 ["X_c1 = λ1·d1", "X_c1 = d1 + λ1", "X_c1 = λ1 + d1", "X_c1 = d1 − λ1"],
                 "X_c1 = λ1·d1", "A2"),
         5, "激波点在相机1系中的坐标 X_c1 沿射线方向 d1，其中 λ1 为深度。X_c1=", "。")
    _put(_choice("两个相机看到同一激波点，约束方程为 ?",
                 ["λ2·d2 = R21·λ1·d1 + t21", "λ2·d2 = R21·λ1·d1 − t21",
                  "λ2·d2 = R21^T·λ1·d1 + t21", "λ1·d1 = R21·λ2·d2 + t21"],
                 "λ2·d2 = R21·λ1·d1 + t21", "A2"),
         6, "两个相机看到同一激波点，相机1系到相机2系的变换为 X_c2=R21·X_c1+t21，故约束方程为", "。")
    _put(_choice("在约束方程两侧叉乘哪个向量可消去 λ2 ?",
                 ["d2", "d1", "R21·d1", "t21"],
                 "d2", "A2"),
         7, "在约束方程两侧叉乘", "，消去λ2，得 d2×(λ1·R21·d1+t21)=0。")
    _put(_choice(
        "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
        "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1。λ1 = ?",
        ["λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²",
         "λ1 = −(d2×t21) / (d2×R21·d1)",
         "λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²"],
        "λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²", "A2"),
         8,
         "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
         "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1，λ1=",
         "。")
    _put(_choice("激波点在相机1系中的坐标 X_c1 已知，它在风轴系中的坐标 X_w = ?",
                 ["X_w = R_cw1^T·(X_c1 − t_cw1)", "X_w = R_cw1·X_c1 + t_cw1",
                  "X_w = R_cw1^T·X_c1 − t_cw1", "X_w = R_cw1·(X_c1 − t_cw1)"],
                 "X_w = R_cw1^T·(X_c1 − t_cw1)", "A2"),
         9, "激波点在相机1系中的坐标 X_c1 已知，则它在风轴系中的坐标 X_w =", "。")
    _put(_choice("激波点在风轴系中的坐标 X_w 已知，它在机体系中的坐标 X_b = ?",
                 ["X_b = R_t·RP1·(X_w − p_nose)", "X_b = RP1^T·R_t'·(X_w − p_nose)",
                  "X_b = R_t·RP1·(X_w + p_nose)", "X_b = R_t'·RP1·(X_w − p_nose)"],
                 "X_b = R_t·RP1·(X_w − p_nose)", "A2"),
         10, "激波点在风轴系中的坐标 X_w 已知，则它在机体系中的坐标 X_b =", "。")

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
        item["process"] = "计算机视觉双目姿态估计"
        item["segment"] = _segment_of(index)
        item["format_code"] = item.pop("format")
    return {
        "spec": spec,
        "declaration": _declaration(spec, scene),
        "questions": questions,
    }


def _declaration(spec: dict, scene: cm.Scene) -> str:
    chirality = {"right": "右", "left": "左"}
    role_cn = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}

    def frame_block(title: str, names: tuple, frame: dict, frame_spec: dict, extra: str = "") -> str:
        lines = [f"{title}（{chirality[frame_spec.get('chirality', 'left')]}手系）" + (f"；{extra}" if extra else "") + "："]
        for name in names:
            vector = frame[name]
            role = cm.role_of(vector)
            pointing = DIR_CN[DIR_NAME[vector]]
            inferred = "声明给出" if name in frame_spec else "由手性推断"
            lines.append(f"- {name}：沿{role_cn[role]}，指向{pointing}（{inferred}）。")
        return "\n".join(lines)

    wind_block = frame_block("风轴系", ("Xw", "Yw", "Zw"), scene.wind, spec["wind"])
    body_block = frame_block("机体系", ("Xb", "Yb", "Zb"), scene.body, spec["body"], "机体原点在机头")
    cam1_block = frame_block("相机1系", ("X1", "Y1", "Z1"), scene.cam1, spec["camera1"])
    cam2_block = frame_block("相机2系", ("X2", "Y2", "Z2"), scene.cam2, spec["camera2"])

    def _camera_attitude_lines() -> list[str]:
        """相机方位、俯仰、滚转与机体偏航、俯仰、滚转对应。正负仍由该样本的相机角设定。"""
        pitch = "正" if scene.angle_toward["tilt"] == "wind_up" else "负"
        yaw = "正" if scene.angle_toward["pan"] == "wind_right" else "负"
        roll = "正" if scene.angle_toward["roll"] == "camera_down" else "负"
        return [
            "相机俯仰角 θ1、θ2：相机纵轴与风轴水平面的夹角，"
            f"相机纵轴抬起时（相机纵轴向风轴的上方偏时）为{pitch}。",
            "相机方位角 ψ1、ψ2：相机纵轴在风轴水平面上的投影，与风轴系中Xw的夹角。"
            f"在Xw的右侧为{yaw}。",
            "相机滚转角 φ1、φ2：相机法向与通过相机纵轴的铅垂面的夹角，"
            f"向相机的右侧滚转时为{roll}。",
            "两个相机的欧拉姿态变化与传统变化顺序一致，没有机构的约束。"
            "相机1系的方位角、俯仰角、滚转角记为 ψ1、θ1、φ1，相机2系记为 ψ2、θ2、φ2。",
        ]

    return "\n".join([
        "【定义声明】",
        "",
        "固定约定",
        "- 坐标系转换一律采用被动变换。姿态旋转一律为动轴内旋。",
        "- 相机成像采用针孔模型 x = (K·S)·[R|t]·X。图像直线 l = (K·S)^(-T)·n，n 为光心与空间直线张成平面的法向。",
        "- 极线约束 x2^T·F·x1 = 0，F = (K2·S2)^(-T)·E·(K1·S1)^(-1)，E = [t21]_×·R21。",
        "- 风轴系与世界坐标系等价，以下一律在风轴系中描述。",
        "- 体轴系原点在机头；模型顶点到支杆底座的距离为 L。",
        "- 所有量均为符号：L、α_M、β_M、γ_M、ψ1、θ1、φ1、ψ2、θ2、φ2、相机光心在风轴系中的坐标为： 相机1系C1=(x1,y1,z1)、相机2系C2=(x2,y2,z2)。",
        "- 相机内参 K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]]，K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]。K 按 [侧向正方向（右）, 法向正方向（上）, 轴向正方向（前）] 作用。S 把该相机坐标变到这个顺序，含取反。有效内参为 K·S。",
        "",
        "绝对方向",
        "风轴、机体、相机各有三根绝对轴。三套绝对轴的正方向定义相同：轴向正方向是前方，法向正方向是上方，侧向正方向是右侧。六个绝对方向是这三根轴上的两个指向。",
        "下文凡写轴向正方向、法向正方向、侧向正方向，以及风洞轴向、机体轴向、相机轴向的正方向，都是这三根绝对轴的正方向，不是坐标轴的正方向。坐标轴声明只是指定每根坐标轴沿哪根绝对轴、指向哪一侧，可以与绝对正方向相同，也可以相反。",
        "三个系共用同一套指向关系：前方 × 右侧 = 下方。沿前方看去，右侧在右手侧，下方铅垂向下。这个关系不随定义改变。",
        "风轴的三轴六方向相对风洞：",
        "- 轴向：风洞轴线。正方向朝前，前方是来流。后方是其反方向。",
        "- 法向：风洞铅垂轴。正方向朝上，上方铅垂向上。下方铅垂向下。",
        "- 侧向：风洞水平横轴。正方向朝右。左侧是其反方向。",
        "机体的三轴六方向相对机体本身：",
        "- 轴向：连接机尾与机头。正方向朝前，指向机头。后方指向机尾。",
        "- 法向：机体的上下轴。正方向朝上。下方是其反方向。",
        "- 侧向：机体的左右轴。正方向朝右。左侧是其反方向。",
        "相机的三轴六方向相对相机：",
        "- 轴向：光轴。正方向朝前，指向拍摄方向。后方是其反方向。",
        "- 法向：像面纵向。正方向朝上。下方是其反方向。",
        "- 侧向：像面横向。正方向朝右。左侧是其反方向。",
        "",
        "场景",
        "初始位置时，飞机模型平正处于风洞中，模型前方朝向来流方向。两台已标定、同步曝光的相机从不同方位拍摄纹影图。相机光轴沿相机轴向，像面横向沿相机侧向，像面纵向沿相机法向。飞机尾部固定在了一个机械结构装置上，该装置会采取一种全新的欧拉角变换方式将机体变换到一种特殊姿态。变换的顺序与正常变换顺序不同，必须先理解机构角度的变换特性，然后推断出正确的角度变换顺序。这些角度的定义方式已在下方的角度定义给出。",
        "其中有两个特殊任务，其一是将风洞中的模型纵向轴线与法向轴线投影到相机图像中，确定来两轴线在图像坐标系中的表达式。其二是给定一个相机图像中一处激波发生位置的坐标。根据双目相机模型，反推激波发生位置在实际机器系中的坐标。其中涉及到的子任务包括，根据机体系与风轴系的姿态角，将机体的纵轴和法向轴转换到风轴系中表达；根据相机坐标系与风轴系的位置关系，推导风轴系向相机坐标系的转换矩阵，并据此进一步推导两个双目相机之间的本质矩阵和基本矩阵；根据风轴系汇总机体两轴的表达式与双目相机模型，计算图像坐标系中，两轴的表达式；根据激波在一个相机图像中的坐标位置，重构激波发生位置在机体坐标系中的坐标。",
        "",
        "角度",
        *_mechanism_definition(scene),
        *_camera_attitude_lines(),
        "",
        "坐标系",
        "坐标轴按地面系的方式建立。沿轴向、指向前方的那根坐标轴是参考轴，相当于地面坐标轴 Xg。另外两根坐标轴在法向与侧向之间绑定，绑定结果必须符合该系的手性。",
        wind_block,
        "",
        body_block,
        "",
        cam1_block,
        "",
        cam2_block,
        "",
        "物理常识",
        "轴向正方向、法向正方向、侧向正方向分别是前方、上方、右侧。单独变换迎角（俯仰）时，旋转轴是绝对侧向轴。单独变换侧滑角（偏航）时，旋转轴是绝对法向轴。单独变换滚转角时，旋转轴是绝对轴向轴。各角的正方向以定义声明为准。坐标系转换采用被动变换。姿态角采用动轴内旋。",
    ])


def generate(spec: dict) -> dict:
    return build(spec)


if __name__ == "__main__":
    for key in FRAME_NAMES:
        print(key, len(alternate_frames(key)))
    sample = build(STANDARD_SPEC)
    print("standard questions", len(sample["questions"]))
