# -*- coding: utf-8 -*-
"""Contract atoms on the answer dependency graph.

The evaluation graph has two kinds of nodes only: contract instances C and
the 194 public answers. Edges are either C→answer direct effects, or the
existing 388 answer→answer derivation edges. A binding rule records which
contracts jointly determine which fields of an answer. It is metadata, not a
node, and it does not enter depends_on, depth, or error propagation.

A declaration that is read again is a new instance. Downstream answers are
descendants in the answer graph. Deleted public ids 52–54 and 101–103 are
not nodes. Later ids do not move.

Indirect contract relations are derived from the answer edges and stored on
each answer as contract_relations. They are not written into depends_on.
That field is a potential dependency: actual_error_path stays null until a
run is joined, and a correct intermediate blocks the error path.
"""

from __future__ import annotations

from functools import lru_cache


RELATION_TYPES = ("direct_answer", "direct_method", "scope_constraint", "downstream")
AUTHORED_RELATIONS = ("direct_answer", "direct_method", "scope_constraint")

PROCESSES = {
    "wind_to_body": "风轴系到体轴系",
    "balance_to_body": "天平系到体轴系",
    "balance_to_wind": "天平系到风轴系",
    "reference_point": "力矩参考点",
}

# Closure only. These stay out of authored contract targets.
_NOT_AUTHORED = frozenset({
    "52", "53", "54", "101", "102", "103",
    "55", "59", "60", "61", "80", "81",
})

_GLOBALS = (
    ("C_CONV_WIND_TO_BODY", "conv", "风轴到体轴的转换方向", "wind_to_body", ""),
    ("C_CONV_BALANCE_TO_BODY", "conv", "天平到体轴的转换方向", "balance_to_body", ""),
    ("C_CONV_BALANCE_TO_WIND", "conv", "天平到风轴的转换方向", "balance_to_wind", ""),
    ("C_CONV_PASSIVE", "conv", "坐标变换用被动形式", "passive", ""),
    ("C_CONV_INTRINSIC", "conv", "姿态旋转用动轴内旋", "intrinsic", ""),
    ("C_CONV_MOVING_FRAME", "conv", "体轴系是动轴", "moving", ""),
    ("C_AXIS_WIND_X", "axis", "风轴 X 沿风洞轴向指向前方", "tunnel_forward", "tunnel_lateral"),
    ("C_AXIS_WIND_Y", "axis", "风轴 Y 沿风洞法向指向上方", "tunnel_up", "tunnel_lateral"),
    ("C_AXIS_WIND_HANDEDNESS", "axis", "风轴系手性", "right_handed", "left_handed"),
    ("C_AXIS_BODY_X", "axis", "体轴 XTZ 沿机体轴向指向前方", "body_forward", "body_aft"),
    ("C_AXIS_BODY_Y", "axis", "体轴 YTZ 沿机体法向指向上方", "body_up", "body_down"),
    ("C_AXIS_BODY_HANDEDNESS", "axis", "体轴系手性", "right_handed", "left_handed"),
    ("C_AXIS_BALANCE_X", "axis", "天平 XTP 沿天平轴向指向前方", "balance_forward", "balance_aft"),
    ("C_AXIS_BALANCE_Y", "axis", "天平 YTP 沿天平法向指向上方", "balance_up", "balance_down"),
    ("C_AXIS_BALANCE_HANDEDNESS", "axis", "天平系手性", "right_handed", "left_handed"),
    ("C_AXIS_ALPHA", "axis", "机构迎角绕绝对侧向", "absolute_lateral", ""),
    ("C_AXIS_BETA", "axis", "机构侧滑绕绝对法向", "absolute_normal", ""),
    ("C_AXIS_GAMMA", "axis", "机构滚转绕绝对轴向", "absolute_forward", ""),
    ("C_AXIS_GAMMA_TP", "axis", "天平安装角绕天平绝对轴向", "balance_axial", ""),
    ("C_AXIS_ELASTIC_ALPHA", "axis", "弹性迎角的旋转轴", "balance_lateral", ""),
    ("C_AXIS_ELASTIC_BETA", "axis", "弹性侧滑的旋转轴", "balance_normal", ""),
    ("C_AXIS_ELASTIC_GAMMA", "axis", "弹性滚转的旋转轴", "balance_axial", ""),
    ("C_SIGN_ALPHA_M", "sign", "机构迎角正方向", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_ALPHA", "sign", "理论迎角正方向", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_BETA_M", "sign", "机构侧滑正方向", "airspeed_body_right", "airspeed_body_left"),
    ("C_SIGN_BETA", "sign", "理论侧滑正方向", "airspeed_body_right", "airspeed_body_left"),
    ("C_SIGN_GAMMA_M", "sign", "机构滚转正方向", "right_wing_down", "right_wing_up"),
    ("C_SIGN_GAMMA", "sign", "理论滚转正方向", "right_wing_down", "right_wing_up"),
    ("C_SIGN_GAMMA_TP", "sign", "天平安装角正方向", "right_hand_about_balance_forward", "opposite"),
    ("C_SIGN_ELASTIC_ALPHA", "sign", "弹性迎角正方向", "balance_up", "balance_down"),
    ("C_SIGN_ELASTIC_BETA", "sign", "弹性侧滑正方向", "balance_left", "balance_right"),
    ("C_SIGN_ELASTIC_GAMMA", "sign", "弹性滚转正方向", "clockwise_looking_forward", "counterclockwise"),
    ("C_SIGN_FLOW_ALPHA", "sign", "气流偏角迎角正方向", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_FLOW_BETA", "sign", "气流偏角侧滑正方向", "airspeed_body_right", "airspeed_body_left"),
    ("C_INSTALL_MODEL", "install", "模型安装正装或反装", "upright", "inverted"),
    ("C_METHOD_EULER_ORDER", "method", "机构角顺序", "alpha_beta_gamma", "beta_alpha_gamma"),
    ("C_METHOD_PREPOSE_SLOTS", "method", "前置矩阵放在连乘式的 R4、R5", "R4_RP2_R5_RP1", ""),
    ("C_METHOD_LOAD_PRODUCT_ORDER", "method", "载荷连乘顺序", "R1_RTP_TZ2_R2_RTP_TZ1_R3_roll", ""),
    ("C_METHOD_WIND_PRODUCT_ORDER", "method", "天平到风轴的连乘顺序", "R1_RP_TP_FD_R2_R_gamma_tp", ""),
    ("C_METHOD_ROLL_DIFFERENCE", "method", "天平系与机体系的滚转角差", "gamma_m_minus_gamma_tp", ""),
    ("C_METHOD_ALPHA_INVERSE", "method", "理论迎角的反三角函数", "arctan", ""),
    ("C_METHOD_BETA_INVERSE", "method", "理论侧滑的反三角函数", "arcsin", ""),
    ("C_METHOD_BETA_UNIT_DENOMINATOR", "method", "理论侧滑的分母取正的 1", "positive_one", ""),
    ("C_METHOD_DELTA_ALPHA_TEMPLATE", "method", "弹性迎角增量写成正分子分母的分式", "positive_quotient", ""),
    ("C_METHOD_DELTA_BETA_TEMPLATE", "method", "弹性侧滑增量写成正分子分母的分式", "positive_quotient", ""),
    ("C_METHOD_SMALL_ANGLE", "method", "弹性变形用小角度近似", "sin_theta_eq_theta", ""),
    ("C_METHOD_TORQUE_TEMPLATE", "method", "等效力矩用 C·L_c/L_RT", "coefficient_dot_arm_over_L_RT", ""),
    ("C_FORCE_AXIS", "force", "力系数对应绝对轴", "CN_normal_CA_axial_CZ_lateral", ""),
    ("C_MOMENT_AXIS", "force", "力矩系数对应绝对轴", "MN_normal_MA_axial_MZ_lateral", ""),
    ("C_GEOM_MODEL_LENGTH", "geom", "模型顶点到模型底部 L", "L", ""),
    ("C_GEOM_SUPPORT_LENGTH", "geom", "模型底部到支杆底部 L_t", "L_t", ""),
    ("C_GEOM_BALANCE_LENGTH", "geom", "天平校心到支杆底部 L_r", "L_r", ""),
    ("C_GEOM_ORIGIN", "geom", "体轴系原点在天平校心", "balance_center", ""),
    ("C_GEOM_ENGINEERING_AXIAL", "geom", "工程轴向从顶点向后量", "rearward_from_nose", ""),
    ("C_TORQUE_SOURCE", "geom", "力矩等效的起点", "origin", ""),
    ("C_TORQUE_TARGET", "geom", "力矩等效的终点", "reference_point", ""),
    ("C_FORCE_APPLICATION", "geom", "产生额外力矩的力的位置", "origin", ""),
    ("C_TORQUE_EFFECT", "geom", "额外力矩的作用位置", "reference_point", ""),
    ("C_SCOPE_ALIGNMENT_TARGET", "scope", "对齐矩阵所作用的坐标系", "target_frame", ""),
    ("C_SCOPE_ALIGNMENT_SOURCE", "scope", "对齐矩阵所对照的坐标系", "source_frame", ""),
    ("C_SCOPE_ROTATION_FRAME", "scope", "姿态旋转写在中间体轴系", "body_frame", ""),
    ("C_SCOPE_LEVEL_FRAME", "scope", "平置后只研究平置机体系", "level_body", ""),
)

_GLOBAL = {item[0]: item for item in _GLOBALS}


def _iid(global_id: str, process: str, slot: str = "") -> str:
    if slot:
        return f"{global_id}@{process}:{slot}"
    return f"{global_id}@{process}"


def _split_instance(instance_id: str) -> tuple[str, str, str]:
    global_id, _, rest = instance_id.partition("@")
    process, _, slot = rest.partition(":")
    return global_id, process, slot


# global, process, slot, targets, relation, field
_RECOGNITION = (
    ("C_CONV_WIND_TO_BODY", "wind_to_body", "", ("1",), "direct_answer", "conversion_direction"),
    ("C_AXIS_WIND_Y", "wind_to_body", "", ("2",), "direct_answer", "axis_direction"),
    ("C_AXIS_WIND_X", "wind_to_body", "", ("3",), "direct_answer", "axis_direction"),
    ("C_AXIS_BODY_Y", "wind_to_body", "", ("5",), "direct_answer", "axis_direction"),
    ("C_AXIS_BODY_X", "wind_to_body", "", ("6",), "direct_answer", "axis_direction"),
    ("C_SCOPE_ALIGNMENT_TARGET", "wind_to_body", "rp1", ("9",), "scope_constraint", "coordinate_frame"),
    ("C_SCOPE_ALIGNMENT_SOURCE", "wind_to_body", "rp1", ("10",), "scope_constraint", "coordinate_frame"),
    ("C_INSTALL_MODEL", "wind_to_body", "mount", ("11", "12"), "direct_answer", "installation"),
    ("C_SCOPE_ALIGNMENT_TARGET", "wind_to_body", "rp2", ("14",), "scope_constraint", "coordinate_frame"),
    ("C_SCOPE_ALIGNMENT_SOURCE", "wind_to_body", "rp2", ("15",), "scope_constraint", "coordinate_frame"),
    ("C_METHOD_EULER_ORDER", "wind_to_body", "", ("16", "17", "18"), "direct_method", "rotation_order"),
    ("C_INSTALL_MODEL", "wind_to_body", "alpha", ("19",), "direct_answer", "installation"),
    ("C_SIGN_ALPHA", "wind_to_body", "", ("21",), "direct_answer", "positive_direction"),
    ("C_SCOPE_ROTATION_FRAME", "wind_to_body", "alpha", ("23",), "scope_constraint", "coordinate_frame"),
    ("C_INSTALL_MODEL", "wind_to_body", "beta", ("27",), "direct_answer", "installation"),
    ("C_SIGN_BETA", "wind_to_body", "", ("29",), "direct_answer", "positive_direction"),
    ("C_SCOPE_ROTATION_FRAME", "wind_to_body", "beta", ("31",), "scope_constraint", "coordinate_frame"),
    ("C_SIGN_GAMMA_M", "wind_to_body", "", ("35",), "direct_answer", "positive_direction"),
    ("C_SIGN_GAMMA", "wind_to_body", "", ("37",), "direct_answer", "positive_direction"),
    ("C_SCOPE_ROTATION_FRAME", "wind_to_body", "gamma", ("40",), "scope_constraint", "coordinate_frame"),
    ("C_CONV_PASSIVE", "wind_to_body", "", ("44",), "direct_method", "transform_type"),
    ("C_CONV_MOVING_FRAME", "wind_to_body", "", ("45",), "direct_method", "frame_motion"),
    ("C_CONV_INTRINSIC", "wind_to_body", "", ("46",), "direct_method", "rotation_type"),
    ("C_SCOPE_LEVEL_FRAME", "wind_to_body", "", ("57",), "scope_constraint", "coordinate_frame"),
    ("C_METHOD_ALPHA_INVERSE", "wind_to_body", "", ("69",), "direct_method", "inverse_function"),
    ("C_METHOD_BETA_INVERSE", "wind_to_body", "", ("75",), "direct_method", "inverse_function"),
    ("C_METHOD_BETA_UNIT_DENOMINATOR", "wind_to_body", "", ("78", "79"), "direct_method", "denominator"),
    ("C_CONV_BALANCE_TO_BODY", "balance_to_body", "", ("82",), "direct_answer", "conversion_direction"),
    ("C_SIGN_GAMMA_TP", "balance_to_body", "", ("83",), "direct_answer", "positive_direction"),
    ("C_SIGN_GAMMA_M", "balance_to_body", "", ("84",), "direct_answer", "positive_direction"),
    ("C_AXIS_BALANCE_Y", "balance_to_body", "", ("89",), "direct_answer", "axis_direction"),
    ("C_AXIS_BALANCE_X", "balance_to_body", "", ("90",), "direct_answer", "axis_direction"),
    ("C_AXIS_BODY_Y", "balance_to_body", "", ("92",), "direct_answer", "axis_direction"),
    ("C_AXIS_BODY_X", "balance_to_body", "", ("93",), "direct_answer", "axis_direction"),
    ("C_INSTALL_MODEL", "balance_to_body", "", ("96",), "direct_answer", "installation"),
    ("C_CONV_BALANCE_TO_WIND", "balance_to_wind", "", ("117",), "direct_answer", "conversion_direction"),
    ("C_AXIS_WIND_X", "balance_to_wind", "forward", ("118",), "direct_answer", "axis_direction"),
    ("C_AXIS_BALANCE_X", "balance_to_wind", "forward", ("119",), "direct_answer", "axis_direction"),
    ("C_AXIS_WIND_Y", "balance_to_wind", "axes", ("120",), "direct_answer", "axis_direction"),
    ("C_AXIS_WIND_X", "balance_to_wind", "axes", ("122",), "direct_answer", "axis_direction"),
    ("C_AXIS_BALANCE_Y", "balance_to_wind", "axes", ("123",), "direct_answer", "axis_direction"),
    ("C_AXIS_BALANCE_X", "balance_to_wind", "axes", ("125",), "direct_answer", "axis_direction"),
    ("C_SCOPE_ALIGNMENT_TARGET", "balance_to_wind", "rp", ("127",), "scope_constraint", "coordinate_frame"),
    ("C_SCOPE_ALIGNMENT_SOURCE", "balance_to_wind", "rp", ("128",), "scope_constraint", "coordinate_frame"),
    ("C_SIGN_ELASTIC_ALPHA", "balance_to_wind", "", ("136",), "direct_answer", "positive_direction"),
    ("C_SIGN_ELASTIC_BETA", "balance_to_wind", "", ("137",), "direct_answer", "positive_direction"),
    ("C_AXIS_ELASTIC_ALPHA", "balance_to_wind", "", ("139",), "direct_method", "rotation_axis"),
    ("C_AXIS_ELASTIC_BETA", "balance_to_wind", "", ("140",), "direct_method", "rotation_axis"),
    ("C_AXIS_ELASTIC_GAMMA", "balance_to_wind", "", ("141",), "direct_method", "rotation_axis"),
    ("C_INSTALL_MODEL", "balance_to_wind", "", ("151",), "direct_answer", "installation"),
    ("C_SIGN_ELASTIC_GAMMA", "balance_to_wind", "", ("154",), "direct_answer", "positive_direction"),
    ("C_SIGN_ALPHA", "balance_to_wind", "", ("155",), "direct_answer", "positive_direction"),
    ("C_SIGN_BETA", "balance_to_wind", "", ("156",), "direct_answer", "positive_direction"),
    ("C_SIGN_GAMMA", "balance_to_wind", "", ("157",), "direct_answer", "positive_direction"),
    ("C_SIGN_FLOW_ALPHA", "balance_to_wind", "", ("179",), "direct_answer", "positive_direction"),
    ("C_SIGN_FLOW_BETA", "balance_to_wind", "", ("180",), "direct_answer", "positive_direction"),
    ("C_GEOM_ORIGIN", "reference_point", "", ("181",), "direct_method", "origin"),
    ("C_TORQUE_SOURCE", "reference_point", "", ("185",), "direct_method", "moment_origin"),
    ("C_TORQUE_TARGET", "reference_point", "", ("186",), "direct_method", "moment_destination"),
    ("C_FORCE_APPLICATION", "reference_point", "", ("187",), "direct_method", "force_place"),
    ("C_TORQUE_EFFECT", "reference_point", "", ("188",), "direct_method", "moment_place"),
)


def _factor(
    factor_id: str,
    process: str,
    parents: tuple[tuple[str, str], ...],
    targets: tuple[str, ...],
    relation: str,
    fields: tuple[str, ...],
    kind: str,
    chain: str,
    note: str = "",
    standard: dict | None = None,
) -> dict:
    return {
        "id": factor_id,
        "process": process,
        "parents": parents,
        "targets": targets,
        "relation": relation,
        "fields": fields,
        "kind": kind,
        "chain": chain,
        "note": note,
        "standard": standard or {},
    }


_W = "wind_to_body"
_B = "balance_to_body"
_D = "balance_to_wind"
_R = "reference_point"

_FACTORS = (
    _factor(
        "F_WIND_Z", _W, (("C_AXIS_WIND_HANDEDNESS", ""), ("C_AXIS_WIND_X", ""), ("C_AXIS_WIND_Y", "")),
        ("4",), "direct_answer", ("axis_direction",), "axis_inference", "wind_axes",
        "侧向轴由风轴手性和已声明的 X、Y 推断。",
    ),
    _factor(
        "F_BODY_Z", _W, (("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", "")),
        ("7",), "direct_answer", ("axis_direction",), "axis_inference", "body_axes",
        "ZTZ 由体轴手性和已声明的 X、Y 推断，不是单独改写的原子。",
    ),
    _factor(
        "F_RP1", _W,
        (
            ("C_AXIS_WIND_X", ""), ("C_AXIS_WIND_Y", ""), ("C_AXIS_WIND_HANDEDNESS", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("8",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rp1",
        "轴声明到对齐矩阵的第一次应用。正装标准下可以是单位阵，单位阵仍是这些声明的结果。",
    ),
    _factor(
        "F_RP2", _W,
        (
            ("C_INSTALL_MODEL", "mount"),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("13",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rp2",
    ),
    _factor(
        "F_ALPHA_M_DIRECTION", _W, (("C_INSTALL_MODEL", "alpha"), ("C_SIGN_ALPHA_M", "")),
        ("20",), "direct_answer", ("airspeed_direction",), "direction", "alpha",
    ),
    _factor(
        "F_ALPHA_SIGN", _W, (("C_SIGN_ALPHA_M", ""), ("C_SIGN_ALPHA", "")),
        ("22",), "direct_answer", ("outer_sign",), "sign", "alpha",
    ),
    _factor(
        "F_ALPHA_AXIS", _W,
        (("C_AXIS_ALPHA", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", "")),
        ("24",), "direct_method", ("rotation_axis",), "axis_slot", "alpha",
    ),
    _factor(
        "F_ALPHA_TURN", _W,
        (
            ("C_SIGN_ALPHA_M", ""), ("C_AXIS_ALPHA", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("25",), "direct_answer", ("rotation_sense",), "sign", "alpha",
    ),
    _factor(
        "F_ALPHA_MATRIX", _W,
        (
            ("C_AXIS_ALPHA", ""), ("C_SIGN_ALPHA_M", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""),
            ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", ""),
        ),
        ("26",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "alpha",
        "被动和内旋规定矩阵写成哪种形式。当前生成器不单独扰动它们来改单轴元素；元素变化来自手性、转轴和正方向。",
    ),
    _factor(
        "F_BETA_M_DIRECTION", _W, (("C_INSTALL_MODEL", "beta"), ("C_SIGN_BETA_M", "")),
        ("28",), "direct_answer", ("airspeed_direction",), "direction", "beta",
    ),
    _factor(
        "F_BETA_SIGN", _W, (("C_SIGN_BETA_M", ""), ("C_SIGN_BETA", "")),
        ("30",), "direct_answer", ("outer_sign",), "sign", "beta",
    ),
    _factor(
        "F_BETA_AXIS", _W,
        (("C_AXIS_BETA", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", "")),
        ("32",), "direct_method", ("rotation_axis",), "axis_slot", "beta",
    ),
    _factor(
        "F_BETA_TURN", _W,
        (
            ("C_SIGN_BETA_M", ""), ("C_AXIS_BETA", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("33",), "direct_answer", ("rotation_sense",), "sign", "beta",
    ),
    _factor(
        "F_BETA_MATRIX", _W,
        (
            ("C_AXIS_BETA", ""), ("C_SIGN_BETA_M", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""),
            ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", ""),
        ),
        ("34",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "beta",
        "被动和内旋的说明与 F_ALPHA_MATRIX 相同。",
    ),
    _factor(
        "F_GAMMA_M_WING", _W, (("C_SIGN_GAMMA_M", ""), ("C_AXIS_BODY_X", "")),
        ("36",), "direct_answer", ("wing_direction",), "direction", "gamma",
    ),
    _factor(
        "F_GAMMA_WING", _W, (("C_SIGN_GAMMA", ""), ("C_AXIS_BODY_X", "")),
        ("38",), "direct_answer", ("wing_direction",), "direction", "gamma",
    ),
    _factor(
        "F_GAMMA_SIGN", _W, (("C_SIGN_GAMMA_M", ""), ("C_SIGN_GAMMA", "")),
        ("39",), "direct_answer", ("outer_sign",), "sign", "gamma",
    ),
    _factor(
        "F_GAMMA_AXIS", _W, (("C_AXIS_GAMMA", ""), ("C_AXIS_BODY_X", "")),
        ("41",), "direct_method", ("rotation_axis",), "axis_slot", "gamma",
    ),
    _factor(
        "F_GAMMA_TURN", _W, (("C_AXIS_GAMMA", ""), ("C_SIGN_GAMMA_M", ""), ("C_AXIS_BODY_X", "")),
        ("42",), "direct_answer", ("rotation_sense",), "sign", "gamma",
    ),
    _factor(
        "F_GAMMA_MATRIX", _W,
        (
            ("C_AXIS_GAMMA", ""), ("C_SIGN_GAMMA_M", ""), ("C_AXIS_BODY_X", ""),
            ("C_AXIS_BODY_HANDEDNESS", ""), ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", ""),
        ),
        ("43",), "direct_answer", ("diagonal_pattern", "upper_off_diagonal_sign", "lower_off_diagonal_sign"),
        "matrix", "gamma",
        "标准右手、绕 X 的被动矩阵是 [[1,0,0],[0,cos,sin],[0,-sin,cos]]。左手性翻转 sin。内旋不单独改这些元素，它进入连乘因子。",
        {"43": {"axis": "X", "upper_off_diagonal_sign": "+", "lower_off_diagonal_sign": "-"}},
    ),
    _factor(
        "F_ATTITUDE_PRODUCT", _W,
        (("C_METHOD_EULER_ORDER", ""), ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", "")),
        ("47", "48", "49"), "direct_method", ("matrix_order",), "product_order", "product",
        "只决定 R1、R2、R3。R4、R5 不读欧拉顺序。",
    ),
    _factor(
        "F_PREPOSE_PRODUCT", _W,
        (("C_METHOD_PREPOSE_SLOTS", ""), ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", "")),
        ("50", "51"), "direct_method", ("matrix_order",), "product_order", "product",
    ),
    _factor(
        "F_ROLL_DIFFERENCE", _B,
        (
            ("C_SIGN_GAMMA_TP", ""), ("C_SIGN_GAMMA_M", ""), ("C_METHOD_ROLL_DIFFERENCE", ""),
            ("C_AXIS_BALANCE_X", ""), ("C_AXIS_BODY_X", ""),
        ),
        ("85",), "direct_answer", ("angle_expression",), "formula", "roll",
    ),
    _factor(
        "F_ROLL_AXIS", _B, (("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_X", ""), ("C_AXIS_BALANCE_Y", ""), ("C_AXIS_BALANCE_HANDEDNESS", "")),
        ("86",), "direct_method", ("rotation_axis",), "axis_slot", "roll",
    ),
    _factor(
        "F_ROLL_TURN", _B,
        (("C_METHOD_ROLL_DIFFERENCE", ""), ("C_SIGN_GAMMA_TP", ""), ("C_SIGN_GAMMA_M", ""), ("C_AXIS_GAMMA_TP", "")),
        ("87",), "direct_answer", ("rotation_sense",), "sign", "roll",
    ),
    _factor(
        "F_ROLL_MATRIX", _B,
        (
            ("C_METHOD_ROLL_DIFFERENCE", ""), ("C_SIGN_GAMMA_TP", ""), ("C_SIGN_GAMMA_M", ""),
            ("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_HANDEDNESS", ""), ("C_CONV_PASSIVE", ""),
        ),
        ("88",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "roll",
        "局部安装角矩阵。总矩阵 ID104 另有连乘因子。",
    ),
    _factor(
        "F_BALANCE_Z", _B, (("C_AXIS_BALANCE_HANDEDNESS", ""), ("C_AXIS_BALANCE_X", ""), ("C_AXIS_BALANCE_Y", "")),
        ("91",), "direct_answer", ("axis_direction",), "axis_inference", "balance_axes",
    ),
    _factor(
        "F_BODY_Z_LOAD", _B, (("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", "")),
        ("94",), "direct_answer", ("axis_direction",), "axis_inference", "body_axes",
        "体轴指向在这个子过程里重新读取，不依赖风轴系到体轴系的 ID7。",
    ),
    _factor(
        "F_RTP_TZ1", _B,
        (
            ("C_AXIS_BALANCE_X", ""), ("C_AXIS_BALANCE_Y", ""), ("C_AXIS_BALANCE_HANDEDNESS", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("95",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rtp_tz1",
    ),
    _factor(
        "F_RTP_TZ2", _B,
        (("C_INSTALL_MODEL", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", "")),
        ("97",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rtp_tz2",
    ),
    _factor(
        "F_LOAD_PRODUCT", _B, (("C_METHOD_LOAD_PRODUCT_ORDER", ""), ("C_CONV_PASSIVE", "")),
        ("98", "99", "100"), "direct_method", ("matrix_order",), "product_order", "load_product",
        "标准顺序是 R1=RTP_TZ2，R2=RTP_TZ1，R3=安装滚转。",
    ),
    _factor(
        "F_LOAD_TOTAL", _B, (("C_METHOD_LOAD_PRODUCT_ORDER", ""), ("C_CONV_PASSIVE", "")),
        ("104",), "direct_answer", ("matrix_product",), "matrix", "load_total",
        "组合执行点。分矩阵是否正确看 ID88、ID95、ID97，不把那些契约再接到 ID104。",
    ),
    _factor(
        "F_LOAD_BINDING", _B, (("C_FORCE_AXIS", ""), ("C_MOMENT_AXIS", "")),
        ("105", "106", "107", "108", "109", "110"), "direct_method", ("component_name",),
        "semantic_binding", "load_names",
    ),
    _factor(
        "F_LOAD_COMPONENTS", _B,
        (("C_FORCE_AXIS", ""), ("C_MOMENT_AXIS", ""), ("C_METHOD_LOAD_PRODUCT_ORDER", "")),
        ("111", "112", "113", "114", "115", "116"), "direct_answer", ("component_expression",),
        "formula", "load_components",
    ),
    _factor(
        "F_WIND_Z_FD", _D,
        (("C_AXIS_WIND_HANDEDNESS", "axes"), ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes")),
        ("121",), "direct_answer", ("axis_direction",), "axis_inference", "wind_axes",
    ),
    _factor(
        "F_BALANCE_Z_FD", _D,
        (("C_AXIS_BALANCE_HANDEDNESS", "axes"), ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes")),
        ("124",), "direct_answer", ("axis_direction",), "axis_inference", "balance_axes",
    ),
    _factor(
        "F_RP_TP_FD", _D,
        (
            ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes"), ("C_AXIS_WIND_HANDEDNESS", "axes"),
            ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes"), ("C_AXIS_BALANCE_HANDEDNESS", "axes"),
        ),
        ("126",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rp_tp_fd",
    ),
    _factor(
        "F_GAMMA_TP_ABSOLUTE", _D, (("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes")),
        ("129",), "direct_method", ("absolute_axis",), "axis_slot", "gamma_tp",
    ),
    _factor(
        "F_GAMMA_TP_AXIS", _D,
        (("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes"), ("C_AXIS_BALANCE_HANDEDNESS", "axes")),
        ("130",), "direct_method", ("rotation_axis",), "axis_slot", "gamma_tp",
    ),
    _factor(
        "F_GAMMA_TP_TURN", _D, (("C_SIGN_GAMMA_TP", ""), ("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_X", "forward")),
        ("131",), "direct_answer", ("rotation_sense",), "sign", "gamma_tp",
    ),
    _factor(
        "F_GAMMA_TP_MATRIX", _D,
        (
            ("C_SIGN_GAMMA_TP", ""), ("C_AXIS_GAMMA_TP", ""), ("C_AXIS_BALANCE_HANDEDNESS", "axes"),
            ("C_CONV_PASSIVE", ""),
        ),
        ("132",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "gamma_tp",
    ),
    _factor(
        "F_WIND_PRODUCT", _D, (("C_METHOD_WIND_PRODUCT_ORDER", ""), ("C_CONV_PASSIVE", "")),
        ("133", "134"), "direct_method", ("matrix_order",), "product_order", "wind_product",
        "标准顺序是 R1=RP_TP_FD，R2=R_γ_tp。",
    ),
    _factor(
        "F_BALANCE_FORWARD", _D,
        (("C_AXIS_BALANCE_X", "forward"), ("C_AXIS_BALANCE_Y", "axes"), ("C_AXIS_BALANCE_HANDEDNESS", "axes")),
        ("135",), "direct_answer", ("vector_component",), "vector", "elastic_vector",
    ),
    _factor(
        "F_ELASTIC_GAMMA_CLOCK", _D, (("C_SIGN_ELASTIC_GAMMA", ""), ("C_AXIS_BALANCE_HANDEDNESS", "axes")),
        ("138",), "direct_answer", ("clock_direction",), "direction", "elastic",
    ),
    _factor(
        "F_ELASTIC_COMPONENT_SIGNS", _D,
        (
            ("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ELASTIC_BETA", ""), ("C_METHOD_SMALL_ANGLE", ""),
            ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes"), ("C_AXIS_BALANCE_HANDEDNESS", "axes"),
        ),
        ("142", "143", "144"), "direct_answer", ("component_sign",), "sign", "elastic_vector",
    ),
    _factor(
        "F_ELASTIC_AXIS_VECTOR", _D,
        (
            ("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ELASTIC_BETA", ""), ("C_METHOD_SMALL_ANGLE", ""),
            ("C_AXIS_BALANCE_X", "axes"), ("C_AXIS_BALANCE_Y", "axes"), ("C_AXIS_BALANCE_HANDEDNESS", "axes"),
        ),
        ("145",), "direct_answer", ("x", "y_alpha", "z_beta"), "vector", "elastic_vector",
        "标准声明下是 [1, α_e, -β_e]。后面的风轴分量和增量式由答案边继续走。",
        {"145": {"x": "1", "y_alpha": "+α_e", "z_beta": "-β_e"}},
    ),
    _factor(
        "F_ELASTIC_TO_WIND", _D,
        (
            ("C_SIGN_GAMMA_TP", ""), ("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ELASTIC_BETA", ""),
            ("C_SIGN_ELASTIC_GAMMA", ""), ("C_CONV_PASSIVE", ""), ("C_METHOD_SMALL_ANGLE", ""),
        ),
        ("146", "147", "148"), "direct_answer",
        ("alpha_coefficient_sign", "beta_coefficient_sign", "sin_gamma_tp_sign", "cos_gamma_tp_sign"),
        "formula", "elastic_wind",
        "这三空是风轴分量。弹性角展开式是 ID176、ID177，不并进这三空。",
        {"146": {"expression": "1"}},
    ),
    _factor(
        "F_DELTA_ALPHA_COMPONENT", _D, (("C_SIGN_ALPHA", ""), ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes")),
        ("149",), "direct_method", ("wind_component",), "component_choice", "delta_alpha",
    ),
    _factor(
        "F_DELTA_BETA_COMPONENT", _D, (("C_SIGN_BETA", ""), ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes")),
        ("150",), "direct_method", ("wind_component",), "component_choice", "delta_beta",
    ),
    _factor(
        "F_ELASTIC_ALPHA_AIRSPEED", _D, (("C_SIGN_ELASTIC_ALPHA", ""), ("C_INSTALL_MODEL", "")),
        ("152",), "direct_answer", ("airspeed_side",), "direction", "elastic",
    ),
    _factor(
        "F_ELASTIC_BETA_AIRSPEED", _D, (("C_SIGN_ELASTIC_BETA", ""), ("C_INSTALL_MODEL", "")),
        ("153",), "direct_answer", ("airspeed_side",), "direction", "elastic",
    ),
    _factor(
        "F_DELTA_ALPHA_SIGN", _D, (("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ALPHA", ""), ("C_INSTALL_MODEL", "")),
        ("158",), "direct_answer", ("outer_sign",), "sign", "delta_alpha",
    ),
    _factor(
        "F_DELTA_BETA_SIGN", _D, (("C_SIGN_ELASTIC_BETA", ""), ("C_SIGN_BETA", ""), ("C_INSTALL_MODEL", "")),
        ("159",), "direct_answer", ("outer_sign",), "sign", "delta_beta",
    ),
    _factor(
        "F_DELTA_GAMMA_SIGN", _D, (("C_SIGN_ELASTIC_GAMMA", ""), ("C_SIGN_GAMMA", "")),
        ("160",), "direct_answer", ("outer_sign",), "sign", "delta_gamma",
    ),
    _factor(
        "F_DELTA_ALPHA_PARTS", _D,
        (("C_METHOD_DELTA_ALPHA_TEMPLATE", ""), ("C_SIGN_ALPHA", ""), ("C_AXIS_WIND_Y", "axes"), ("C_AXIS_WIND_X", "axes")),
        ("164", "165", "166", "167", "168"), "direct_answer", ("outer_sign", "numerator", "denominator"),
        "formula_parts", "delta_alpha",
    ),
    _factor(
        "F_DELTA_BETA_PARTS", _D,
        (("C_METHOD_DELTA_BETA_TEMPLATE", ""), ("C_SIGN_BETA", ""), ("C_AXIS_WIND_Y", "axes"), ("C_AXIS_WIND_X", "axes")),
        ("169", "170", "171", "172", "173"), "direct_answer", ("outer_sign", "numerator", "denominator"),
        "formula_parts", "delta_beta",
    ),
    _factor(
        "F_DELTA_ALPHA_FORMULA", _D,
        (("C_METHOD_DELTA_ALPHA_TEMPLATE", ""), ("C_SIGN_ALPHA", ""), ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes")),
        ("174",), "direct_answer", ("wind_component_quotient",), "formula", "delta_alpha",
        "只写风轴分量分式，不写弹性角。",
        {"174": {"expression": "(YFD)/(XFD)"}},
    ),
    _factor(
        "F_DELTA_BETA_FORMULA", _D,
        (("C_METHOD_DELTA_BETA_TEMPLATE", ""), ("C_SIGN_BETA", ""), ("C_AXIS_WIND_X", "axes"), ("C_AXIS_WIND_Y", "axes")),
        ("175",), "direct_answer", ("wind_component_quotient",), "formula", "delta_beta",
        {"175": {"expression": "(-ZFD)/(XFD)"}},
    ),
    _factor(
        "F_DELTA_ALPHA_ELASTIC", _D,
        (("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ELASTIC_BETA", ""), ("C_SIGN_GAMMA_TP", ""), ("C_CONV_PASSIVE", "")),
        ("176",), "direct_answer", ("alpha_coefficient_sign", "beta_coefficient_sign", "sin_sign", "cos_sign"),
        "formula", "delta_alpha",
        "把风轴分量再展开成弹性角。和 ID174 不是同一个应用。",
        {"176": {"expression": "α_e*cos(γ_tp) - β_e*sin(γ_tp)"}},
    ),
    _factor(
        "F_DELTA_BETA_ELASTIC", _D,
        (("C_SIGN_ELASTIC_ALPHA", ""), ("C_SIGN_ELASTIC_BETA", ""), ("C_SIGN_GAMMA_TP", ""), ("C_CONV_PASSIVE", "")),
        ("177",), "direct_answer", ("alpha_coefficient_sign", "beta_coefficient_sign", "sin_sign", "cos_sign"),
        "formula", "delta_beta",
        {"177": {"expression": "α_e*sin(γ_tp) + β_e*cos(γ_tp)"}},
    ),
    _factor(
        "F_DELTA_GAMMA_FORMULA", _D, (("C_SIGN_ELASTIC_GAMMA", ""), ("C_SIGN_GAMMA", "")),
        ("178",), "direct_answer", ("outer_sign", "angle_symbol"), "formula", "delta_gamma",
        {"178": {"expression": "γ_e"}},
    ),
    _factor(
        "F_XR", _R, (("C_GEOM_MODEL_LENGTH", ""), ("C_GEOM_SUPPORT_LENGTH", ""), ("C_GEOM_BALANCE_LENGTH", "")),
        ("182",), "direct_answer", ("distance_expression",), "formula", "geometry",
        {"182": {"expression": "L + L_t - L_r"}},
    ),
    _factor(
        "F_ENGINEERING_SLOT", _R, (("C_GEOM_ENGINEERING_AXIAL", ""), ("C_AXIS_BODY_X", "")),
        ("183",), "direct_answer", ("engineering_component",), "component_choice", "coordinate",
        {"183": {"component": "Xg"}},
    ),
    _factor(
        "F_REFERENCE_COORDINATE", _R,
        (
            ("C_GEOM_ENGINEERING_AXIAL", ""), ("C_AXIS_BODY_X", ""),
            ("C_GEOM_MODEL_LENGTH", ""), ("C_GEOM_SUPPORT_LENGTH", ""), ("C_GEOM_BALANCE_LENGTH", ""),
        ),
        ("184",), "direct_answer", ("axial_component", "normal_component", "lateral_component"),
        "vector", "coordinate",
        "标准声明、XTZ 指向前方时，轴向分量是 Xr - Xg。",
        {"184": {"expression": "[Xr - Xg, Yg, Zg]", "axial_component": "Xr - Xg"}},
    ),
    _factor(
        "F_TORQUE_SIGNS", _R,
        (
            ("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""),
            ("C_GEOM_ENGINEERING_AXIAL", ""),
        ),
        ("189", "190", "191", "192", "193", "194"), "direct_answer", ("outer_sign",),
        "sign", "torque_sign",
        "只负责外层正负。力系数绑定是另一个因子。",
    ),
    _factor(
        "F_TORQUE_BINDING", _R, (("C_FORCE_AXIS", ""), ("C_MOMENT_AXIS", "")),
        ("189", "190", "191", "192", "193", "194"), "direct_method", ("coefficient",),
        "semantic_binding", "torque_binding",
    ),
    _factor(
        "F_TORQUE_FORMULA", _R,
        (
            ("C_METHOD_TORQUE_TEMPLATE", ""), ("C_FORCE_AXIS", ""), ("C_MOMENT_AXIS", ""),
            ("C_GEOM_ENGINEERING_AXIAL", ""), ("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""),
        ),
        ("195", "196", "197", "198", "199", "200"), "direct_answer",
        ("coefficient", "lever_arm", "outer_sign"), "formula", "torque_formula",
        "标准声明下的六项。换手性或轴向指向后，系数、力臂和符号都会变，不能把这一列当成所有定义的金标。",
        {
            "195": {"name": "MCN_CA", "coefficient": "CN", "lever_arm": "Zg", "outer_sign": "+"},
            "196": {"name": "MCZ_CA", "coefficient": "CZ", "lever_arm": "Yg", "outer_sign": "-"},
            "197": {"name": "MCA_CZ", "coefficient": "CA", "lever_arm": "Yg", "outer_sign": "+"},
            "198": {"name": "MCN_CZ", "coefficient": "CN", "lever_arm": "Xr - Xg", "outer_sign": "-"},
            "199": {"name": "MCA_CN", "coefficient": "CA", "lever_arm": "Zg", "outer_sign": "-"},
            "200": {"name": "MCZ_CN", "coefficient": "CZ", "lever_arm": "Xr - Xg", "outer_sign": "+"},
        },
    ),
)


def _answers():
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    return NODE_BY_ID


def _ids(values) -> list[str]:
    return sorted(set(values), key=int)


def _instance_process(instance_id: str) -> str:
    _global_id, process, _slot = _split_instance(instance_id)
    if process not in PROCESSES:
        raise RuntimeError(f"{instance_id} 的子过程无法识别。")
    return process


def _parent_instance(process: str, global_id: str, slot: str) -> str:
    if global_id not in _GLOBAL:
        raise RuntimeError(f"未知契约原子 {global_id}。")
    return _iid(global_id, process, slot)


def _rule_id(factor_id: str) -> str:
    if not factor_id.startswith("F_"):
        raise RuntimeError(f"绑定规则应由应用记录转换，收到 {factor_id}。")
    return "RULE_" + factor_id[2:]


def _recognition_index() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for global_id, process, slot, targets, _relation, _field in _RECOGNITION:
        instance_id = _parent_instance(process, global_id, slot)
        found.setdefault(instance_id, set()).update(targets)
    return found


# These conventions are fixed in the generator. Without a direct question they
# do not enter the evaluation graph; a binding may still name them as context.
_LATENT_UNLESS_ASKED = frozenset({
    "C_CONV_PASSIVE",
    "C_CONV_INTRINSIC",
    "C_CONV_MOVING_FRAME",
    "C_METHOD_SMALL_ANGLE",
})


def _status(instance_id: str, recognition: dict[str, set[str]]) -> str:
    """observed has a question. derived has a checkable answer field but no question.

    latent is fixed context. It stays out of the evaluation graph.
    """
    if recognition.get(instance_id):
        return "observed"
    global_id, _process, _slot = _split_instance(instance_id)
    if global_id in _LATENT_UNLESS_ASKED:
        return "latent"
    return "derived"


def _statuses() -> dict[str, str]:
    recognition = _recognition_index()
    found = {instance_id: _status(instance_id, recognition) for instance_id in recognition}
    for factor in _FACTORS:
        for global_id, slot in factor["parents"]:
            instance_id = _parent_instance(factor["process"], global_id, slot)
            found.setdefault(instance_id, _status(instance_id, recognition))
    return found


def _authored() -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Direct C→answer edges. Joint parents share a binding id; they are not nodes."""
    statuses = _statuses()
    found: dict[str, dict[str, str]] = {}
    edges: list[dict] = []

    def add(instance_id: str, target: str, relation: str, field: str, combination: str, binding: str) -> None:
        if statuses[instance_id] == "latent":
            return
        slot = found.setdefault(instance_id, {})
        previous = slot.get(target)
        if previous is not None and previous != relation:
            raise RuntimeError(f"{instance_id} 对 ID{target} 同时是 {previous} 和 {relation}。")
        slot[target] = relation
        if previous is None:
            edges.append({
                "source": instance_id,
                "target": target,
                "relation": relation,
                "field": field,
                "combination": combination,
                "binding": binding,
            })

    for global_id, process, slot, targets, relation, field in _RECOGNITION:
        instance_id = _parent_instance(process, global_id, slot)
        for target in targets:
            add(instance_id, target, relation, field, "single", "")
    for factor in _FACTORS:
        rule_id = _rule_id(factor["id"])
        for global_id, slot in factor["parents"]:
            instance_id = _parent_instance(factor["process"], global_id, slot)
            for target in factor["targets"]:
                add(
                    instance_id, target, factor["relation"], ",".join(factor["fields"]),
                    "joint", rule_id,
                )
    return found, edges


def _check() -> None:
    answers = _answers()
    edge_count = sum(len(item.depends_on) for item in answers.values())
    if len(answers) != 194 or edge_count != 388:
        raise RuntimeError(f"答案图应为 194 空、388 边，现在是 {len(answers)} 空、{edge_count} 边。")
    factor_ids = [item["id"] for item in _FACTORS]
    if len(factor_ids) != len(set(factor_ids)):
        raise RuntimeError("契约绑定规则编号重复。")
    if any(not item["id"].startswith("F_") for item in _FACTORS):
        raise RuntimeError("内部应用记录必须以 F_ 开头，输出时才改成 RULE_。")
    direct, edges = _authored()
    for edge in edges:
        if edge["source"].startswith(("F_", "RULE_")) or edge["target"].startswith(("F_", "RULE_")):
            raise RuntimeError("绑定规则不能成为图节点。")
        if edge["combination"] not in ("single", "joint"):
            raise RuntimeError(f"未知的契约作用方式 {edge['combination']}。")
    depth = max(item.dependency_depth for item in answers.values())
    if depth != 8:
        raise RuntimeError(f"答案图深度应为 8，现在是 {depth}。")
    for item in answers.values():
        for parent in item.depends_on:
            if parent.startswith(("C_", "F_", "RULE_")):
                raise RuntimeError(f"ID{item.blank_id} 的推导边连到了契约对象 {parent}。")
    for instance_id, targets in direct.items():
        process = _instance_process(instance_id)
        process_name = PROCESSES[process]
        for target, relation in targets.items():
            if target not in answers:
                raise RuntimeError(f"{instance_id} 指向不存在的 ID{target}。")
            if target in _NOT_AUTHORED:
                raise RuntimeError(f"{instance_id} 把下游或已删除空 ID{target} 标成了直接作用。")
            if answers[target].process != process_name:
                raise RuntimeError(f"{instance_id} 跨子过程连到了 ID{target}。")
            if relation not in AUTHORED_RELATIONS:
                raise RuntimeError(f"{instance_id} 不能手工标注 {relation}。")
    for factor in _FACTORS:
        if factor["process"] not in PROCESSES:
            raise RuntimeError(f"{factor['id']} 的子过程无法识别。")
        if len(factor["parents"]) < 2:
            raise RuntimeError(f"{factor['id']} 不是联合条件。")
        if factor["relation"] not in AUTHORED_RELATIONS:
            raise RuntimeError(f"{factor['id']} 不能手工标注 {factor['relation']}。")
        if "52" in factor["targets"] or "101" in factor["targets"]:
            raise RuntimeError(f"{factor['id']} 指向已删除空。")
    gamma = next(item for item in _FACTORS if item["id"] == "F_GAMMA_MATRIX")
    if gamma["targets"] != ("43",):
        raise RuntimeError("R_γ_M 的应用点应为 ID43。")
    if "55" not in answers["43"].descendants or "52" in answers:
        raise RuntimeError("ID43 的答案后继应包含 ID55，且 ID52 不在图中。")
    installs = [key for key in direct if key.startswith("C_INSTALL_MODEL@")]
    if len(installs) < 5:
        raise RuntimeError("安装声明的重复读取没有分成不同实例。")
    wind_passive = direct[_iid("C_CONV_PASSIVE", _W)]
    if "44" not in wind_passive:
        raise RuntimeError("风轴系的被动约定应直接询问 ID44。")
    if "43" not in wind_passive:
        raise RuntimeError("风轴系的被动约定应作为 ID43 绑定规则的联合父项。")
    if any(answers[target].process != PROCESSES[_W] for target in wind_passive):
        raise RuntimeError("风轴系的被动实例连到了其他子过程。")
    formula = next(item for item in _FACTORS if item["id"] == "F_TORQUE_FORMULA")
    if "195" not in formula["targets"] or "184" in formula["targets"]:
        raise RuntimeError("力臂式的直接作用点是 ID195–ID200。")
    vector = next(item for item in _FACTORS if item["id"] == "F_ELASTIC_AXIS_VECTOR")
    if vector["targets"] != ("145",):
        raise RuntimeError("小角度轴向向量是 ID145。")
    elastic = next(item for item in _FACTORS if item["id"] == "F_DELTA_ALPHA_ELASTIC")
    quotient = next(item for item in _FACTORS if item["id"] == "F_DELTA_ALPHA_FORMULA")
    if set(elastic["targets"]) & set(quotient["targets"]):
        raise RuntimeError("ID174 和 ID176 必须分属两个因子。")
    _check_paths()


def _contribution(blank_ids: list[str]) -> list[str]:
    names = {
        "matrix": "matrix",
        "vector": "vector",
        "expression": "formula",
        "sign": "sign",
        "choice": "selection",
    }
    found = []
    answers = _answers()
    for blank_id in blank_ids:
        name = names[answers[blank_id].answer_type]
        if name not in found:
            found.append(name)
    return found


@lru_cache(maxsize=None)
def _simple_vias(start: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    """All simple answer paths from start to each descendant.

    The returned via includes start and every intermediate, and excludes the
    destination. Each start has fewer than 200 root-to-leaf paths, so every
    simple path is kept. Shortest-only would drop ID60 → ID63 → ID70 → ID80
    beside the direct ID60 → ID80 edge.
    """
    answers = _answers()
    found: dict[str, list[tuple[str, ...]]] = {}

    def walk(node: str, via: tuple[str, ...]) -> None:
        for child in sorted(answers[node].direct_dependents, key=int):
            route = via + (node,)
            found.setdefault(child, []).append(route)
            walk(child, route)

    walk(start, ())
    return {target: tuple(routes) for target, routes in found.items()}


def _indirect_records(instance_id: str, directs: dict[str, str]) -> list[dict]:
    records = []
    direct_ids = set(directs)
    for start in sorted(direct_ids, key=int):
        for target, routes in _simple_vias(start).items():
            if target in direct_ids:
                continue
            for via in routes:
                records.append({
                    "contract": instance_id,
                    "target": f"ID{target}",
                    "relation": "indirect",
                    "direct_target": f"ID{start}",
                    "via": [f"ID{item}" for item in via],
                    "contribution_types": _contribution(list(via) + [target]),
                    "potential": True,
                    "actual_error_path": None,
                })
    return records


def _check_paths() -> None:
    """The stored routes follow the real answer edges, not the schematic chain."""
    removed = {"ID52", "ID53", "ID54", "ID101", "ID102", "ID103"}
    from_sixty = [
        tuple(item["via"])
        for item in _indirect_records("probe", {"60": "direct_answer"})
        if item["target"] == "ID80"
    ]
    if ("ID60",) not in from_sixty or ("ID60", "ID63", "ID70") not in from_sixty:
        raise RuntimeError("ID60 到 ID80 应同时保留直连和 ID63、ID70 那条路径。")
    if any("ID22" in via or "ID68" in via for via in from_sixty):
        raise RuntimeError("ID60 的路径不能接到 ID22→ID68。")
    from_sign = [
        tuple(item["via"])
        for item in _indirect_records("probe", {"22": "direct_answer"})
        if item["target"] == "ID80"
    ]
    if ("ID22", "ID68") not in from_sign or any("ID60" in via for via in from_sign):
        raise RuntimeError("ID22 到 ID80 只走 ID68，不经过 ID60。")
    from_gamma = _indirect_records("probe", {"43": "direct_answer"})
    if not any(item["target"] == "ID55" and item["via"] == ["ID43"] for item in from_gamma):
        raise RuntimeError("ID43 的直接后继是 ID55。")
    if any(part in removed for item in from_gamma for part in item["via"] + [item["target"]]):
        raise RuntimeError("路径里出现了已删除的编号。")
    direct, _edges = _authored()
    alpha = _iid("C_SIGN_ALPHA", _W)
    alpha_routes = [
        via
        for start in direct[alpha]
        for via in _simple_vias(start).get("80", ())
    ]
    if not alpha_routes or any("60" in via for via in alpha_routes):
        raise RuntimeError("理论迎角符号到 ID80 的路径不能经过 ID60。")
    if not any(via[-2:] == ("22", "68") for via in alpha_routes):
        raise RuntimeError("理论迎角符号到 ID80 应经过 ID22→ID68。")
    handed = _iid("C_AXIS_BODY_HANDEDNESS", _W)
    handed_routes = [
        via
        for start in direct[handed]
        for via in _simple_vias(start).get("80", ())
    ]
    if not any(
        via[0] == "43" and "55" in via and "22" not in via
        for via in handed_routes
    ):
        raise RuntimeError("体轴手性到 ID80 应有一条从 ID43 经 ID55、且不经过 ID22 的路径。")


def _direct_relation(edge: dict) -> dict:
    target = edge["target"]
    return {
        "contract": edge["source"],
        "target": f"ID{target}",
        "relation": "direct",
        "direct_target": f"ID{target}",
        "effect": edge["field"] or edge["relation"],
        "combination": edge["combination"],
        "binding": edge["binding"],
        "via": [],
        "contribution_types": _contribution([target]),
        "potential": True,
        "actual_error_path": None,
    }


def _attach_relations(answer_rules: dict, direct: dict, edges: list) -> dict[str, int]:
    """Store derived contract relations on answers. Do not touch depends_on."""
    by_answer: dict[str, list] = {}
    for edge in edges:
        by_answer.setdefault(edge["target"], []).append(_direct_relation(edge))
    counts: dict[str, int] = {}
    for instance_id, targets in direct.items():
        records = _indirect_records(instance_id, targets)
        counts[instance_id] = len(records)
        for record in records:
            by_answer.setdefault(record["target"].removeprefix("ID"), []).append(record)
    answers = _answers()
    removed = {"ID52", "ID53", "ID54", "ID101", "ID102", "ID103"}
    for blank_id, records in by_answer.items():
        records.sort(key=lambda item: (
            item["contract"],
            0 if item["relation"] == "direct" else 1,
            len(item["via"]),
            tuple(item["via"]),
        ))
        for item in records:
            if item["target"] != f"ID{blank_id}":
                raise RuntimeError(f"ID{blank_id} 的契约关系指向了 {item['target']}。")
            if item["potential"] is not True or item["actual_error_path"] is not None:
                raise RuntimeError("结构文件只保存潜在契约依赖，实际错误路径必须为空。")
            if any(part in removed for part in item["via"]):
                raise RuntimeError(f"ID{blank_id} 的路径含已删除编号。")
            if item["relation"] == "direct" and item["via"]:
                raise RuntimeError(f"ID{blank_id} 的直接契约关系不应带路径。")
            if item["relation"] == "indirect" and item["via"][0] != item["direct_target"]:
                raise RuntimeError(f"ID{blank_id} 的间接路径没有从直接作用点开始。")
        slot = answer_rules.setdefault(blank_id, {
            "contract_targets": [],
            "sensitive_fields": [],
            "binding_rule_ids": [],
            "process": answers[blank_id].process,
        })
        slot["contract_relations"] = records
    for slot in answer_rules.values():
        slot.setdefault("contract_relations", [])
    return counts


def _impacts(direct: dict[str, dict[str, str]]) -> dict[str, dict[str, tuple[str, ...]]]:
    answers = _answers()
    impacts = {}
    for instance_id, targets in direct.items():
        reached: set[str] = set()
        for target in targets:
            reached.add(target)
            reached.update(answers[target].descendants)
        impacts[instance_id] = {
            "direct": tuple(_ids(targets)),
            "downstream": tuple(_ids(reached - set(targets))),
        }
    return impacts


def _answer_rules(statuses: dict[str, str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for factor in _FACTORS:
        rule_id = _rule_id(factor["id"])
        parents = [
            _parent_instance(factor["process"], global_id, slot)
            for global_id, slot in factor["parents"]
            if statuses[_parent_instance(factor["process"], global_id, slot)] != "latent"
        ]
        for target in factor["targets"]:
            slot = found.setdefault(target, {
                "contract_targets": [],
                "sensitive_fields": [],
                "binding_rule_ids": [],
            })
            slot["binding_rule_ids"].append(rule_id)
            slot["contract_targets"].extend(parents)
            slot["sensitive_fields"].extend(factor["fields"])
    direct, _edges = _authored()
    for instance_id, targets in direct.items():
        for target in targets:
            slot = found.setdefault(target, {
                "contract_targets": [],
                "sensitive_fields": [],
                "binding_rule_ids": [],
            })
            slot["contract_targets"].append(instance_id)
    for target, slot in found.items():
        slot["contract_targets"] = list(dict.fromkeys(slot["contract_targets"]))
        slot["sensitive_fields"] = list(dict.fromkeys(slot["sensitive_fields"]))
        slot["binding_rule_ids"] = list(dict.fromkeys(slot["binding_rule_ids"]))
        slot["process"] = _answers()[target].process
    return found


def contract_layer() -> dict:
    """Serializable layer. Binding rules are not graph nodes."""
    _check()
    statuses = _statuses()
    direct, edges = _authored()
    impacts = _impacts(direct)
    answer_rules = _answer_rules(statuses)
    indirect_counts = _attach_relations(answer_rules, direct, edges)
    instances = []
    latent = []
    for instance_id, status in sorted(statuses.items()):
        global_id, process, slot = _split_instance(instance_id)
        spec = _GLOBAL[global_id]
        record = {
            "id": instance_id,
            "atom_type": "contract_instance",
            "global_id": global_id,
            "kind": spec[1],
            "name": spec[2],
            "standard": spec[3],
            "perturbed": spec[4],
            "process": PROCESSES[process],
            "process_key": process,
            "slot": slot,
            "status": status,
            "observed": status == "observed",
        }
        if status == "latent":
            latent.append(record)
            continue
        impact = impacts[instance_id]
        record["direct_targets"] = [f"ID{item}" for item in impact["direct"]]
        record["downstream_targets"] = [f"ID{item}" for item in impact["downstream"]]
        record["indirect_path_count"] = indirect_counts.get(instance_id, 0)
        instances.append(record)
    globals_out = []
    for global_id, spec in _GLOBAL.items():
        owned = [item["id"] for item in instances if item["global_id"] == global_id]
        hidden = [item["id"] for item in latent if item["global_id"] == global_id]
        if not owned and not hidden:
            continue
        globals_out.append({
            "id": global_id,
            "atom_type": "global_contract",
            "kind": spec[1],
            "name": spec[2],
            "standard": spec[3],
            "perturbed": spec[4],
            "instances": owned,
            "latent_instances": hidden,
        })
    bindings = []
    for factor in _FACTORS:
        parents = []
        context = []
        for global_id, slot in factor["parents"]:
            instance_id = _parent_instance(factor["process"], global_id, slot)
            if statuses[instance_id] == "latent":
                context.append(instance_id)
            else:
                parents.append(instance_id)
        bindings.append({
            "rule_id": _rule_id(factor["id"]),
            "kind": "contract_binding",
            "node": False,
            "in_graph": bool(parents),
            "process": PROCESSES[factor["process"]],
            "process_key": factor["process"],
            "chain": factor["chain"],
            "combination": "joint",
            "relation": factor["relation"],
            "rule_type": factor["kind"],
            "contract_parents": parents,
            "fixed_context": context,
            "targets": [f"ID{target}" for target in factor["targets"]],
            "sensitive_fields": list(factor["fields"]),
            "standard_sensitive": {
                f"ID{target}": value for target, value in factor["standard"].items()
            },
            "note": factor["note"],
        })
    return {
        "meaning": (
            "正式图只人工标注契约到直接答案，以及原有的 388 条答案推导边。"
            "契约到后续答案的间接关系由答案图的全部简单路径计算，记在答案的 contract_relations 上，不写回 depends_on。"
            "这是潜在依赖，与某次运行无关。actual_error_path 在本文件中为空："
            "只有路径上每个答案都错，才构成实际错误路径；正确的中间节点阻断串联。"
            "单点注入只观察被改答案的直接子节点。模型前缀沿后代闭包看潜在影响，归因停在最早的错误节点。"
        ),
        "formula": "契约直接作用答案，答案边推出间接路径。潜在依赖和实际错误路径分开存放。",
        "storage": {
            "authored_direct": "direct_edges。只人工标注契约原子对直接受影响答案的作用。",
            "answer_edges": "答案图的 depends_on。仍是 388 条推导边，不写入契约，也不写入间接路径。",
            "contract_relations": "answer_bindings 上的派生字段。direct 是人工边，indirect 带 via 和贡献类型。同一契约的多条简单路径都保留。",
            "potential_contract_dependency": "contract_relations 中 potential 为 true 的记录。只说明直接作用的答案若变化，标准推导可能影响哪些后续答案。",
            "actual_error_path": "结构文件里恒为 null。合并一次运行时，只有 via 与目标上的每个答案都错，才把该 via 写入；中间有正确节点则为 null。",
            "single_point_injection": "只改变一个直接作用答案时，观察它的 direct_dependents，不用整段后代闭包。",
            "model_prefix": "沿答案后代看潜在影响。第一个错节点是归因起点，影响只沿仍然错误的后代继续，正确的中间节点阻断实际错误路径。",
            "paths": "不把多源影响压成一条路径。ID22→ID68→ID80 与 ID60→ID80、ID60→ID63→ID70→ID80 分开保存。ID43 的后继是 ID55，不使用已删除的 ID52。",
        },
        "id_notes": [
            "公开编号用答案图的编号。天平系到体轴系从 ID82 起，天平系到风轴系从 ID117 起，力矩参考点从 ID181 起。",
            "ID52–54 和 ID101–103 不建节点。ID43 的答案后继是 ID55。",
            "安装滚转矩阵是 ID88，RTP_TZ1 是 ID95，RTP_TZ2 是 ID97，载荷总矩阵是 ID104。",
            "小角度天平轴向向量是 ID145。风轴分量是 ID146–148。风轴分式是 ID174、ID175。弹性角展开是 ID176、ID177。",
            "力矩参考点的数学坐标是 ID184，符号空是 ID189–194，力臂式是 ID195–200。",
            "ID45 问的是动轴或定轴，不是体轴手性。手性进入 ID7，并和滚转轴、正方向一起绑定 ID43 的符号。",
            "RULE_* 是契约绑定规则，不是答案，也不占路径长度。",
            "间接契约关系按答案图的全部简单路径保存。ID43 之后是 ID55，不是已删除的 ID52。",
        ],
        "relation_states": {
            "direct": "存在人工标注的契约作用边。",
            "indirect": "没有这条直接边，但契约的某个直接作用点能沿答案边到达该空。via 是具体路径。",
            "none": "没有可行路径。contract_relations 里不出现该契约。",
        },
        "relation_types": {
            "direct_answer": "该契约直接改变这一空的答案取值。",
            "direct_method": "该契约直接规定方法、转轴、顺序或名称绑定。",
            "scope_constraint": "该契约约束坐标系或对象范围，不因此改变这一空的金标。",
            "downstream": "只从答案图的后代计算，不是手工边，也不是契约作用边。",
        },
        "statuses": {
            "observed": "有直接问答。",
            "derived": "没有直接问答，但答案上有可核对的字段。",
            "latent": "生成器里的固定约定。没有单独问答时不进入评测图，只留在绑定规则的固定上下文里。",
        },
        "global_atom_count": len(globals_out),
        "atom_count": len(instances),
        "latent_count": len(latent),
        "binding_count": len(bindings),
        "direct_edge_count": len(edges),
        "indirect_path_count": sum(indirect_counts.values()),
        "contract_relation_count": sum(len(slot["contract_relations"]) for slot in answer_rules.values()),
        "graph": {
            "contract_nodes": len(instances),
            "answer_nodes": len(_answers()),
            "node_count": len(instances) + len(_answers()),
            "contract_edges": len(edges),
            "answer_edges": sum(len(item.depends_on) for item in _answers().values()),
            "bindings_are_nodes": False,
        },
        "answer_graph": {
            "blank_count": len(_answers()),
            "edge_count": sum(len(item.depends_on) for item in _answers().values()),
        },
        "diagnosis_fields": {
            "run_state": "一次运行在答案图上的 local_error、propagated_error、recovered 或 correct。绑定规则没有运行状态。",
            "definition_identified": "该实例的直接问答是否答对。",
            "sensitive_element_correct": "绑定规则列出的敏感字段是否答对。",
            "full_answer_correct": "完整矩阵、向量或公式是否答对。",
            "failure_stage": "直接问答正确而敏感字段错误时，标成契约没有落实到数学对象。",
            "potential_relation": "contract_relations 记录图上的潜在路径，与某次运行无关。",
            "actual_error_path": "结构文件中为空。路径上出现正确的中间答案时，不能把前后错误串成一条实际错误路径。",
        },
        "globals": globals_out,
        "instances": instances,
        "latent_contracts": latent,
        "contract_bindings": bindings,
        "direct_edges": [
            {
                "source": item["source"],
                "target": f"ID{item['target']}",
                "relation": item["relation"],
                "field": item["field"],
                "combination": item["combination"],
                "binding": item["binding"],
            }
            for item in edges
        ],
        "answer_bindings": {
            f"ID{blank_id}": slot
            for blank_id, slot in sorted(answer_rules.items(), key=lambda item: int(item[0]))
        },
        "processes": [
            {
                "key": key,
                "name": name,
                "instances": sum(item["process_key"] == key for item in instances),
                "latent": sum(item["process_key"] == key for item in latent),
                "bindings": sum(item["process_key"] == key for item in bindings),
                "direct_answers": len({
                    target
                    for instance_id, targets in direct.items()
                    if _instance_process(instance_id) == key
                    for target in targets
                }),
            }
            for key, name in PROCESSES.items()
        ],
    }


if __name__ == "__main__":
    layer = contract_layer()
    graph = layer["graph"]
    print(
        f"graph nodes {graph['node_count']} = contracts {graph['contract_nodes']} "
        f"+ answers {graph['answer_nodes']}; "
        f"contract edges {graph['contract_edges']}, answer edges {graph['answer_edges']}; "
        f"bindings {layer['binding_count']} not nodes; latent {layer['latent_count']} excluded; "
        f"indirect paths {layer['indirect_path_count']}; "
        f"contract relations {layer['contract_relation_count']}"
    )
    for process in layer["processes"]:
        print(
            f"{process['name']}: instances {process['instances']}, "
            f"latent {process['latent']}, bindings {process['bindings']}, "
            f"direct answers {process['direct_answers']}"
        )
    for note in layer["id_notes"]:
        print(note)
