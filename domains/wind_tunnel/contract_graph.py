# -*- coding: utf-8 -*-
"""Contract atoms on the answer dependency graph.

The evaluation graph has two kinds of nodes only: contract instances C and
the 207 public answers. Edges are either C→answer direct effects, or the
existing 389 answer→answer derivation edges. A binding rule records which
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
    "wind_to_body": "wind frame to body-axis frame",
    "balance_to_body": "balance frame to body-axis frame",
    "balance_to_wind": "balance frame to wind frame",
    "reference_point": "moment reference point",
}

# Closure only. These stay out of authored contract targets.
_NOT_AUTHORED = frozenset({
    "52", "53", "54", "101", "102", "103",
    "55", "59", "60", "61", "80", "81",
})

_GLOBALS = (
    ("C_CONV_WIND_TO_BODY", "conv", "conversion direction from the wind frame to the body-axis frame", "wind_to_body", ""),
    ("C_CONV_BALANCE_TO_BODY", "conv", "conversion direction from the balance frame to the body-axis frame", "balance_to_body", ""),
    ("C_CONV_BALANCE_TO_WIND", "conv", "conversion direction from the balance frame to the wind frame", "balance_to_wind", ""),
    ("C_CONV_PASSIVE", "conv", "coordinate transforms use the passive form", "passive", ""),
    ("C_CONV_INTRINSIC", "conv", "attitude rotations use an intrinsic moving-axis sequence", "intrinsic", ""),
    ("C_CONV_MOVING_FRAME", "conv", "the body-axis frame is moving axes", "moving", ""),
    ("C_AXIS_WIND_X", "axis", "wind-frame X points forward along the tunnel axial axis", "tunnel_forward", "tunnel_lateral"),
    ("C_AXIS_WIND_Y", "axis", "wind-frame Y points upward along the tunnel normal", "tunnel_up", "tunnel_lateral"),
    ("C_AXIS_WIND_HANDEDNESS", "axis", "handedness of the wind frame", "right_handed", "left_handed"),
    ("C_AXIS_BODY_X", "axis", "body-axis XTZ points forward along the airframe axial axis", "body_forward", "body_aft"),
    ("C_AXIS_BODY_Y", "axis", "body-axis YTZ points upward along the airframe normal", "body_up", "body_down"),
    ("C_AXIS_BODY_HANDEDNESS", "axis", "handedness of the body-axis frame", "right_handed", "left_handed"),
    ("C_AXIS_BALANCE_X", "axis", "balance XTP points forward along the balance axial axis", "balance_forward", "balance_aft"),
    ("C_AXIS_BALANCE_Y", "axis", "balance YTP points upward along the balance normal", "balance_up", "balance_down"),
    ("C_AXIS_BALANCE_HANDEDNESS", "axis", "handedness of the balance frame", "right_handed", "left_handed"),
    ("C_AXIS_ALPHA", "axis", "mechanism angle of attack about the absolute lateral axis", "absolute_lateral", ""),
    ("C_AXIS_BETA", "axis", "mechanism sideslip about the absolute normal axis", "absolute_normal", ""),
    ("C_AXIS_GAMMA", "axis", "mechanism roll about the absolute axial axis", "absolute_forward", ""),
    ("C_AXIS_GAMMA_TP", "axis", "balance installation angle about the absolute balance axis", "balance_axial", ""),
    ("C_AXIS_ELASTIC_ALPHA", "axis", "rotation axis of the elastic angle of attack", "balance_lateral", ""),
    ("C_AXIS_ELASTIC_BETA", "axis", "rotation axis of the elastic sideslip", "balance_normal", ""),
    ("C_AXIS_ELASTIC_GAMMA", "axis", "rotation axis of the elastic roll", "balance_axial", ""),
    ("C_SIGN_ALPHA_M", "sign", "positive direction of the mechanism angle of attack", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_ALPHA", "sign", "positive direction of the theoretical angle of attack", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_BETA_M", "sign", "positive direction of the mechanism sideslip", "airspeed_body_right", "airspeed_body_left"),
    ("C_SIGN_BETA", "sign", "positive direction of the theoretical sideslip", "airspeed_body_right", "airspeed_body_left"),
    ("C_SIGN_GAMMA_M", "sign", "positive direction of the mechanism roll", "right_wing_down", "right_wing_up"),
    ("C_SIGN_GAMMA", "sign", "positive direction of the theoretical roll", "right_wing_down", "right_wing_up"),
    ("C_SIGN_GAMMA_TP", "sign", "positive direction of the balance installation angle", "right_hand_about_balance_forward", "opposite"),
    ("C_SIGN_ELASTIC_ALPHA", "sign", "positive direction of the elastic angle of attack", "balance_up", "balance_down"),
    ("C_SIGN_ELASTIC_BETA", "sign", "positive direction of the elastic sideslip", "balance_left", "balance_right"),
    ("C_SIGN_ELASTIC_GAMMA", "sign", "positive direction of the elastic roll", "clockwise_looking_forward", "counterclockwise"),
    ("C_SIGN_FLOW_ALPHA", "sign", "positive direction of the flow-offset angle of attack", "airspeed_body_down", "airspeed_body_up"),
    ("C_SIGN_FLOW_BETA", "sign", "positive direction of the flow-offset sideslip", "airspeed_body_right", "airspeed_body_left"),
    ("C_INSTALL_MODEL", "install", "upright or inverted model installation", "upright", "inverted"),
    ("C_METHOD_EULER_ORDER", "method", "order of the mechanism angles", "alpha_beta_gamma", "beta_alpha_gamma"),
    ("C_METHOD_PREPOSE_SLOTS", "method", "pre-matrices occupy R4 and R5 in the product", "R4_RP2_R5_RP1", ""),
    ("C_METHOD_LOAD_PRODUCT_ORDER", "method", "order of the load product", "R1_RTP_TZ2_R2_RTP_TZ1_R3_roll", ""),
    ("C_METHOD_WIND_PRODUCT_ORDER", "method", "order of the product from the balance frame to the wind frame", "R1_RP_TP_FD_R2_R_gamma_tp", ""),
    ("C_METHOD_ROLL_DIFFERENCE", "method", "roll difference between the balance frame and the airframe", "gamma_m_minus_gamma_tp", ""),
    ("C_METHOD_ALPHA_INVERSE", "method", "inverse function of the theoretical angle of attack", "arctan", ""),
    ("C_METHOD_BETA_INVERSE", "method", "inverse function of the theoretical sideslip", "arcsin", ""),
    ("C_METHOD_BETA_UNIT_DENOMINATOR", "method", "the theoretical sideslip denominator is positive 1", "positive_one", ""),
    ("C_METHOD_DELTA_ALPHA_TEMPLATE", "method", "the elastic angle of attack increment is a quotient with a positive numerator and denominator", "positive_quotient", ""),
    ("C_METHOD_DELTA_BETA_TEMPLATE", "method", "the elastic sideslip increment is a quotient with a positive numerator and denominator", "positive_quotient", ""),
    ("C_METHOD_SMALL_ANGLE", "method", "elastic deformation uses the small-angle approximation", "sin_theta_eq_theta", ""),
    ("C_METHOD_TORQUE_TEMPLATE", "method", "equivalent moments use C·L_c/L_RT", "coefficient_dot_arm_over_L_RT", ""),
    ("C_FORCE_AXIS", "force", "force coefficients correspond to absolute axes", "CN_normal_CA_axial_CZ_lateral", ""),
    ("C_MOMENT_AXIS", "force", "moment coefficients correspond to absolute axes", "MN_normal_MA_axial_MZ_lateral", ""),
    ("C_GEOM_MODEL_LENGTH", "geom", "L from the model nose to the model base", "L", ""),
    ("C_GEOM_SUPPORT_LENGTH", "geom", "L_t from the model base to the bottom of the sting", "L_t", ""),
    ("C_GEOM_BALANCE_LENGTH", "geom", "L_r from the balance center to the bottom of the sting", "L_r", ""),
    ("C_GEOM_ORIGIN", "geom", "the body-axis origin is at the balance center", "balance_center", ""),
    ("C_GEOM_ENGINEERING_AXIAL", "geom", "the engineering axial component is measured rearward from the nose", "rearward_from_nose", ""),
    ("C_TORQUE_SOURCE", "geom", "start of the moment transfer", "origin", ""),
    ("C_TORQUE_TARGET", "geom", "end of the moment transfer", "reference_point", ""),
    ("C_FORCE_APPLICATION", "geom", "location of the force that produces the extra moment", "origin", ""),
    ("C_TORQUE_EFFECT", "geom", "location where the extra moment acts", "reference_point", ""),
    ("C_SCOPE_ALIGNMENT_TARGET", "scope", "frame the alignment matrix acts on", "target_frame", ""),
    ("C_SCOPE_ALIGNMENT_SOURCE", "scope", "frame the alignment matrix is compared with", "source_frame", ""),
    ("C_SCOPE_ROTATION_FRAME", "scope", "attitude rotations are written in the intermediate body-axis frame", "body_frame", ""),
    ("C_SCOPE_LEVEL_FRAME", "scope", "after the level pose only the level airframe is studied", "level_body", ""),
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
        "The lateral axis is inferred from the wind-frame handedness and the declared X and Y.",
    ),
    _factor(
        "F_BODY_Z", _W, (("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", "")),
        ("7",), "direct_answer", ("axis_direction",), "axis_inference", "body_axes",
        "ZTZ is inferred from the body-axis handedness and the declared X and Y. It is not an atom rewritten on its own.",
    ),
    _factor(
        "F_RP1", _W,
        (
            ("C_AXIS_WIND_X", ""), ("C_AXIS_WIND_Y", ""), ("C_AXIS_WIND_HANDEDNESS", ""),
            ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", ""), ("C_AXIS_BODY_HANDEDNESS", ""),
        ),
        ("8",), "direct_answer", ("diagonal_pattern", "off_diagonal_sign"), "matrix", "rp1",
        "The first use of the axis declaration in the alignment matrix. Under the upright standard it can be the identity, and the identity is still the result of these declarations.",
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
        "Passive form and intrinsic rotation decide how the matrix is written. The current generator does not perturb them alone to change a single-axis element; element changes come from handedness, the rotation axis, and the positive direction.",
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
        "The note on passive form and intrinsic rotation is the same as for F_ALPHA_MATRIX.",
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
        "The standard right-handed passive matrix about X is [[1,0,0],[0,cos,sin],[0,-sin,cos]]. Left-handedness flips sin. Intrinsic rotation does not change these elements by itself; it enters the product factors.",
        {"43": {"axis": "X", "upper_off_diagonal_sign": "+", "lower_off_diagonal_sign": "-"}},
    ),
    _factor(
        "F_ATTITUDE_PRODUCT", _W,
        (("C_METHOD_EULER_ORDER", ""), ("C_CONV_PASSIVE", ""), ("C_CONV_INTRINSIC", "")),
        ("47", "48", "49"), "direct_method", ("matrix_order",), "product_order", "product",
        "It decides only R1, R2, and R3. R4 and R5 do not read the Euler order.",
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
        "The local installation-angle matrix. The total matrix ID104 has its own product factors.",
    ),
    _factor(
        "F_BALANCE_Z", _B, (("C_AXIS_BALANCE_HANDEDNESS", ""), ("C_AXIS_BALANCE_X", ""), ("C_AXIS_BALANCE_Y", "")),
        ("91",), "direct_answer", ("axis_direction",), "axis_inference", "balance_axes",
    ),
    _factor(
        "F_BODY_Z_LOAD", _B, (("C_AXIS_BODY_HANDEDNESS", ""), ("C_AXIS_BODY_X", ""), ("C_AXIS_BODY_Y", "")),
        ("94",), "direct_answer", ("axis_direction",), "axis_inference", "body_axes",
        "The body-axis directions are read again in this subprocess and do not depend on ID7 of wind frame to body-axis frame.",
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
        "The standard order is R1=RTP_TZ2, R2=RTP_TZ1, R3=the installation roll.",
    ),
    _factor(
        "F_LOAD_TOTAL", _B, (("C_METHOD_LOAD_PRODUCT_ORDER", ""), ("C_CONV_PASSIVE", "")),
        ("104",), "direct_answer", ("matrix_product",), "matrix", "load_total",
        "The combined execution point. Whether the component matrices are right is read at ID88, ID95, and ID97; those contracts are not attached again at ID104.",
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
        "The standard order is R1=RP_TP_FD, R2=R_γ_tp.",
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
        "Under the standard declaration it is [1, α_e, -β_e]. Later wind-frame components and increment formulas continue along answer edges.",
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
        "These three blanks are wind-frame components. The elastic-angle expansions are ID176 and ID177 and are not folded into these three blanks.",
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
        "Write only the wind-frame quotient, not an elastic angle.",
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
        "Expand the wind-frame components into elastic angles. This is not the same application as ID174.",
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
        "Under the standard declaration, when XTZ points forward, the axial component is Xr - Xg.",
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
        "It is responsible only for the outer sign. The force-coefficient binding is another factor.",
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
        "The six terms under the standard declaration. After a change of handedness or axial direction, the coefficient, the arm, and the sign all change, so this column is not the gold for every definition.",
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
        raise RuntimeError(f"The subprocess of {instance_id} is not recognized.")
    return process


def _parent_instance(process: str, global_id: str, slot: str) -> str:
    if global_id not in _GLOBAL:
        raise RuntimeError(f"Unknown contract atom {global_id}.")
    return _iid(global_id, process, slot)


def _rule_id(factor_id: str) -> str:
    if not factor_id.startswith("F_"):
        raise RuntimeError(f"A binding rule should be converted from an application record; got {factor_id}.")
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
            raise RuntimeError(f"{instance_id} is both {previous} and {relation} for ID{target}.")
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
    if len(answers) != 207 or edge_count != 406:
        raise RuntimeError(f"The answer graph should have 207 blanks and 406 edges, found {len(answers)} blanks and {edge_count} edges.")
    factor_ids = [item["id"] for item in _FACTORS]
    if len(factor_ids) != len(set(factor_ids)):
        raise RuntimeError("Contract binding-rule ids are duplicated.")
    if any(not item["id"].startswith("F_") for item in _FACTORS):
        raise RuntimeError("An internal application record must start with F_ and is renamed to RULE_ only on output.")
    direct, edges = _authored()
    for edge in edges:
        if edge["source"].startswith(("F_", "RULE_")) or edge["target"].startswith(("F_", "RULE_")):
            raise RuntimeError("A binding rule cannot be a graph node.")
        if edge["combination"] not in ("single", "joint"):
            raise RuntimeError(f"Unknown contract combination {edge['combination']}.")
    depth = max(item.dependency_depth for item in answers.values())
    if depth != 8:
        raise RuntimeError(f"The answer-graph depth should be 8, found {depth}.")
    for item in answers.values():
        for parent in item.depends_on:
            if parent.startswith(("C_", "F_", "RULE_")):
                raise RuntimeError(f"A derivation edge of ID{item.blank_id} points at contract object {parent}.")
    for instance_id, targets in direct.items():
        process = _instance_process(instance_id)
        process_name = PROCESSES[process]
        for target, relation in targets.items():
            if target not in answers:
                raise RuntimeError(f"{instance_id} points at missing ID{target}.")
            if target in _NOT_AUTHORED:
                raise RuntimeError(f"{instance_id} marks downstream or removed blank ID{target} as a direct effect.")
            if answers[target].process != process_name:
                raise RuntimeError(f"{instance_id} crosses subprocesses to ID{target}.")
            if relation not in AUTHORED_RELATIONS:
                raise RuntimeError(f"{instance_id} cannot be hand-labeled {relation}.")
    for factor in _FACTORS:
        if factor["process"] not in PROCESSES:
            raise RuntimeError(f"The subprocess of {factor['id']} is not recognized.")
        if len(factor["parents"]) < 2:
            raise RuntimeError(f"{factor['id']} is not a joint condition.")
        if factor["relation"] not in AUTHORED_RELATIONS:
            raise RuntimeError(f"{factor['id']} cannot be hand-labeled {factor['relation']}.")
        if "52" in factor["targets"] or "101" in factor["targets"]:
            raise RuntimeError(f"{factor['id']} points at a removed blank.")
    gamma = next(item for item in _FACTORS if item["id"] == "F_GAMMA_MATRIX")
    if gamma["targets"] != ("43",):
        raise RuntimeError("The application point of R_γ_M should be ID43.")
    if "55" not in answers["43"].descendants or "52" in answers:
        raise RuntimeError("The answer successors of ID43 should include ID55, and ID52 is not in the graph.")
    installs = [key for key in direct if key.startswith("C_INSTALL_MODEL@")]
    if len(installs) < 5:
        raise RuntimeError("Repeated reads of the installation declaration were not split into different instances.")
    wind_passive = direct[_iid("C_CONV_PASSIVE", _W)]
    if "44" not in wind_passive:
        raise RuntimeError("The passive convention of the wind frame should ask ID44 directly.")
    if "43" not in wind_passive:
        raise RuntimeError("The passive convention of the wind frame should be a joint parent of the ID43 binding rule.")
    if any(answers[target].process != PROCESSES[_W] for target in wind_passive):
        raise RuntimeError("The passive instance of the wind frame connects to another subprocess.")
    formula = next(item for item in _FACTORS if item["id"] == "F_TORQUE_FORMULA")
    if "195" not in formula["targets"] or "184" in formula["targets"]:
        raise RuntimeError("The direct effect of the moment-arm formula is ID195-ID200.")
    vector = next(item for item in _FACTORS if item["id"] == "F_ELASTIC_AXIS_VECTOR")
    if vector["targets"] != ("145",):
        raise RuntimeError("The small-angle axial vector is ID145.")
    elastic = next(item for item in _FACTORS if item["id"] == "F_DELTA_ALPHA_ELASTIC")
    quotient = next(item for item in _FACTORS if item["id"] == "F_DELTA_ALPHA_FORMULA")
    if set(elastic["targets"]) & set(quotient["targets"]):
        raise RuntimeError("ID174 and ID176 must belong to two different factors.")
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
        raise RuntimeError("ID60 to ID80 should keep both the direct edge and the path through ID63 and ID70.")
    if any("ID22" in via or "ID68" in via for via in from_sixty):
        raise RuntimeError("The path of ID60 must not connect to ID22->ID68.")
    from_sign = [
        tuple(item["via"])
        for item in _indirect_records("probe", {"22": "direct_answer"})
        if item["target"] == "ID80"
    ]
    if ("ID22", "ID68") not in from_sign or any("ID60" in via for via in from_sign):
        raise RuntimeError("ID22 to ID80 goes only through ID68 and does not pass ID60.")
    from_gamma = _indirect_records("probe", {"43": "direct_answer"})
    if not any(item["target"] == "ID55" and item["via"] == ["ID43"] for item in from_gamma):
        raise RuntimeError("The direct successor of ID43 is ID55.")
    if any(part in removed for item in from_gamma for part in item["via"] + [item["target"]]):
        raise RuntimeError("A path contains a removed id.")
    direct, _edges = _authored()
    alpha = _iid("C_SIGN_ALPHA", _W)
    alpha_routes = [
        via
        for start in direct[alpha]
        for via in _simple_vias(start).get("80", ())
    ]
    if not alpha_routes or any("60" in via for via in alpha_routes):
        raise RuntimeError("The path from the theoretical angle-of-attack sign to ID80 must not pass ID60.")
    if not any(via[-2:] == ("22", "68") for via in alpha_routes):
        raise RuntimeError("The path from the theoretical angle-of-attack sign to ID80 should pass ID22->ID68.")
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
        raise RuntimeError("Body-axis handedness to ID80 should have a path from ID43 through ID55 that does not pass ID22.")


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
                raise RuntimeError(f"The contract relation of ID{blank_id} points at {item['target']}.")
            if item["potential"] is not True or item["actual_error_path"] is not None:
                raise RuntimeError("The structure file stores only potential contract dependencies, so the actual error path must be empty.")
            if any(part in removed for part in item["via"]):
                raise RuntimeError(f"The path of ID{blank_id} contains a removed id.")
            if item["relation"] == "direct" and item["via"]:
                raise RuntimeError(f"The direct contract relation of ID{blank_id} should not carry a path.")
            if item["relation"] == "indirect" and item["via"][0] != item["direct_target"]:
                raise RuntimeError(f"The indirect path of ID{blank_id} does not start at the direct effect.")
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
            "The published graph hand-labels only contracts onto direct answers, plus the existing 389 answer-derivation edges. "
            "Indirect relations from a contract to later answers are computed from every simple path in the answer graph, stored on the answer's contract_relations, and not written back into depends_on. "
            "That is a potential dependency and is independent of any one run. actual_error_path is empty in this file: "
            "an actual error path exists only when every answer on the path is wrong; a correct intermediate node blocks the chain. "
            "A single-point injection watches only the direct children of the changed answer. A model prefix follows the descendant closure for potential effect, and attribution stops at the earliest wrong node."
        ),
        "formula": "A contract acts directly on an answer, and answer edges produce the indirect path. Potential dependencies and actual error paths are stored apart.",
        "storage": {
            "authored_direct": "direct_edges. Hand-labeled effects of a contract atom on the answers it changes directly.",
            "answer_edges": "depends_on of the answer graph. Still 389 derivation edges, with no contracts and no indirect paths written in.",
            "contract_relations": "A derived field on answer_bindings. direct is a hand edge, and indirect carries via and a contribution type. Every simple path of the same contract is kept.",
            "potential_contract_dependency": "Records in contract_relations whose potential is true. They say which later answers the standard derivation may affect if a directly acted-on answer changes.",
            "actual_error_path": "Always null in the structure file. When a run is merged in, a via is written only if every answer on that via and the target is wrong; a correct intermediate node leaves it null.",
            "single_point_injection": "When only one directly acted-on answer changes, watch its direct_dependents and do not use the whole descendant closure.",
            "model_prefix": "Follow answer descendants for potential effect. The first wrong node is where attribution starts, and the effect continues only along descendants that are still wrong. A correct intermediate node blocks the actual error path.",
            "paths": "Do not collapse multi-source effects into one path. ID22->ID68->ID80, ID60->ID80, and ID60->ID63->ID70->ID80 are stored separately. The successor of ID43 is ID55, and the removed ID52 is not used.",
        },
        "id_notes": [
            "Public ids are the answer-graph ids. balance frame to body-axis frame starts at ID82, balance frame to wind frame starts at ID117, and moment reference point starts at ID181.",
            "ID52-54 and ID101-103 are not nodes. The answer successor of ID43 is ID55.",
            "The installation roll matrix is ID88, RTP_TZ1 is ID95, RTP_TZ2 is ID97, and the total load matrix is ID104.",
            "The small-angle balance axial vector is ID145. The wind-frame components are ID146-148. The wind-frame quotients are ID174 and ID175. The elastic-angle expansions are ID176 and ID177.",
            "The mathematical coordinates of the moment reference point are ID184, the sign blanks are ID189-194, and the moment-arm formulas are ID195-200.",
            "ID45 asks whether the axes are moving or fixed, not the body-axis handedness. Handedness enters at ID7 and, with the roll axis and the positive direction, binds the sign of ID43.",
            "RULE_* is a contract binding rule. It is not an answer and it does not add to the path length.",
            "Indirect contract relations are stored as every simple path of the answer graph. After ID43 comes ID55, not the removed ID52.",
        ],
        "relation_states": {
            "direct": "A hand-labeled contract-effect edge exists.",
            "indirect": "There is no such direct edge, but some direct effect of the contract can reach this blank along answer edges. via is the concrete path.",
            "none": "No feasible path. The contract does not appear in contract_relations.",
        },
        "relation_types": {
            "direct_answer": "This contract directly changes the answer value of this blank.",
            "direct_method": "This contract directly sets the method, the rotation axis, the order, or a name binding.",
            "scope_constraint": "This contract constrains the frame or the object, and does not by that change the gold of this blank.",
            "downstream": "Computed only from descendants in the answer graph. It is not a hand edge and not a contract-effect edge.",
        },
        "statuses": {
            "observed": "There is a direct question and answer.",
            "derived": "There is no direct question, but the answer has a field that can be checked.",
            "latent": "A fixed convention in the generator. When there is no separate question it stays out of the scored graph and remains only in the fixed context of a binding rule.",
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
            "run_state": "One run on the answer graph is local_error, propagated_error, recovered, or correct. A binding rule has no run state.",
            "definition_identified": "Whether the direct question of this instance was answered correctly.",
            "sensitive_element_correct": "Whether the sensitive fields listed by the binding rule were answered correctly.",
            "full_answer_correct": "Whether the full matrix, vector, or formula was answered correctly.",
            "failure_stage": "When the direct question is right and a sensitive field is wrong, mark that the contract was not carried into the mathematical object.",
            "potential_relation": "contract_relations records potential paths on the graph and is independent of any one run.",
            "actual_error_path": "Empty in the structure file. When a correct intermediate answer appears on the path, the errors before and after it cannot be chained into one actual error path.",
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
