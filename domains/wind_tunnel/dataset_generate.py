# -*- coding: utf-8 -*-
"""Build review samples as one closed declaration plus the continuous cloze.

Human review shows the whole subprocess, with every blank marked in the formal
slot. An evaluation call answers one whole sentence. Every blank in that
sentence stays open together, earlier blanks are the gold answers, and the
text ends at that sentence's period. The first ten samples are for manual
checking.

Run::

    python -m domains.wind_tunnel.dataset_generate
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from cloze_score import answer_letter, format_choices, resolve_choice
from body_frame_model import (
    ALPHA,
    ALPHA_E,
    ALPHA_M,
    ALPHA_V,
    BETA,
    BETA_E,
    BETA_M,
    BETA_V,
    BODY_AXES,
    BALANCE_VOCAB,
    GAMMA,
    GAMMA_E,
    GAMMA_M,
    GAMMA_MODEL,
    GAMMA_TP,
    WIND_VOCAB,
    BodyFrameModel,
    OPPOSITE,
    PLACE_UPRIGHT,
    SENSE_SAME,
    WRIST_ORDER,
    _body_direction_in_tunnel,
    _direction_name,
    roll_about_nose_sense,
)
from domain_layout import DOMAINS
from domains.wind_tunnel.error_graph import bind_reading_order, question_fields
from domains.wind_tunnel.ref_point_generate import generate as generate_ref
from domains.wind_tunnel.tp_fd_generate import generate as generate_tp_fd
from domains.wind_tunnel.tp_tz_generate import generate as generate_tp_tz
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC, generate as generate_tz_fd


OUT_DIR = DOMAINS["wind_tunnel"]["review"]

PROCESSES = (
    ("wind frame to body-axis frame", generate_tz_fd),
    ("balance frame to body-axis frame", generate_tp_tz),
    ("balance frame to wind frame", generate_tp_fd),
    ("moment reference point", generate_ref),
)
PROCESS_SIZE = {
    "wind frame to body-axis frame": 81,
    "balance frame to body-axis frame": 35,
    "balance frame to wind frame": 64,
    "moment reference point": 33,
}
# The component-matrix expansions before the total matrix were removed. Those ids stay empty and later numbers are not shifted.
REMOVED_SLOTS = {
    "wind frame to body-axis frame": frozenset({52, 53, 54}),
    "balance frame to body-axis frame": frozenset({101, 102, 103}),
}
PROCESS_OFFSET = {}
_cursor = 0
for _name, _size in PROCESS_SIZE.items():
    PROCESS_OFFSET[_name] = _cursor
    _cursor += _size


def _public_id(process: str, local_id: str) -> str:
    return str(int(local_id) + PROCESS_OFFSET[process])


def _local_id(process: str, public_id: str) -> str:
    return str(int(public_id) - PROCESS_OFFSET[process])

DIRECTION_ZH = {
    "body_forward": "forward on the airframe",
    "body_rearward": "rearward on the airframe",
    "body_up": "upward on the airframe",
    "body_down": "downward on the airframe",
    "body_left": "left on the airframe",
    "body_right": "right on the airframe",
    "tunnel_forward": "forward in the tunnel (freestream)",
    "tunnel_rearward": "rearward in the tunnel",
    "tunnel_up": "upward in the tunnel",
    "tunnel_down": "downward in the tunnel",
    "tunnel_left": "left in the tunnel",
    "tunnel_right": "right in the tunnel",
    "balance_forward": "forward on the balance",
    "balance_rearward": "rearward on the balance",
    "balance_up": "upward on the balance",
    "balance_down": "downward on the balance",
    "balance_left": "left on the balance",
    "balance_right": "right on the balance",
}
AXIS_ZH = {
    "body_axial": "the airframe axial axis",
    "body_normal": "the airframe normal axis",
    "body_lateral": "the airframe lateral axis",
    "tunnel_axial": "the tunnel axial axis",
    "tunnel_normal": "the tunnel normal axis",
    "tunnel_lateral": "the tunnel lateral axis",
    "balance_axial": "the balance axial axis",
    "balance_normal": "the balance normal axis",
    "balance_lateral": "the balance lateral axis",
}
ANGLE_NAME = {
    ALPHA: "theoretical angle of attack α",
    BETA: "theoretical sideslip β",
    GAMMA: "theoretical roll γ",
    ALPHA_M: "mechanism angle of attack α_M",
    BETA_M: "mechanism sideslip β_M",
    GAMMA_M: "mechanism roll γ_M",
}
HAND_ZH = {"right": "right-handed", "left": "left-handed"}
SOURCE_ZH = {"declared": "declared", "inferred": "inferred from chirality"}

MATRIX_FORMAT = (
    "Write one 3×3 matrix on a single line, as [[a, b, c], [d, e, f], [g, h, i]]. "
    "Put exactly one space after each comma. Each entry is an integer or an expression in cos, sin, and the angle symbols of this item. Use * for multiplication. Do not break the line and do not use LaTeX."
)
VECTOR_FORMAT = (
    "Write three components on one line, as [a, b, c]. Put exactly one space after each comma. "
    "Each component is an integer or an expression in symbols already used in this item. Use * for multiplication, and write trigonometric functions as cos(α_M). "
    "Do not break the line and do not use LaTeX."
)
EXPRESSION_FORMAT = (
    "Write one symbolic expression on a single line. Use * for multiplication and / for division. Write trigonometric functions as cos(α_M), with no space inside the parentheses. "
    "Put a minus sign directly before a term, as in -sin(α_M). Do not use LaTeX and do not write an equals sign."
)
ANGLE_SYMBOL_NOTE = (
    "Angle symbols in a formula are the symbols of this item: α_M, β_M, γ_M, γ_tp, α_e, and β_e. The symbol θ appears only in format examples; do not write θ in an answer."
)
DELTA_ALPHA_QUOTIENT_NOTE = (
    "This blank must be a quotient of the wind-frame components XFD, YFD, and ZFD. "
    "The numerator and the denominator must both be positive, and a minus sign may appear only before the whole quotient. "
    "Do not write an elastic-angle expression. The answer must not contain α_e, β_e, γ_e, or γ_tp."
)
DELTA_ALPHA_QUOTIENT_FORMAT = (
    "Write one quotient on a single line. The numerator and the denominator must each be one of XFD, YFD, and ZFD, and both must be positive; a minus sign goes only before the whole quotient. "
    "Do not write trigonometric functions, and do not use α_e, β_e, γ_e, or γ_tp. Do not use LaTeX and do not write an equals sign."
)
DELTA_ALPHA_QUOTIENT_EXAMPLE = "-(A)/(B)"
DISTANCE_FORMAT = (
    "Write one distance expression on a single line, built from L, L_t, and L_r with plus or minus signs. Leave one space on each side of every plus or minus. "
    "Do not use LaTeX."
)
ARCTAN_FORMAT = (
    "Write one line containing exactly one arctan. The numerator and the denominator must both be positive; the sign of the angle is written outside the whole arctan. "
    "A positive angle is arctan((numerator)/(denominator)), and a negative angle is -arctan((numerator)/(denominator)). "
    "Do not make the numerator and the denominator negative at the same time, and do not put the minus sign inside. Do not use LaTeX."
)
ARCSIN_FORMAT = (
    "Write one line containing exactly one arcsin. The numerator must be positive and the denominator stays positive; the sign of the angle is written outside the whole arcsin. "
    "A positive angle is arcsin(numerator), and a negative angle is -arcsin(numerator). "
    "Do not make the numerator negative, do not put the minus sign inside, and do not write a quotient. Do not use LaTeX."
)
MOMENT_FORMAT = (
    "Write one line of the form -CN·Xg/L_RT. Use a middle dot · between the force coefficient and the moment arm. "
    "Put the minus sign outside the whole expression. The arm itself must be positive and is the absolute value of the extra moment; do not put a minus sign inside the arm. "
    "The force coefficient must be one of CA, CZ, and CN. The arm is built from Xg, Yg, Zg, and Xr; add parentheses when the arm contains a subtraction. "
    "Keep L_RT as the denominator. Do not leave the letters C or L_c, do not write an equals sign, and do not use LaTeX."
)
FORMAT_CODE = {
    MATRIX_FORMAT: "matrix",
    VECTOR_FORMAT: "vector",
    EXPRESSION_FORMAT: "expression",
    DISTANCE_FORMAT: "distance",
    ARCTAN_FORMAT: "arctan",
    ARCSIN_FORMAT: "arcsin",
    MOMENT_FORMAT: "moment-arm",
}
FORMAT_TEXT = {
    "choice": "Multiple choice. Write only the option letter, with no quotation marks and without the option text.",
    "matrix": MATRIX_FORMAT,
    "vector": VECTOR_FORMAT,
    "expression": EXPRESSION_FORMAT,
    "distance": DISTANCE_FORMAT,
    "arctan": ARCTAN_FORMAT,
    "arcsin": ARCSIN_FORMAT,
    "moment-arm": MOMENT_FORMAT,
}
FORMAT_EXAMPLE = {
    "choice": "A",
    "matrix": "[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]",
    "vector": "[0, -1, 1]",
    "expression": "cos(θ) - sin(θ)",
    "distance": "L - L_t + L_r",
    "arctan": "-arctan((cos(θ))/(sin(θ)))",
    "arcsin": "arcsin(cos(θ))",
    "moment-arm": "-CN·Xg/L_RT",
}
PRIOR = (
    "The positive axial direction is forward, the positive normal direction is upward, and the positive lateral direction is to the right. "
    "The positive model axis, the positive airframe axis, the positive balance axis, and the positive tunnel axis are this same absolute forward direction, not the positive direction of a coordinate axis. "
    "Changing only the angle of attack rotates about the absolute lateral axis. Changing only sideslip rotates about the absolute normal axis. "
    "Changing only roll rotates about the absolute axial axis. The positive sense of each angle is the sense in the declaration. "
    "Coordinate transforms are passive. Attitude angles use an intrinsic moving-axis sequence."
)
SLOT_MARK = re.compile(r"@@(\d+)@@")
PROCESS_TEXT = {
    "wind frame to body-axis frame": """1. In the initial state the body axes stay level. To compute the effect of the mechanism angles α_M, β_M, and γ_M on α_t, β_t, and roll before compensation, the required transform is @@1@@.

2. The positive absolute normal (upward), positive absolute axial (forward), and positive absolute lateral (right) directions of the wind frame correspond to @@2@@, @@3@@, and @@4@@. The same three directions of the airframe correspond to @@5@@, @@6@@, and @@7@@. The pre-transform from the wind frame to the body-axis frame at the level pose is therefore RP1. RP1 is @@8@@. That matrix makes the relation between the axes of @@9@@ and its own absolute directions the same as the relation between the axes of @@10@@ and its own absolute directions. The model installation is @@11@@. The positive absolute normal (upward) and positive absolute lateral (right) of the model are @@12@@ the positive absolute normal and positive absolute lateral of the tunnel, and the corresponding pre-transform is RP2. RP2 is @@13@@. That matrix makes the axis directions of @@14@@ the same as the coordinate directions of @@15@@. Later attitude transforms start from this intermediate body-axis frame, which keeps the airframe symbols XTZ, YTZ, and ZTZ and introduces no new symbols.

3. From the mechanism layout, the first rotation angle is @@16@@. The second rotation angle is @@17@@. The third rotation angle is @@18@@.

4. When α_M > 0 the balance deflects @@BALANCE_A@@ in the tunnel. The model uses @@19@@. When α_M > 0 the airspeed points @@20@@ on the airframe. When the theoretical angle of attack is positive the airspeed points @@21@@ on the airframe. When α_M > 0 the sign of the theoretical angle of attack is @@22@@. This is the level pose of @@23@@. The rotation is about @@24@@. The sense about that axis is @@25@@, through the angle α_M, and the rotation matrix is R_α_M. R_α_M is @@26@@.

5. When β_M > 0 the balance deflects @@BALANCE_B@@ in the tunnel. The model uses @@27@@. When β_M > 0 the airspeed points @@28@@ on the airframe. When the theoretical sideslip is positive the airspeed points @@29@@ on the airframe. When β_M > 0 the sign of the theoretical sideslip is @@30@@. This is the level pose of @@31@@. The rotation is about @@32@@. The sense about that axis is @@33@@, through the angle β_M, and the rotation matrix is R_β_M. R_β_M is @@34@@.

6. When γ_M > 0 the rotation axis is the positive absolute model axis (forward) and the sense is @@35@@ the right-hand rule. The right wing then points @@36@@ on the airframe. When the theoretical roll is positive the rotation axis is the positive absolute airframe axis (forward) and the sense is @@37@@ the right-hand rule. The right wing then points @@38@@ on the airframe. When γ_M > 0 the sign of the theoretical roll is @@39@@. This is the level pose of @@40@@. The rotation is about the coordinate axis @@41@@. The sense about that axis is @@42@@, through the angle γ_M, and the rotation matrix is R_γ_M. R_γ_M is @@43@@.

7. This step changes coordinates, so the transform is @@44@@. The body-axis frame is @@45@@. For that kind of frame the rotation uses @@46@@. The axis correspondence is transformed before the attitude. Write [XTZ, YTZ, ZTZ]^T = R1*R2*R3*R4*R5*[XFD, YFD, ZFD]^T. Choose R1 through R5 from the matrix names already obtained. R1 is @@47@@. R2 is @@48@@. R3 is @@49@@. R4 is @@50@@. R5 is @@51@@.

8. Write the symbolically simplified product R_t = R1*R2*R3. R_t = @@55@@.

9. The airspeed coordinates in the level airframe do not depend on the wind-frame coordinates, so RP2*RP1 times the wind-frame airspeed unit vector is @@56@@. It is therefore enough to map the airspeed coordinates in @@57@@ into the airframe after the attitude change.

10. Compute [XTZ, YTZ, ZTZ]^T = R_t*[X_V, Y_V, Z_V]^T. The airspeed unit vector [X_V, Y_V, Z_V] in the level frame is @@58@@. Substitution gives XTZ = @@59@@. It gives YTZ = @@60@@. It gives ZTZ = @@61@@.

11. When only α_M > 0, XTZ is @@62@@. YTZ is @@63@@. ZTZ is @@64@@.

12. When only β_M > 0, XTZ is @@65@@. YTZ is @@66@@. ZTZ is @@67@@.

13. The template for α_t is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The angle sign of α_t is @@68@@. The inverse function of α_t is @@69@@. The body-axis component in the numerator of α_t is @@71@@. From the sign of that component when only α_M > 0, the numerator sign of α_t is @@70@@. The body-axis component in the denominator of α_t is @@73@@. From the sign of that component when only α_M > 0, the denominator sign of α_t is @@72@@.

14. The template for β_t is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The angle sign of β_t is @@74@@. The inverse function of β_t is @@75@@. The body-axis component in the numerator of β_t is @@77@@. From the sign of that component when only β_M > 0, the numerator sign of β_t is @@76@@. The denominator sign of β_t is @@78@@. The denominator component of β_t is @@79@@.

15. In the initial state the body axes stay level, and the freestream is aligned with the wind-frame axis and blows toward the nose. After α_M, β_M, and γ_M change, the airframe is in a new attitude. From the known α_M, β_M, and γ_M, derive α_t and β_t in that attitude, without balance deformation or flow compensation. The template for the theoretical angle of attack α_t is [angle sign][inverse trigonometric function]([numerator]/[denominator]), and the template for β_t is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The theoretical angle of attack is α_t = @@80@@. The theoretical sideslip is β_t = @@81@@.""",
    "balance frame to body-axis frame": """1. Forces and moments measured in the balance frame must be converted to the body-axis frame, so the transform is @@1@@. The body-axis origin is taken to coincide with the balance center, so the frame is only rotated and is not translated.

2. First fix how the balance-frame axes and the airframe axes relate to their own absolute directions. The positive absolute normal (upward), positive absolute axial (forward), and positive absolute lateral (right) of the balance frame correspond to @@8@@, @@9@@, and @@10@@. The same three directions of the airframe correspond to @@11@@, @@12@@, and @@13@@. The pre-matrix from the balance frame to the body-axis frame is therefore RTP_TZ1. RTP_TZ1 is @@14@@.

3. Then remove the attitude difference caused by the installation angle, using the axis already identified. The installation rotation is performed in the axes originally defined for the balance. The balance installation angle is positive when it is @@2@@ the right-hand rule about the positive absolute balance axis (forward). The airframe roll is positive when it is @@3@@ the right-hand rule about the positive absolute airframe axis (forward). The positive absolute balance axis and the positive absolute airframe axis may point the same way or the opposite way. The absolute value of the airframe roll γ_M is much larger than the balance installation angle γ_tp. With the balance axis, the airframe axis, and both positive directions fixed, the roll between the balance frame and the airframe is @@4@@. That turn is about the coordinate axis @@5@@ that corresponds to the absolute balance axis. The sense about that axis is @@6@@, and the rotation matrix is R_(γ_M-γ_tp). R_(γ_M-γ_tp) is @@7@@.

4. Then account for upright or inverted installation so that the coordinate directions agree. If the model is inverted, the positive absolute normal (upward) and positive absolute lateral (right) of the airframe are opposite the tunnel, and RTP_TZ2 is also required. Under this declaration the model is @@15@@. The pre-matrix under this declaration is RTP_TZ2. RTP_TZ2 is @@16@@.

5. The attitude is transformed before the axis correspondence. The conversion is [XTZ, YTZ, ZTZ]^T = R1*R2*R3*[XTP, YTP, ZYP]^T. Choose R1 through R3 from the matrix names already obtained. R1 = @@17@@. R2 = @@18@@. R3 = @@19@@.

6. Write the symbolically simplified product R = R1*R2*R3. R = @@23@@.

7. In the six-component force and moment conversion, XTZ is associated with @@24@@, YTZ with @@25@@, ZTZ with @@26@@, XTP with @@27@@, YTP with @@28@@, and ZYP with @@29@@.

8. Substitute the six components into the conversion matrix. The body-axis components are CN_TZ = @@30@@, CA_TZ = @@31@@, CZ_TZ = @@32@@, MN_TZ = @@33@@, MA_TZ = @@34@@, and MZ_TZ = @@35@@.""",
    "balance frame to wind frame": """1. To compute the angle-of-attack and sideslip increments that the balance elastic angles produce in the wind frame, the required transform is @@1@@. The positive absolute tunnel axis (forward) is @@2@@, and the positive absolute balance axis (forward) is @@3@@. The positive absolute normal (upward), positive absolute lateral (right), and positive absolute axial (forward) of the wind frame correspond to @@4@@, @@5@@, and @@6@@. The same three directions of the balance correspond to @@7@@, @@8@@, and @@9@@, and the transform matrix is RP_TP_FD. RP_TP_FD is @@10@@. This matrix makes the axis directions of @@11@@ the same as the coordinate directions of @@12@@. The balance is first rotated to the level pose, and the coordinate directions are unified after that. RP_TP_FD is therefore applied after the balance rotation.

2. When the balance is level and the installation angle satisfies γ_tp > 0, the rotation is about the absolute @@13@@ of the balance. On the coordinate axes, the rotation is about @@14@@. The sense about that axis is @@15@@, through the angle γ_tp, and the rotation matrix is R_γ_tp. R_γ_tp is @@16@@.

3. The attitude is transformed before the axis correspondence. In [XFD, YFD, ZFD]^T = R1*R2*[XTP, YTP, ZYP]^T, choose R1 and R2 from the matrix names already obtained. R1 = @@17@@. R2 = @@18@@.

4. In the balance frame, the unit vector along the positive absolute balance axis (forward) is @@19@@. When the balance produces α_e, β_e, and γ_e, the balance axis bends toward @@20@@ and toward @@21@@. Looking along the positive absolute balance axis (forward), it also twists @@22@@. The three rotation axes are @@23@@, @@24@@, and @@25@@. The deformed axial unit vector, written in the undeformed frame, is [XTP_e, YTP_e, ZTP_e]. XTP_e is @@26@@. YTP_e is @@27@@. ZTP_e is @@28@@. The elastic angles are small, so use sin θ = θ and cos θ = 1. The axial unit vector in the original frame is then @@29@@.

5. Substitute the deformed axial unit vector. XFD = @@30@@. YFD = @@31@@. ZFD = @@32@@.

6. Express Δα_e and Δβ_e with XFD, YFD, and ZFD. Use sin θ = θ and cos θ = 1. For a positive theoretical angle of attack the airspeed points @@39@@ on the airframe. With that airspeed direction and the wind-frame axial, normal, and lateral directions fixed, the wind-frame component used by Δα_e is @@33@@. For a positive theoretical sideslip the airspeed points @@40@@ on the airframe. With that airspeed direction and the wind-frame axial, normal, and lateral directions fixed, the wind-frame component used by Δβ_e is @@34@@.

7. This step fixes the signs in the formulas for Δα_e and Δβ_e. γ_tp can make a sign depend on the values of α_e and β_e. Assume γ_tp = 0 and judge the signs from the boundary condition. The balance is then installed level. When the deformations α_e, β_e, and γ_e occur, the model is @@35@@. After those deformations the airspeed lies @@36@@ of the model. The airspeed also lies @@37@@ of the model. The rotation about the positive absolute balance axis (forward) is @@38@@ the right-hand rule. Roll is defined as positive when it is @@41@@ the right-hand rule about the positive absolute airframe axis (forward). Among the increments produced by the balance elastic angles, Δα_e is @@42@@. Δβ_e is @@43@@. Δγ_e is @@44@@. The same boundary condition fixes the coordinate signs: XFD is @@45@@. YFD is @@46@@. ZFD is @@47@@. When γ_tp is present, the directions of Δα_e and Δβ_e change from this initial state with the values of α_e and β_e.

8. The template in XFD, YFD, and ZFD is [angle sign]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the quotient. Do not put the sign inside the quotient in place of the outer sign. The angle sign of Δα_e is @@48@@. The wind-frame component in the numerator of Δα_e is @@50@@. From the sign of that component, the numerator sign of Δα_e is @@49@@. The wind-frame component in the denominator of Δα_e is @@52@@. From the sign of that component, the denominator sign of Δα_e is @@51@@. The angle sign of Δβ_e is @@53@@. The wind-frame component in the numerator of Δβ_e is @@55@@. From the sign of that component, the numerator sign of Δβ_e is @@54@@. The wind-frame component in the denominator of Δβ_e is @@57@@. From the sign of that component, the denominator sign of Δβ_e is @@56@@. Δα_e = @@58@@. This blank must be a quotient of the wind-frame components XFD, YFD, and ZFD, with a positive numerator and a positive denominator, and with a minus sign only before the whole quotient. Do not write an elastic-angle expression that contains α_e, β_e, or γ_e. Δβ_e = @@59@@. In α_e and β_e, Δα_e = @@60@@. In α_e and β_e, Δβ_e = @@61@@. The roll increment produced by the balance elastic angle does not change with the balance installation angle γ_tp. Its sign is whether the elastic roll and the theoretical roll have the same sense about the positive absolute axis (forward), so Δγ_e = @@62@@.

9. Δα_V and Δβ_V are the angle-of-attack and sideslip compensations caused by tunnel-flow offset. The positive sense is the sense that increases the actual angle of attack or sideslip of the airframe. The compensation formulas are α = α_t + Δα_e + Δα_V and β = β_t + Δβ_e + Δβ_V. The sign of the compensation Δα_V is @@63@@. The sign of the compensation Δβ_V is @@64@@.""",
    "moment reference point": """1. After the balance frame is converted to the body-axis frame, the origin of that body-axis frame is at @@1@@. The engineering data give a moment reference point, and the body-axis moments must be transferred to that point.

2. The balance is embedded in the model. The distance from the balance center to the model nose is Xr, and that distance is obtained by an indirect measurement. The distance from the balance center to the bottom of the balance sting is L_r, the model length is L, and the distance from the model base to the bottom of the sting is L_t. All three are measured, they are collinear, and the center lies between the nose and the sting base. Therefore Xr = @@2@@.

3. The engineering coordinates of the moment reference point are [Xg, Yg, Zg]. The engineering convention differs from the mathematical coordinates as follows: the axial component is the distance from the reference point to the model nose, so this engineering positive direction points rearward on the model and is opposite the positive absolute axis (forward); the other two components are positive in the same direction as the body-axis coordinate axes they lie on, and they are not rewritten as the absolute normal (upward) or the absolute lateral (right). The coordinate component @@3@@ may therefore differ from the mathematical body-axis coordinate. After conversion to mathematical body-axis coordinates, the moment reference point is @@4@@.

4. Transferring the moment from the body-axis @@5@@ to @@6@@ must include the extra moment that the force at @@7@@ produces at @@8@@.

5. MCA_g = MCA + [MCN_CA] + [MCZ_CA], MCZ_g = MCZ + [MCA_CZ] + [MCN_CZ], and MCN_g = MCN + [MCA_CN] + [MCZ_CN]. MCN_CA is the equivalent axial moment coefficient at the reference point produced by the normal force at the origin, and MCN_CZ is the equivalent lateral moment coefficient at the reference point produced by the normal force at the origin. Each term may be written only as C·L_c/L_RT, where L_RT is the fixed reference length. The letters C and L_c must not remain in the answer. C must be the force coefficient that produces that moment, and it must be one of CA, CZ, and CN. L_c must be the moment arm, and the arm may use only Xg, Yg, Zg, and Xr. Assume Xg, Yg, Zg, and Xr are all positive. The sign is placed outside the whole expression, and the arm itself must be positive and is the absolute value of the extra moment. Under that assumption, the sign of MCN_CA is @@9@@. The sign of MCZ_CA is @@10@@. The sign of MCA_CZ is @@11@@. The sign of MCN_CZ is @@12@@. The sign of MCA_CN is @@13@@. The sign of MCZ_CN is @@14@@. Which force coefficient is in the numerator of MCN_CA? @@21@@. Which force coefficient is in the numerator of MCZ_CA? @@22@@. Which force coefficient is in the numerator of MCA_CZ? @@23@@. Which force coefficient is in the numerator of MCN_CZ? @@24@@. Which force coefficient is in the numerator of MCA_CN? @@25@@. Which force coefficient is in the numerator of MCZ_CN? @@26@@. Which coordinate component is in the numerator of MCN_CA? @@27@@. Which coordinate component is in the numerator of MCZ_CA? @@28@@. Which coordinate component is in the numerator of MCA_CZ? @@29@@. Which coordinate component is in the numerator of MCN_CZ? @@30@@. Which coordinate component is in the numerator of MCA_CN? @@31@@. Which coordinate component is in the numerator of MCZ_CN? @@32@@. Here L_RT = L + L_t - L_r. Which length is the denominator of the equivalent moment coefficient? @@33@@. Therefore MCN_CA = @@15@@. MCZ_CA = @@16@@. MCA_CZ = @@17@@. MCN_CZ = @@18@@. MCA_CN = @@19@@. MCZ_CN = @@20@@.""",
}

bind_reading_order(PROCESS_TEXT, PROCESS_OFFSET)


def _base() -> dict:
    return deepcopy(STANDARD_SPEC)


def _samples() -> list[dict]:
    standard = _base()
    inverted = _base()
    inverted["placement"] = "inverted"
    alpha_up = _base()
    alpha_up["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}
    beta_left = _base()
    beta_left["beta"] = {"airspeed_toward": "body_left", "sign": "positive"}
    roll_install = _base()
    roll_install["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    roll_install["gamma_tp"] = {"sense": "opposite"}
    body_left = _base()
    body_left["body_frame"] = {
        "chirality": "left",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_rearward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    }
    wind_swap = _base()
    wind_swap["wind_frame"] = {
        "chirality": "right",
        "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
        "YFD": {"along": "tunnel_lateral", "points": "tunnel_right"},
    }
    elastic_flow = _base()
    elastic_flow["alpha_e"] = {"axial_bends": "balance_down", "sign": "positive"}
    elastic_flow["beta_e"] = {"axial_bends": "balance_right", "sign": "positive"}
    elastic_flow["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    elastic_flow["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    elastic_flow["beta_v"] = {"airspeed_toward": "body_left", "sign": "positive"}
    balance_left = _base()
    balance_left["gamma_model"] = {"sense": "opposite"}
    balance_left["balance_frame"] = {
        "chirality": "left",
        "XTP": {"along": "balance_axial", "points": "balance_rearward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    }
    return [
        {"id": "01", "title": "common default", "change": "Every definition stays at the common default.", "spec": standard},
        {"id": "02", "title": "inverted", "change": "Installation is changed to inverted; everything else stays at the default.", "spec": inverted},
        {"id": "03", "title": "theoretical angle of attack reversed", "change": "The theoretical angle of attack is positive when the airspeed points upward on the airframe. The mechanism angle of attack still points downward on the airframe.", "spec": alpha_up},
        {"id": "04", "title": "theoretical sideslip reversed", "change": "The theoretical sideslip is positive when the airspeed points left on the airframe. The mechanism sideslip still points right on the airframe.", "spec": beta_left},
        {"id": "05", "title": "roll and installation angle reversed", "change": "Mechanism roll is opposite the right-hand rule about the positive absolute model axis (forward). The balance installation angle is opposite the right-hand rule about the positive absolute balance axis (forward).", "spec": roll_install},
        {"id": "07", "title": "left-handed body axes pointing rearward", "change": "The body-axis frame is left-handed. XTZ points rearward along the airframe axial axis, and YTZ points upward on the airframe.", "spec": body_left},
        {"id": "08", "title": "wind-frame lateral and normal swapped", "change": "Wind-frame YFD points right along the tunnel lateral axis, and ZFD is inferred from the right-handed frame.", "spec": wind_swap},
        {"id": "09", "title": "elastic angles and flow-offset angles reversed", "change": "The positive directions of the three elastic angles and the two flow-offset angles are all opposite the default.", "spec": elastic_flow},
        {"id": "10", "title": "left-handed balance axes pointing rearward", "change": "The balance frame is left-handed. XTP points rearward along the balance axial axis. The model installation angle is opposite the right-hand rule.", "spec": balance_left},
    ]


def _direction(name: str) -> str:
    if name not in DIRECTION_ZH:
        raise KeyError(name)
    return DIRECTION_ZH[name]


def _tunnel_side(body_direction: str, placement: str) -> str:
    """Tunnel side occupied by a body direction under the current installation."""
    tunnel_name = _direction_name(_body_direction_in_tunnel(body_direction, placement), WIND_VOCAB)
    side = tunnel_name.split("_", 1)[1]
    return {
        "forward": "forward",
        "rearward": "rearward",
        "up": "upward",
        "down": "downward",
        "left": "left",
        "right": "right",
    }[side]


def _balance_deflection(airspeed_toward: str) -> str:
    """Tunnel direction of the balance. An upright model's nose moves the same way."""
    return _tunnel_side(OPPOSITE[airspeed_toward], PLACE_UPRIGHT)


def _mechanism_phrases(model: BodyFrameModel) -> dict[str, str]:
    """Balance directions used to state the mechanism angles."""
    return {
        "@@BALANCE_A@@": _balance_deflection(model.mechanism_angles[ALPHA_M].positive_airspeed_toward),
        "@@BALANCE_B@@": _balance_deflection(model.mechanism_angles[BETA_M].positive_airspeed_toward),
    }


def _positive_sentence(name: str, toward: str, kind: str) -> str:
    target = _direction(toward)
    if name == ALPHA:
        if target not in {"downward on the airframe", "upward on the airframe"}:
            raise RuntimeError(f"Theoretical angle of attack is not an up/down direction: {target}")
        sign = "positive" if target == "downward on the airframe" else "negative"
        return (
            "Theoretical angle of attack α: the angle between the projection of the airspeed onto the aircraft plane of symmetry and the positive axial direction of the airframe. "
            f"The angle is {sign} when that projection lies below the airframe axis."
        )
    if name == BETA:
        if target not in {"right on the airframe", "left on the airframe"}:
            raise RuntimeError(f"Theoretical sideslip is not a left/right direction: {target}")
        sign = "positive" if target == "right on the airframe" else "negative"
        return (
            "Theoretical sideslip β: the angle between the airspeed vector and the aircraft plane of symmetry. "
            f"The angle is {sign} when the airspeed lies to the right of the aircraft."
        )
    if kind == "airspeed":
        return f"{ANGLE_NAME[name]}: positive when the airspeed points {target}."
    if kind == "wing":
        return f"{ANGLE_NAME[name]}: positive when the right wing points {target}."
    raise KeyError(kind)


ROD_ORDER_PARAGRAPH = (
    "3. From the mechanism layout, the first rotation angle is @@16@@. "
    "The second rotation angle is @@17@@. The third rotation angle is @@18@@.\n\n"
)
WRIST_ORDER_PARAGRAPH = (
    "3. This item executes the mechanism attitude angles in the order yaw, pitch, roll. "
    "The first rotation angle is @@16@@. The second rotation angle is @@17@@. "
    "The third rotation angle is @@18@@.\n\n"
)


def _process_template(model: BodyFrameModel, process: str) -> str:
    """Stock subprocess text, with the wrist order stated when that mechanism is declared."""
    template = PROCESS_TEXT[process]
    if process == "wind frame to body-axis frame" and tuple(model.euler_order) == WRIST_ORDER:
        if template.count(ROD_ORDER_PARAGRAPH) != 1:
            raise RuntimeError("The original mechanism-order sentence does not appear exactly once.")
        return template.replace(ROD_ORDER_PARAGRAPH, WRIST_ORDER_PARAGRAPH, 1)
    return template


def _mechanism_lines(model: BodyFrameModel) -> list[str]:
    """Angle-mechanism sentences. The wrist names an order; the rod leaves it implicit."""
    mechanism = model.mechanism_angles
    alpha_side = _balance_deflection(mechanism[ALPHA_M].positive_airspeed_toward)
    beta_side = _balance_deflection(mechanism[BETA_M].positive_airspeed_toward)
    gamma_word = (
        "same"
        if roll_about_nose_sense(mechanism[GAMMA_M].right_wing_toward) == SENSE_SAME
        else "opposite"
    )
    if tuple(model.euler_order) == WRIST_ORDER:
        return [
            "The mechanism angles are executed by a three-degree-of-freedom moving-axis spherical wrist. The three axes, in order, are yaw, pitch, and roll, and the execution order is yaw, pitch, roll. Yaw is the mechanism sideslip β_M, pitch is the mechanism angle of attack α_M, and roll is the mechanism roll γ_M.",
            f"Mechanism angle of attack α_M: the pitch rotation of the wrist, about the absolute lateral axis. Positive when the balance deflects {alpha_side} in the tunnel.",
            f"Mechanism sideslip β_M: the yaw rotation of the wrist, about the absolute normal axis. Positive when the balance deflects {beta_side} in the tunnel.",
            (
                "Mechanism roll γ_M: the roll rotation of the wrist, about the positive absolute model axis (forward). "
                f"When the angle is positive, its sense is {gamma_word} the right-hand rule."
            ),
        ]
    return [
        (
            "Mechanism angle of attack α_M: the pitch sting moves in the vertical plane and may be treated as a rotation about the lateral axis of the wind frame. "
            "It is the angle between the sting (the balance axis) and the horizontal plane. "
            f"Positive when the balance deflects {alpha_side} in the tunnel."
        ),
        (
            "Mechanism sideslip β_M: the sideslip sting moves in the plane that is perpendicular to the vertical plane and contains the pitch sting, "
            "so the sideslip mechanism is driven on top of the pitch mechanism. "
            "β_M is the angle between the sideslip sting and the vertical plane. "
            f"Positive when the balance deflects {beta_side} in the tunnel."
        ),
        (
            "Mechanism roll γ_M: the rotation of the airframe about the sideslip sting. "
            "When the angle is positive, the rotation is about the positive absolute model axis (forward) and is "
            f"{gamma_word} the right-hand rule."
        ),
    ]


def _frame_lines(title: str, names: tuple[str, ...], frame: dict) -> list[str]:
    lines = [title]
    for name in names:
        axis = frame[name]
        lines.append(
            f"- {name}: along {AXIS_ZH[axis.absolute_axis]}, pointing {_direction(axis.points_to)} ({SOURCE_ZH[axis.source]})."
        )
    return lines


def _declaration(model: BodyFrameModel) -> str:
    angles = model.angles
    elastic = model.elastic_angles
    flow = model.flow_angles
    lines = [
        "Fixed conventions",
        "- Every coordinate transform is passive. Every airframe attitude rotation is an intrinsic moving-axis rotation.",
        "- Forces and moments stay tied to the absolute axes: CN is normal, CZ is lateral, and CA is axial. MZ (pitch) is lateral, MA (roll) is axial, and MN (yaw) is normal. The body axes and the balance use the same correspondence.",
        "- The body-axis origin is the balance center. L is the length from the model nose to the model base, L_t is the length from the model base to the sting base, and L_r is the length from the balance center to the sting base. The three lengths are collinear, and the center lies between the nose and the sting base. Xr is the distance from the balance center to the model nose.",
        "- In engineering coordinates the axial component is measured rearward from the nose to the reference point, so that engineering positive direction is opposite the positive absolute axis (forward). The other two components follow the positive body-axis coordinate axes and are not rewritten as the absolute normal (upward) or the absolute lateral (right). Forces act at the origin, and moments are transferred to the reference point. An extra moment is written only as C·L_c/L_RT.",
        "",
        "Absolute directions",
        "The airframe, the tunnel, and the balance each have three absolute axes. The three sets use the same positive directions: axial is forward, normal is upward, and lateral is to the right. The six absolute directions are the two senses of these three axes.",
        "Wherever this item says positive axial, positive normal, or positive lateral, and wherever it says the positive model axis, the positive airframe axis, the positive balance axis, or the positive tunnel axis, it means these absolute axes, not the positive directions of the coordinate axes X, Y, and Z. An axis declaration only says which absolute axis a coordinate axis follows and which side it points to. It may agree with the absolute positive direction or oppose it.",
        "The three frames share one cross-product relation: forward × right = downward. Looking forward, right is on the right-hand side and downward is vertically down. This relation does not change with the declaration.",
        "The six directions of the airframe are fixed on the airframe itself. Upright and inverted installation do not rewrite them:",
        "- Axial: the line from the tail to the nose. The positive sense is forward, toward the nose. Rearward points toward the tail.",
        "- Normal: the up-down axis of the airframe. The positive sense is upward. Downward is the opposite sense.",
        "- Lateral: the left-right axis of the airframe. The positive sense is to the right. Left is the opposite sense.",
        "The six directions of the tunnel are fixed on the tunnel:",
        "- Axial: the tunnel centerline. The positive sense is forward. Forward is the freestream, the same direction as the airspeed, and the opposite of the flow direction. Rearward is the opposite sense.",
        "- Normal: the vertical axis of the tunnel. The positive sense is upward, vertically up. Downward is vertically down.",
        "- Lateral: the horizontal cross axis of the tunnel. The positive sense is to the right. Left is the opposite sense.",
        "The six directions of the balance coincide with the tunnel. They are not given a separate sense, and they do not flip when the model is mounted upright or inverted. Forward, rearward, upward, downward, right, and left on the balance are the same directions on the tunnel. The balance axial, normal, and lateral axes are the tunnel axial, normal, and lateral axes. The positive absolute balance axis is forward, the positive absolute normal is upward, and the positive absolute lateral is to the right. The positive directions of the balance axes XTP, YTP, and ZYP are given by the axis declaration below. They may agree with the absolute positive directions or oppose them.",
        "",
        "Scene",
        "In the initial pose the model is level in the tunnel and its forward direction faces the freestream. Installation is either upright or inverted. Upright puts the top of the model toward the top of the tunnel; inverted puts the top of the model toward the bottom of the tunnel. The balance is inside the model. Its tail leaves the model through a sting and connects to the tunnel angle mechanism. When the mechanism moves, the sting carries the balance and the model together. When the model is upright and level, the top of the model agrees with upward in the tunnel. When the model is inverted and level, the top of the model agrees with downward in the tunnel. The sting supports the balance and can deform elastically, which produces elastic angles. The tunnel flow is not perfectly aligned with the tunnel axis, which produces flow-offset angles. Both the elastic angles and the flow-offset angles change the actual theoretical angle of attack, sideslip, and roll, and they are corrected.",
        "",
        "Angles",
        _positive_sentence(ALPHA, angles[ALPHA].positive_airspeed_toward, "airspeed"),
        _positive_sentence(BETA, angles[BETA].positive_airspeed_toward, "airspeed"),
        _positive_sentence(GAMMA, angles[GAMMA].right_wing_toward, "wing"),
        *_mechanism_lines(model),
        "",
        "Installation",
        (
            f"- Model installation: {'upright' if model.placement == 'upright' else 'inverted'}. "
            "When the model is upright, upward, downward, left, right, and forward on the airframe agree with the same directions in the tunnel. "
            "When the model is inverted, upward on the airframe points downward in the tunnel, downward on the airframe points upward in the tunnel, left and right are swapped, and forward on the airframe still agrees with forward in the tunnel."
        ),
        (
            "- Balance installation angle γ_tp: the normal departs from the vertical plane through the balance axis. "
            f"The positive sense is {'the same as' if model.installation.sense == 'same' else 'opposite'} the right-hand rule about the positive absolute balance axis (forward)."
        ),
        (
            "- Model installation angle γ_model: the normal departs from the vertical plane through the airframe axis. "
            f"The positive sense is {'the same as' if model.model_installation.sense == 'same' else 'opposite'} the right-hand rule about the positive absolute airframe axis (forward)."
        ),
        "",
        "Elastic angles and flow-offset angles",
        f"- Elastic angle of attack α_e: positive when the balance axis bends toward {_direction(elastic[ALPHA_E].positive_toward)}.",
        f"- Elastic sideslip β_e: positive when the balance axis bends toward {_direction(elastic[BETA_E].positive_toward)}.",
        f"- Elastic roll γ_e: positive when the balance lateral axis moves toward {_direction(elastic[GAMMA_E].positive_toward)}.",
        "- Elastic angle-of-attack increment Δα_e: after the balance deforms, the angle between the tunnel axis and the projection of the balance axis onto the vertical plane through the tunnel axis.",
        "- Elastic sideslip increment Δβ_e: after the balance deforms, the angle between the balance axis and the vertical plane through the tunnel axis.",
        f"- Flow-offset angle α_V: positive when the model is upright and level and the airspeed points {_direction(flow[ALPHA_V].positive_airspeed_toward)}.",
        f"- Flow-offset angle β_V: positive when the model is upright and level and the airspeed points {_direction(flow[BETA_V].positive_airspeed_toward)}.",
        "",
    ]
    lines.extend(_frame_lines(
        f"Body-axis frame ({HAND_ZH[model.chirality]})",
        BODY_AXES,
        model.body_frame,
    ))
    lines.append("")
    lines.extend(_frame_lines(
        f"Wind frame ({HAND_ZH[model.wind_chirality]})",
        WIND_VOCAB.coordinate_axes,
        model.wind_frame,
    ))
    lines.append("")
    lines.extend(_frame_lines(
        f"Balance frame ({HAND_ZH[model.balance_chirality]})",
        BALANCE_VOCAB.coordinate_axes,
        model.balance_frame,
    ))
    return "\n".join(lines)


_DECLARATION_REQUIRED = (
    "Theoretical angle of attack α:",
    "Theoretical sideslip β:",
    "theoretical roll γ:",
    "Mechanism angle of attack α_M:",
    "Mechanism sideslip β_M:",
    "Mechanism roll γ_M:",
    "Model installation:",
    "Balance installation angle γ_tp:",
    "Model installation angle γ_model:",
    "Elastic angle of attack α_e:",
    "Elastic sideslip β_e:",
    "Elastic roll γ_e:",
    "Elastic angle-of-attack increment Δα_e:",
    "Elastic sideslip increment Δβ_e:",
    "Flow-offset angle α_V:",
    "Flow-offset angle β_V:",
    "Body-axis frame (",
    "- XTZ:",
    "- YTZ:",
    "- ZTZ:",
    "Wind frame (",
    "- XFD:",
    "- YFD:",
    "- ZFD:",
    "Balance frame (",
    "- XTP:",
    "- YTP:",
    "- ZYP:",
)
_DECLARATION_FORBIDDEN = ("definition change", "kept at the common default", "not counted", "changed to", "is not a variable definition")



def _check_declaration(declaration: str) -> None:
    missing = [item for item in _DECLARATION_REQUIRED if item not in declaration]
    if missing:
        raise RuntimeError(f"The declaration is missing: {', '.join(missing)}")
    present = [item for item in _DECLARATION_FORBIDDEN if item in declaration]
    if present:
        raise RuntimeError(f"The declaration still contrasts a change: {', '.join(present)}")


def _output_format(process: str, item: dict) -> str | None:
    if item["options"]:
        return None
    answer = item["answer"]
    if answer.startswith("[["):
        return MATRIX_FORMAT
    if answer.startswith("["):
        return VECTOR_FORMAT
    if "L_RT" in answer:
        return MOMENT_FORMAT
    if "arctan" in answer:
        return ARCTAN_FORMAT
    if "arcsin" in answer:
        return ARCSIN_FORMAT
    if process == "moment reference point" and item["manuscript_id"] == "2":
        return DISTANCE_FORMAT
    return EXPRESSION_FORMAT


def _question(process: str, item: dict) -> dict:
    output_format = _output_format(process, item)
    if item["options"] and item["answer"] not in item["options"]:
        raise RuntimeError(
            f"The answer {item['answer']!r} of {process} ID {item['manuscript_id']} is not among the options."
        )
    question = {
        "process": process,
        "id": _public_id(process, item["manuscript_id"]),
        "local_id": item["manuscript_id"],
        "prompt": item["prompt"],
        "options": item["options"],
        "output_format": output_format,
        "format_code": "choice" if output_format is None else FORMAT_CODE[output_format],
        "label": item["label"],
        "answer": answer_letter(item["answer"], item.get("options") or []),
    }
    question.update(question_fields(process, question["id"], question["label"]))
    return question


def _options_text(item: dict) -> str:
    options = item.get("options") or []
    return format_choices([str(option) for option in options])


def _gold_phrase(item: dict) -> str:
    """The object named by a stored letter, used when a prior blank is filled in."""
    answer = str(item["answer"])
    options = [str(option) for option in (item.get("options") or []) if option and str(option) not in {"none"}]
    if not options:
        return answer
    _letter, text = resolve_choice(answer, options)
    return text


def _review_slot(item: dict) -> str:
    mention = ""
    if item.get("rule_mention"):
        kind = "recall" if item["rule_mention"] == "recall" else "method"
        mention = f", rule-mention:{kind}/{item.get('rule_definition_name') or item.get('rule_definition')}"
    return (
        f"{{ID:{item['id']}, options:{_options_text(item) or 'none'}, format:{item['format_code']}, "
        f"label:{item['label']}, answer:{item['answer']}{mention}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, options:{_options_text(item) or 'none'}, format:{item['format_code']}}}"


def _prepared_template(process: str, phrases: dict[str, str] | None) -> str:
    template = PROCESS_TEXT[process]
    for token, text in (phrases or {}).items():
        template = template.replace(token, text)
    return template


def _period_index(tail: str) -> int:
    """Index of the sentence-ending period, or -1."""
    for index, char in enumerate(tail):
        if char == "。":
            return index
        if char != ".":
            continue
        nxt = tail[index + 1] if index + 1 < len(tail) else ""
        if nxt in ("", " ", "\n", "\t"):
            return index
    return -1


def _sentence_break(chunk: str) -> bool:
    """True when chunk contains a sentence end.

    A Chinese period or an ASCII period followed by whitespace or the end of
    the chunk closes a sentence. A period inside a number does not.
    """
    if "。" in chunk:
        return True
    for index, char in enumerate(chunk):
        if char != ".":
            continue
        nxt = chunk[index + 1] if index + 1 < len(chunk) else ""
        if nxt in ("", " ", "\n", "\t"):
            return True
    return False


def _sentence_groups(
    process: str,
    phrases: dict[str, str] | None = None,
    template: str | None = None,
) -> list[list[str]]:
    """Blank ids that share a sentence, in manuscript order.

    A sentence ends at 。. Blanks with no period between them are answered
    together. ``template`` replaces the stock subprocess text for an ablation.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    cursor = 0
    if template is None:
        template = _prepared_template(process, phrases)
    else:
        for token, text in (phrases or {}).items():
            template = template.replace(token, text)
    for match in SLOT_MARK.finditer(template):
        if _sentence_break(template[cursor:match.start()]) and current:
            groups.append(current)
            current = []
        current.append(match.group(1))
        cursor = match.end()
    if current:
        groups.append(current)
    return groups


def _same_blank_ids(process: str, seen: list[str], expected: list[str]) -> None:
    """The manuscript may ask blanks in a different order from their numbers."""
    if len(seen) != len(set(seen)):
        raise RuntimeError(f"The {process} problem text repeats a blank.")
    if sorted(seen, key=int) != sorted(expected, key=int):
        missing = [item for item in expected if item not in set(seen)]
        extra = [item for item in seen if item not in set(expected)]
        raise RuntimeError(f"The {process} blanks do not match the generated blanks; missing {missing}, extra {extra}.")


def _render_process(
    process: str,
    by_id: dict[str, dict],
    target: str | None = None,
    phrases: dict[str, str] | None = None,
    targets: list[str] | None = None,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
    template: str | None = None,
) -> str:
    """Render the manuscript.

    target=None keeps every review slot. Otherwise the text runs from the
    start of the subprocess through the period of the sentence that contains
    the target. Every blank in that sentence is an open slot. Earlier blanks
    are gold answers, unless memory supplies the raw strings to put there.
    Later sentences are omitted unless keep_suffix is set; those later blanks
    stay open and are not filled with gold.
    """
    if template is None:
        template = _prepared_template(process, phrases)
    elif phrases:
        for token, text in phrases.items():
            template = template.replace(token, text)
    if targets is None and target is not None:
        targets = next(
            group for group in _sentence_groups(process, phrases, template) if target in group
        )
    open_ids = list(targets or [])
    open_set = set(open_ids)
    last_open = open_ids[-1] if open_ids else None
    pieces: list[str] = []
    cursor = 0
    seen: list[str] = []
    seen_open = False
    for match in SLOT_MARK.finditer(template):
        blank_id = match.group(1)
        if blank_id not in by_id:
            raise KeyError(f"{process} has no answer for ID {blank_id}.")
        seen.append(blank_id)
        pieces.append(template[cursor:match.start()])
        item = by_id[blank_id]
        if not open_set:
            pieces.append(_review_slot(item))
        elif blank_id in open_set:
            seen_open = True
            pieces.append(_open_slot(item))
            if blank_id == last_open and not keep_suffix:
                tail = template[match.end():]
                period = _period_index(tail)
                if period == -1:
                    raise RuntimeError(f"The sentence of {process} ID {blank_id} has no period.")
                if SLOT_MARK.search(tail[:period]):
                    raise RuntimeError(f"The sentence of {process} ID {blank_id} still has a blank that was not included.")
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            if not keep_suffix:
                raise RuntimeError(f"{process} ID {blank_id} appears after the open slots of the current sentence.")
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_phrase(item) if memory is None else memory.get(item["id"], ""))
        cursor = match.end()
    if open_ids:
        if not seen_open:
            missing = target if target is not None else ", ".join(open_ids)
            raise KeyError(f"{process} has no sentence for {missing}.")
        pieces.append(template[cursor:])
        return "".join(pieces).strip()
    pieces.append(template[cursor:])
    expected = [item.get("local_id", item["id"]) for item in by_id.values()]
    _same_blank_ids(process, seen, expected)
    return "".join(pieces).strip()


def _json_field(item: dict, example: bool = False, letter: str = "A") -> str:
    blank_id = item["id"]
    if item.get("options"):
        return f"{blank_id}:{letter if example else 'letter'}"
    if example:
        shown = DELTA_ALPHA_QUOTIENT_EXAMPLE if _is_delta_alpha_quotient(item) else FORMAT_EXAMPLE[item["format_code"]]
        return f'{blank_id}:"{shown}"'
    return f'{blank_id}:"answer"'


def _json_object(items: list[dict], example: bool = False) -> str:
    fields = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_index = 0
    for item in items:
        letter = letters[choice_index % len(letters)]
        if item.get("options"):
            choice_index += 1
        fields.append(_json_field(item, example, letter))
    return "{" + ", ".join(fields) + "}"


def _output_preamble() -> list[str]:
    return [
        "[Output]",
        "Put every answer in one JSON object. The key is the blank id and the value is the answer, for example {1:A}.",
        "This is a step-by-step question: one sentence produces one JSON object, and every blank in that sentence is written in that object.",
        "Only an end-to-end question puts every answer of the whole subprocess in one JSON object. Here, write only the current sentence.",
        "For a multiple-choice blank, write only the letter, with no quotation marks. A formula, matrix, vector, expression, or distance must be a string in English double quotes.",
    ]


def _is_delta_alpha_quotient(item: dict) -> bool:
    """The substituted Δα_e in balance frame to wind frame. This blank must be a quotient of wind-frame components and does not accept an elastic-angle expression."""
    return item.get("process") == "balance frame to wind frame" and str(item.get("local_id")) == "58"


def _formula_note(item: dict) -> str:
    if _is_delta_alpha_quotient(item):
        return DELTA_ALPHA_QUOTIENT_NOTE
    return ANGLE_SYMBOL_NOTE


def _output_block(item: dict) -> str:
    lines = _output_preamble()
    lines.append(f"Write only one line {_json_object([item])}.")
    if item.get("options"):
        lines.append("Do not write the option text, a period, an explanation, Markdown, or LaTeX.")
    else:
        lines.extend([
            "The double quotes only mark a JSON string; they are not part of the answer. Do not add an explanation, Markdown, or LaTeX outside the quotes.",
            _formula_note(item),
            DELTA_ALPHA_QUOTIENT_FORMAT if _is_delta_alpha_quotient(item) else FORMAT_TEXT[item["format_code"]],
        ])
    lines.append(f"Format example (not the answer to this blank): {_json_object([item], example=True)}")
    return "\n".join(lines)


def _output_block_many(by_id: dict[str, dict], blank_ids: list[str]) -> str:
    items = [by_id[blank_id] for blank_id in blank_ids]
    lines = _output_preamble()
    lines.extend([
        f"Write only one line {_json_object(items)}. Keep the blanks in id order inside the same object, and do not split them into several JSON objects.",
        "Do not write the sentence text.",
    ])
    if not any(_is_delta_alpha_quotient(item) for item in items):
        lines.append(ANGLE_SYMBOL_NOTE)
    for item in items:
        if item.get("options"):
            lines.append(f"The value of ID {item['id']} is an option letter.")
            continue
        code = item["format_code"]
        if _is_delta_alpha_quotient(item):
            lines.append(
                f"The value of ID {item['id']} is a {code} and must be a string in English double quotes. "
                f"{DELTA_ALPHA_QUOTIENT_NOTE}{DELTA_ALPHA_QUOTIENT_FORMAT}"
            )
        else:
            lines.append(
                f"The value of ID {item['id']} is a {code} and must be a string in English double quotes. {FORMAT_TEXT[code]}"
            )
    lines.append(f"Format example (not the answer to these blanks): {_json_object(items, example=True)}")
    return "\n".join(lines)


def _injection(
    declaration: str,
    process: str,
    by_id: dict[str, dict],
    target: str,
    phrases: dict[str, str] | None = None,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
    template: str | None = None,
) -> str:
    local_target = target if template is not None else _local_id(process, target)
    group = next(item for item in _sentence_groups(process, phrases, template) if local_target in item)
    if len(group) == 1:
        item = by_id[group[0]]
        lead = (
            f"Fill only the current sentence. This is a step-by-step question. Write one JSON object {_json_object([item])}. "
            "Do not explain, and do not fill any other blank."
        )
        output = _output_block(item)
    else:
        listed = ", ".join(group)
        lead = (
            f"Fill every blank in this sentence. The ids are {listed}. "
            "These blanks belong to one sentence, so answer the whole sentence at once. "
            f"This is a step-by-step question. Write one JSON object {_json_object([by_id[blank_id] for blank_id in group])}. "
            "Put every blank of this sentence in that one object. Do not split them into several objects. "
            "A multiple-choice value is a letter, with no quotation marks. A formula, matrix, vector, expression, or distance is a string in English double quotes. "
            "Do not write the sentence text, and do not explain."
        )
        output = _output_block_many(by_id, group)
    if keep_suffix:
        lead += " Text after the period is context only. Do not fill those blanks."
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        declaration,
        "",
        "[Background]",
        PRIOR,
        "",
        "[Subprocess]",
        process,
        "",
        "[Process]",
        _render_process(
            process, by_id, phrases=phrases, targets=group, memory=memory,
            keep_suffix=keep_suffix, template=template,
        ),
        "",
        output,
    ])


def _injection_end_to_end(
    declaration: str,
    process: str,
    by_id: dict[str, dict],
    phrases: dict[str, str] | None = None,
    template: str | None = None,
) -> str:
    """One call for a whole subprocess. Every blank stays open."""
    blank_ids = [
        blank_id
        for group in _sentence_groups(process, phrases, template)
        for blank_id in group
    ]
    lead = (
        "This is an end-to-end question. Fill every blank in this subprocess in one call. "
        "Write one JSON object. The key is the blank id and the value is the answer. "
        "The object must contain every id in this subprocess. Do not split it into several objects, and do not answer sentence by sentence. "
        "A multiple-choice value is a letter, with no quotation marks. A formula, matrix, vector, expression, or distance is a string in English double quotes. "
        "Do not write the sentence text, and do not explain."
    )
    output = "\n".join([
        "[Output]",
        "Put every answer in the same JSON object. The key is the blank id and the value is the answer.",
        'For example {1:A, 2:C, 11:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}. The example is not the answer.',
        "A multiple-choice value is only a letter, with no quotation marks. A formula, matrix, vector, expression, or distance must be a string in English double quotes.",
        "Write only that one line of JSON. Do not add an explanation, Markdown, or LaTeX.",
    ])
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        declaration,
        "",
        "[Background]",
        PRIOR,
        "",
        "[Subprocess]",
        process,
        "",
        "[Process]",
        _render_process(
            process, by_id, phrases=phrases, targets=blank_ids,
            keep_suffix=True, template=template,
        ),
        "",
        output,
    ])


def _render_markdown(record: dict) -> str:
    lines = [
        f"# Data {record['id']} {record['title']}",
        "",
        "This file is a review manuscript. It shows one complete process. Evaluation answers one sentence at a time and does not send the whole process at once. The input keeps the declaration, the background, the current subprocess name, and the text from the start of that subprocess through the period of the current sentence. Blanks before that sentence are replaced by the option text, for example XFD or upright, not by a letter. Every blank in that sentence stays an open slot and is filled together. Text after the period is not sent. An open slot has no label and no answer. Labels are only for scoring. The model writes every blank of that sentence as one JSON object whose key is the id and whose value is the answer.",
        "",
        "Open-slot format: {ID:id, options:A: option1; B: option2, output format:format name}. When there are no options, write options:none. The output-format names are explained in data/wind_tunnel/review/00_injection_format.md.",
        "",
        "## Declaration",
        "",
        record["declaration"],
        "",
        "## Background",
        "",
        PRIOR,
        "",
        "## Full process",
        "",
        "Each slot below also shows its label and gold answer so it can be reviewed directly. Those are review fields. The model does not see them on an open slot.",
        "",
    ]
    for process, _generate in PROCESSES:
        lines.extend([f"### {process}", "", record["processes"][process], ""])
    return "\n".join(lines)


def build_record(sample: dict) -> dict:
    model = BodyFrameModel().declare(deepcopy(sample["spec"]))
    questions = []
    processes = {}
    phrases = _mechanism_phrases(model)
    for process, generate in PROCESSES:
        payload = generate(deepcopy(sample["spec"]))
        group = [_question(process, item) for item in payload["items"]]
        by_id = {item["local_id"]: item for item in group}
        processes[process] = _render_process(
            process, by_id, phrases=phrases, template=_process_template(model, process),
        )
        questions.extend(group)
    removed = sum(len(slots) for slots in REMOVED_SLOTS.values())
    expected = sum(PROCESS_SIZE.values()) - removed
    if len(questions) != expected:
        raise RuntimeError(f"{sample['id']} produced {len(questions)} questions, expected {expected}.")
    declaration = _declaration(model)
    _check_declaration(declaration)
    return {
        "id": sample["id"],
        "title": sample["title"],
        "change": sample["change"],
        "spec": sample["spec"],
        "declaration": declaration,
        "prior": PRIOR,
        "processes": processes,
        "questions": questions,
        "injection_example": _injection(
            declaration,
            "wind frame to body-axis frame",
            {item["local_id"]: item for item in questions if item["process"] == "wind frame to body-axis frame"},
            "22",
            phrases,
        ),
    }


def _format_document(example: str) -> str:
    lines = [
        "# Formal injection format",
        "",
        "One call scores one sentence. When that sentence has several blanks, they are filled together and are not split into independent blanks. The input has five parts, in a fixed order.",
        "",
        "1. Output requirement: put every answer in one JSON object. The key is the blank id and the value is the answer, for example {1:A}. A step-by-step question covers one sentence: when that sentence has one blank, output {1:A}; when it has several blanks, output one object, for example {1:A, 2:C}. Do not split them into several objects. Only an end-to-end question puts every answer of the whole process into the same object. A choice value is a letter, with no quotation marks. A formula, matrix, vector, expression, or distance must be a string in English double quotes, for example {11:\"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]\"}. Do not output the sentence and do not explain.",
        "2. [Declaration] The full declaration of this row.",
        "3. [Background] The fixed text below, attached every time.",
        "4. [Subprocess] The name of the subprocess that contains the current sentence.",
        "5. [Process] Text from the start of that subprocess through the period of the current sentence. Blanks before that sentence are replaced by the option text, for example XFD or upright, not by letters such as A or B. Every slot in that sentence is an open slot. Text after the period does not appear. This is condition A. The gold used for scoring is still the letter.",
        "",
        "An open slot has only three fields: {ID:id, options:A: option1; B: option2, output format:format name}. When there are no options, write options:none. An open slot contains no label and no answer.",
        "",
        "Each injection appends [Output] after the process. It states the JSON form of every open slot and gives an output example. The example only shows how the characters are arranged; it is not the answer to that slot. When there are options, the open slot writes them as A: option1; B: option2, and the model outputs {id:letter}. Scoring reads the JSON value. The correct letter, the option text, and a fragment that uniquely matches the correct option all count. A formula answer is compared after the outer double quotes are removed.",
        "",
        "Each review manuscript shows the whole process. Every slot there also writes a label and an answer, and those two fields are for review. A person can read it this way: the sentence being scored has every slot turned into an open slot, earlier slots are replaced by answers, and the text ends at the period.",
        "",
        "The four sentence-by-sentence conditions share one template and one sentence order, and each subprocess starts again at its beginning. A writes the option text before the blank and cuts at the period. B replaces the text before the blank with the model's raw output from the previous sentence, without correction or normalization; a failed call or an empty return is also placed into the next sentence as it is. C' still writes the option text before the blank and keeps the later text after the period, whose blanks stay open and are not prefilled. C'' uses the model's own raw output before the blank and also keeps the later text. A and B differ only in the prefix source. A and C' differ only in whether later text is kept.",
        "",
        "Blank ids follow the subprocess order. wind frame to body-axis frame uses 1-81, and the three component-matrix expansions 52, 53, and 54 were removed. balance frame to body-axis frame uses 82-116, and the three component-matrix expansions 101, 102, and 103 were removed. balance frame to wind frame uses 117-180. moment reference point uses 181-213. 201-213 are the numerator and denominator probes of the moment; they are written after the sign sentence and before the formula sentence. 195-200 are still the six moment formulas. Removed blank numbers are not shifted.",
        "",
        "D is end to end. Each subprocess is one call. The text is kept from the start through the end, every blank is an open slot, and no answer is prefilled. The model outputs one JSON object, and the keys use the continuous ids above.",
        "",
        "## Background",
        "",
        PRIOR,
        "",
        "## Output format names",
        "",
    ]
    for code, text in FORMAT_TEXT.items():
        lines.append(f"- {code}: {text} Example: {FORMAT_EXAMPLE[code]}")
    lines.extend([
        "",
        "## One real injection",
        "",
        "Below is the full input for data 01, subprocess wind frame to body-axis frame, ID 22. This A1 is its own sentence. Other sentences are cut by the same rule.",
        "",
        "```text",
        example,
        "```",
        "",
    ])
    return "\n".join(lines)


def _check_injection_sentence(sample: dict, record: dict) -> None:
    """Each A blank is its own sentence, and no sentence mixes A with C."""
    spec = deepcopy(sample["spec"])
    model = BodyFrameModel().declare(spec)
    phrases = _mechanism_phrases(model)
    for process, generate in PROCESSES:
        payload = generate(deepcopy(sample["spec"]))
        labels = {item["manuscript_id"]: item["label"] for item in payload["items"]}
        covered = [blank_id for sentence in _sentence_groups(process, phrases) for blank_id in sentence]
        expected = [item["manuscript_id"] for item in payload["items"]]
        _same_blank_ids(process, covered, expected)
        for sentence in _sentence_groups(process, phrases):
            kinds = [labels[blank_id] for blank_id in sentence]
            families = {kind[0] for kind in kinds}
            if "A" in families and "C" in families:
                raise RuntimeError(f"{process} puts A and C in the same sentence: {list(zip(sentence, kinds))}")
            if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
                raise RuntimeError(f"An A blank of {process} is not in a sentence by itself: {list(zip(sentence, kinds))}")

    numbers = [item["id"] for item in record["questions"]]
    removed = {str(number) for slots in REMOVED_SLOTS.values() for number in slots}
    expected_numbers = [str(number) for number in range(1, 214) if str(number) not in removed]
    if numbers != expected_numbers:
        raise RuntimeError("The ids of the four subprocesses do not match the kept blank numbers.")
    wind_count = PROCESS_SIZE["wind frame to body-axis frame"] - len(REMOVED_SLOTS["wind frame to body-axis frame"])
    if record["questions"][wind_count]["id"] != "82" or record["questions"][wind_count]["process"] != "balance frame to body-axis frame":
        raise RuntimeError("balance frame to body-axis frame does not continue from 82.")
    if record["questions"][-1]["id"] != "213" or record["questions"][-1]["process"] != "moment reference point":
        raise RuntimeError("moment reference point does not end at 213.")
    group = {
        item["local_id"]: item
        for item in record["questions"]
        if item["process"] == "wind frame to body-axis frame"
    }

    def process_body(blank_id: str) -> str:
        text = _injection(record["declaration"], "wind frame to body-axis frame", group, blank_id, phrases)
        return text.split("[Process]\n", 1)[1].split("\n\n[Output]", 1)[0]

    sentence_22 = process_body("22")
    if not sentence_22.endswith(
        "When α_M > 0 the sign of the theoretical angle of attack is "
        "{ID:22, options:A: positive; B: negative; C: indeterminate, format:choice}."
    ):
        raise RuntimeError("ID 22 is not its own sentence.")
    if "{ID:21," in sentence_22:
        raise RuntimeError("The ID 22 injection still keeps ID 21 in the same sentence.")
    if process_body("23") == process_body("24"):
        raise RuntimeError("ID 23 and ID 24 are still injected together.")
    if "This is the level pose of intermediate body-axis frame." not in process_body("24") or "{ID:23," in process_body("24"):
        raise RuntimeError("The ID 24 prefix does not contain the ID 23 answer.")
    if "Later attitude transforms start from this intermediate body-axis frame, which keeps the airframe symbols XTZ, YTZ, and ZTZ and introduces no new symbols." not in process_body("16"):
        raise RuntimeError("The attitude rotation is not preceded by the intermediate body-axis frame.")
    if "through the angle α_M" in process_body("23"):
        raise RuntimeError("The ID 23 sentence still includes the rotation sense.")
    sentence_25 = process_body("25")
    if not sentence_25.endswith(
        "The sense about that axis is {ID:25, options:A: positive; B: negative; C: indeterminate, format:choice}, "
        "through the angle α_M, and the rotation matrix is R_α_M."
    ):
        raise RuntimeError("The ID 25 sentence does not name the rotation sense and the matrix.")
    if "{ID:26," in sentence_25 or "{ID:24," in sentence_25:
        raise RuntimeError("The ID 25 injection includes a slot from a neighboring sentence.")
    earlier = process_body("8")
    if "correspond to YFD, XFD, and ZFD" not in earlier or "correspond to YTZ, XTZ, and ZTZ" not in earlier:
        raise RuntimeError("Answers before the blank are not written as axis names.")
    if "correspond to A" in earlier or "correspond to D" in earlier:
        raise RuntimeError("Answers before the blank are still letters.")
    leaked = _injection(
        record["declaration"], "wind frame to body-axis frame", group, "22", phrases, keep_suffix=True,
    )
    if "This is the level pose" not in leaked or "{ID:23," not in leaked:
        raise RuntimeError("The kept suffix does not retain the text after ID 22.")
    if "Text after the period is context only" not in leaked:
        raise RuntimeError("The kept suffix does not say that later blanks stay closed.")
    remembered = _injection(
        record["declaration"], "wind frame to body-axis frame", group, "22", phrases, memory={"19": "MEMTEST"},
    )
    if "MEMTEST" not in remembered or "MEMTEST" in sentence_22:
        raise RuntimeError("The memory prefix did not replace the earlier answer.")
    if group["19"]["answer"] in remembered.split("MEMTEST", 1)[0].split("The model uses ", 1)[-1]:
        raise RuntimeError("The memory prefix still contains the gold answer of ID 19.")
    both = _injection(
        record["declaration"], "wind frame to body-axis frame", group, "22", phrases,
        memory={"19": "MEMTEST"}, keep_suffix=True,
    )
    if "MEMTEST" not in both or "{ID:23," not in both:
        raise RuntimeError("The memory-and-suffix injection dropped either the memory or the suffix.")
    end_to_end = _injection_end_to_end(record["declaration"], "wind frame to body-axis frame", group, phrases)
    if "This is an end-to-end question" not in end_to_end or "correspond to YFD, XFD, and ZFD" in end_to_end:
        raise RuntimeError("The end-to-end prompt still fills earlier answers.")
    if end_to_end.count("{ID:") != 78:
        raise RuntimeError(
            f"The end-to-end prompt did not leave every blank of this subprocess open; it now has {end_to_end.count('{ID:')}."
        )
    choice = _injection(record["declaration"], "wind frame to body-axis frame", group, "22", phrases)
    if "{22:letter}" not in choice or '{22:"' in choice:
        raise RuntimeError("A choice question did not ask for {22:letter}.")
    several = _injection(record["declaration"], "wind frame to body-axis frame", group, "2", phrases)
    if "{2:letter, 3:letter, 4:letter}" not in several or "Do not split them into several objects" not in several:
        raise RuntimeError("The blanks in one sentence were not asked for in one JSON object.")
    matrix = _injection(record["declaration"], "wind frame to body-axis frame", group, "8", phrases)
    if '{8:"[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]"}' not in matrix:
        raise RuntimeError("A matrix was not asked for inside JSON double quotes.")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    samples = _samples()
    records = [build_record(sample) for sample in samples]
    _check_injection_sentence(samples[0], records[0])
    stored = []
    for record in records:
        stored.append({key: value for key, value in record.items() if key != "injection_example"})
    (OUT_DIR / "samples.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index = [
        "# Nine check rows",
        "",
        "The formal injection format is in 00_injection_format.md. Each review manuscript gives the declaration and the full text of the four subprocesses, with the blanks left in the original sentences.",
        "",
    ]
    for record in records:
        path = OUT_DIR / f"sample_{record['id']}.md"
        path.write_text(_render_markdown(record), encoding="utf-8")
        index.append(f"- Data {record['id']} {record['title']}: {record['change']}")
    (OUT_DIR / "00_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (OUT_DIR / "00_injection_format.md").write_text(_format_document(records[0]["injection_example"]), encoding="utf-8")
    print(f"Wrote {len(records)} samples, {len(records[0]['questions'])} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
