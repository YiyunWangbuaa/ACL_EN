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
    ("风轴系到体轴系", generate_tz_fd),
    ("天平系到体轴系", generate_tp_tz),
    ("天平系到风轴系", generate_tp_fd),
    ("力矩参考点", generate_ref),
)
PROCESS_SIZE = {
    "风轴系到体轴系": 81,
    "天平系到体轴系": 35,
    "天平系到风轴系": 64,
    "力矩参考点": 20,
}
# 总矩阵前复写分矩阵的展开式已删除。空号留空，后面的编号不前移。
REMOVED_SLOTS = {
    "风轴系到体轴系": frozenset({52, 53, 54}),
    "天平系到体轴系": frozenset({101, 102, 103}),
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
    "body_forward": "机体前方",
    "body_rearward": "机体后方",
    "body_up": "机体上方",
    "body_down": "机体下方",
    "body_left": "机体左侧",
    "body_right": "机体右侧",
    "tunnel_forward": "风洞前方（来流）",
    "tunnel_rearward": "风洞后方",
    "tunnel_up": "风洞上方",
    "tunnel_down": "风洞下方",
    "tunnel_left": "风洞左侧",
    "tunnel_right": "风洞右侧",
    "balance_forward": "天平前方",
    "balance_rearward": "天平后方",
    "balance_up": "天平上方",
    "balance_down": "天平下方",
    "balance_left": "天平左侧",
    "balance_right": "天平右侧",
}
AXIS_ZH = {
    "body_axial": "机体轴向",
    "body_normal": "机体法向",
    "body_lateral": "机体侧向",
    "tunnel_axial": "风洞轴向",
    "tunnel_normal": "风洞法向",
    "tunnel_lateral": "风洞侧向",
    "balance_axial": "天平轴向",
    "balance_normal": "天平法向",
    "balance_lateral": "天平侧向",
}
ANGLE_NAME = {
    ALPHA: "理论迎角 α",
    BETA: "理论侧滑角 β",
    GAMMA: "理论滚转角 γ",
    ALPHA_M: "机构迎角 α_M",
    BETA_M: "机构侧滑角 β_M",
    GAMMA_M: "机构滚转角 γ_M",
}
HAND_ZH = {"right": "右手系", "left": "左手系"}
SOURCE_ZH = {"declared": "声明给出", "inferred": "由手性推断"}

MATRIX_FORMAT = (
    "输出一行3×3矩阵，形如[[a, b, c], [d, e, f], [g, h, i]]。"
    "逗号后恰好一个空格。元素为整数，或由cos、sin和本题角度符号组成的表达式。乘号用*。不要换行，不要使用LaTeX。"
)
VECTOR_FORMAT = (
    "输出一行三个分量，形如[a, b, c]。逗号后恰好一个空格。"
    "分量为整数，或由本题已出现的符号组成的表达式。乘号用*，三角函数写成cos(α_M)这种形式。"
    "不要换行，不要使用LaTeX。"
)
EXPRESSION_FORMAT = (
    "输出一行符号表达式。乘号用*，除号用/。三角函数写成cos(α_M)这种形式，括号内不留空格。"
    "负号直接写在项前，如-sin(α_M)。不要使用LaTeX，不要写等号。"
)
ANGLE_SYMBOL_NOTE = (
    "公式里的角度符号使用本题的α_M、β_M、γ_M、γ_tp、α_e、β_e。示例中的θ只展示格式，答案里不要写θ。"
)
DELTA_ALPHA_QUOTIENT_NOTE = (
    "这个空只能填风轴分量 XFD、YFD、ZFD 组成的分式。"
    "分子和分母都必须为正，负号只能写在整个分式之前。"
    "不能直接写弹性角式。答案里不能出现 α_e、β_e、γ_e 或 γ_tp。"
)
DELTA_ALPHA_QUOTIENT_FORMAT = (
    "输出一行分式。分子和分母都只能是 XFD、YFD、ZFD 之一，并且都为正；负号只放在整个分式之前。"
    "不要写三角函数，不要出现 α_e、β_e、γ_e 或 γ_tp。不要使用LaTeX，不要写等号。"
)
DELTA_ALPHA_QUOTIENT_EXAMPLE = "-(A)/(B)"
DISTANCE_FORMAT = (
    "输出一行由L、L_t、L_r用加号或减号连成的距离表达式。加号和减号两侧各留一个空格。"
    "不要使用LaTeX。"
)
ARCTAN_FORMAT = (
    "输出一行，且只能有一个arctan。分子和分母都必须为正；角度正负写在整个arctan外面。"
    "正角度写成arctan((分子)/(分母))，负角度写成-arctan((分子)/(分母))。"
    "不要让分子和分母同时为负，不要把负号放进内部。不要使用LaTeX。"
)
ARCSIN_FORMAT = (
    "输出一行，且只能有一个arcsin。分子必须为正，分母保持为正；角度正负写在整个arcsin外面。"
    "正角度写成arcsin(分子)，负角度写成-arcsin(分子)。"
    "不要让分子为负，不要把负号放进内部，不要写成分式。不要使用LaTeX。"
)
MOMENT_FORMAT = (
    "输出一行，形如-CN·Xg/L_RT。力系数与力臂之间用中点·。"
    "负号写在整个式子最外侧。力臂内部必须为正，表示额外力矩的绝对值，不要把负号写进力臂。"
    "力系数只能是CA、CZ、CN之一。力臂由Xg、Yg、Zg、Xr组成；力臂含减法时加圆括号。"
    "分母保持L_RT。不要留下字母C或L_c，不要写等号。不要使用LaTeX。"
)
FORMAT_CODE = {
    MATRIX_FORMAT: "矩阵",
    VECTOR_FORMAT: "向量",
    EXPRESSION_FORMAT: "表达式",
    DISTANCE_FORMAT: "距离",
    ARCTAN_FORMAT: "arctan",
    ARCSIN_FORMAT: "arcsin",
    MOMENT_FORMAT: "力臂",
}
FORMAT_TEXT = {
    "无": "选择题。值只写选项字母，不加引号，不写选项原文。",
    "矩阵": MATRIX_FORMAT,
    "向量": VECTOR_FORMAT,
    "表达式": EXPRESSION_FORMAT,
    "距离": DISTANCE_FORMAT,
    "arctan": ARCTAN_FORMAT,
    "arcsin": ARCSIN_FORMAT,
    "力臂": MOMENT_FORMAT,
}
FORMAT_EXAMPLE = {
    "无": "甲",
    "矩阵": "[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]",
    "向量": "[0, -1, 1]",
    "表达式": "cos(θ) - sin(θ)",
    "距离": "L - L_t + L_r",
    "arctan": "-arctan((cos(θ))/(sin(θ)))",
    "arcsin": "arcsin(cos(θ))",
    "力臂": "-CN·Xg/L_RT",
}
PRIOR = (
    "轴向正方向、法向正方向、侧向正方向分别是前方、上方、右侧。"
    "模型轴向的正方向、机体轴向的正方向、天平轴向正方向、风洞轴向的正方向都是这个绝对前方，不是坐标轴的正方向。"
    "单独变换迎角时，旋转轴是绝对侧向轴。单独变换侧滑角时，旋转轴是绝对法向轴。"
    "单独变换滚转角时，旋转轴是绝对轴向轴。各角的正方向以定义声明为准。"
    "坐标系转换采用被动变换。姿态角采用动轴内旋。"
)
SLOT_MARK = re.compile(r"@@(\d+)@@")
PROCESS_TEXT = {
    "风轴系到体轴系": """1.初始状态，体轴保持平正，为了计算机构角α_M、β_M、γ_M对补偿前的α_t、β_t和滚转角的影响，应当进行@@1@@转换。

2.由于风轴系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@2@@、@@3@@、@@4@@。机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@5@@、@@6@@、@@7@@。所以有风轴系向体轴系平正位置处的前置变换矩阵为RP1。前置变换矩阵RP1为@@8@@。该矩阵能够使@@9@@的坐标轴与自身绝对方向的关系与@@10@@的坐标轴与自身绝对方向的关系一致。此外，模型安装方式为@@11@@。此时模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞的绝对法向正方向和绝对侧向正方向@@12@@，对应的前置变换矩阵为RP2。前置变换矩阵RP2为@@13@@。该矩阵使得@@14@@的坐标轴方向与@@15@@的坐标方向相同。后续姿态变换时，坐标起点变为此时的中间体轴系，该坐标系依然沿用机体系的XTZ、YTZ、ZTZ，不做另行的符号设定。

3.根据机构结构，推断第一个旋转角为@@16@@。第二个旋转角为@@17@@。第三个旋转角为@@18@@。

4.当α_M>0时，天平向风洞@@BALANCE_A@@偏。模型采用@@19@@。α_M>0时，空速偏机体的@@20@@。理论迎角>0时，空速偏机体的@@21@@。α_M>0时，理论迎角的符号为@@22@@。这相当于@@23@@平正时。绕@@24@@旋转。绕该轴旋转方向为@@25@@，旋转α_M角度，对应的旋转矩阵为R_α_M。对应的旋转矩阵R_α_M为@@26@@。

5.当β_M>0时，天平向风洞@@BALANCE_B@@偏。模型采用@@27@@。β_M>0时，空速偏机体的@@28@@。理论侧滑角>0时，空速偏机体的@@29@@。β_M>0时，理论侧滑角的符号为@@30@@。这相当于@@31@@平正时。绕@@32@@旋转。绕该轴旋转方向为@@33@@，旋转β_M角度，对应的旋转矩阵为R_β_M。对应的旋转矩阵R_β_M为@@34@@。

6.当γ_M>0时，以模型绝对轴向正方向（前方）为旋转轴，与右手定则@@35@@。此时右翼偏机体的@@36@@。由于理论滚转角>0时，以机体绝对轴向正方向（前方）为旋转轴，与右手定则@@37@@。此时右翼偏机体的@@38@@。γ_M>0时，理论滚转角的符号为@@39@@。这相当于@@40@@平正时。绕坐标轴@@41@@旋转。绕该轴旋转方向为@@42@@，旋转γ_M角度，对应的旋转矩阵为R_γ_M。对应的旋转矩阵R_γ_M为@@43@@。

7.此过程涉及到坐标系转换，应当采用坐标系的@@44@@。此外体轴系是@@45@@。根据该坐标系类型，旋转时采用@@46@@。必须先变换坐标轴对应关系，之后再变换姿态。写出[XTZ,YTZ,ZTZ]^T=R1*R2*R3*R4*R5*[XFD,YFD,ZFD]^T。R1到R5从前面得到的矩阵代号中选择。公式中对应的R1矩阵为@@47@@。对应的R2矩阵为@@48@@。对应的R3矩阵为@@49@@。对应的R4矩阵为@@50@@。对应的R5矩阵为@@51@@。

8.写出R1*R2*R3符号化简后的总矩阵R_t，R_t=@@55@@。

9.由于空速矢量在平置机体系中的坐标与风轴系的坐标无关，故可以确定，RP2*RP1乘风轴系中的空速单位向量，结果一定是@@56@@。因此只需要研究@@57@@中的空速坐标到姿态角变化后的机体系的坐标映射即可。

10.计算[XTZ,YTZ,ZTZ]^T=R_t*[X_V,Y_V,Z_V]^T。此时平置坐标系中的空速单位向量[X_V,Y_V,Z_V]是@@58@@。代入公式计算得到的XTZ=@@59@@。计算得到的YTZ=@@60@@。计算得到的ZTZ=@@61@@。

11.当仅α_M>0时，推断XTZ为@@62@@。推断YTZ为@@63@@。推断ZTZ为@@64@@。

12.当仅β_M>0时，推断XTZ为@@65@@。推断YTZ为@@66@@。推断ZTZ为@@67@@。

13.此时α_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。其中α_t的角度符号为@@68@@。α_t的反三角函数符号为@@69@@。α_t的分子应使用的体轴分量为@@71@@。根据该分量在仅α_M>0时的符号，α_t分子的符号为@@70@@。α_t的分母应使用的体轴分量为@@73@@。根据该分量在仅α_M>0时的符号，α_t分母的符号为@@72@@。

14.此时β_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。其中β_t的角度符号为@@74@@。β_t的反三角函数符号为@@75@@。β_t的分子应使用的体轴分量为@@77@@。根据该分量在仅β_M>0时的符号，β_t分子的符号为@@76@@。β_t分母的符号为@@78@@。β_t的分母分量为@@79@@。

15.初始状态，体轴保持平正，已知来流在与风轴系轴向重合，吹向机头。当机构产生α_M、β_M、γ_M变化之后，机体处于一个新的姿态位置。现在需要根据已知量α_M、β_M、γ_M，推导新姿态位置下，机体的理论气流角α_t、β_t的表达式（不考虑天平形变和气流补偿）。这里给出公式模版：理论迎角α_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])，β_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。求得理论迎角α_t=@@80@@。求得理论侧滑角β_t=@@81@@。""",
    "天平系到体轴系": """1.由于天平系测量到的力和力矩需要转换到体轴系上，所以要进行@@1@@的转换。这里默认体轴系的原点和天平校心重合，坐标系只需旋转，无需平移。

2.先确定天平系和机体系的坐标轴与各自绝对方向的关系。天平系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@8@@、@@9@@、@@10@@。机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@11@@、@@12@@、@@13@@。所以有天平系向体轴系的前置矩阵为RTP_TZ1。前置矩阵RTP_TZ1是@@14@@。

3.再根据已经确定的轴向消除安装角带来的姿态差异。安装角旋转必须在天平原先定义的轴系中完成。天平安装角以天平绝对轴向正方向（前方）的右手定则@@2@@为正。机体滚转角以机体绝对轴向正方向（前方）的右手定则@@3@@为正。此时天平绝对轴向与机体绝对轴向的指向可能相同，也可能相反。机体滚转角γ_M的绝对值远大于天平安装角γ_tp。在天平绝对轴向、机体绝对轴向和两个正方向都已确定的条件下，天平系与机体系相差的滚转角度为@@4@@。该转角应绕天平绝对轴向所对应的坐标轴@@5@@。绕该轴旋转方向为@@6@@，对应的旋转矩阵为R_(γ_M-γ_tp)。对应的旋转矩阵R_(γ_M-γ_tp)为@@7@@。

4.然后再考虑模型正装或倒装，使坐标定义方向一致。如果倒装，机体绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞相反，还需要RTP_TZ2。在该定义下，模型是@@15@@的。因此该定义下的前置矩阵为RTP_TZ2。前置矩阵RTP_TZ2是@@16@@。

5.必须先变换姿态，之后再变换坐标轴对应关系。坐标转换公式为[XTZ,YTZ,ZTZ]^T=R1*R2*R3*[XTP,YTP,ZYP]^T。R1到R3从前面得到的矩阵代号中选择。其中R1=@@17@@。其中R2=@@18@@。其中R3=@@19@@。

6.写出R1*R2*R3符号化简后的总矩阵R，R=@@23@@。

7.进行力和力矩6分量转换时，XTZ与@@24@@关联，YTZ与@@25@@关联，ZTZ与@@26@@关联，XTP与@@27@@关联，YTP与@@28@@关联，ZYP与@@29@@关联。

8.将6分量带入转换矩阵，计算得到的体轴系6分量分别为CN_TZ=@@30@@。计算得到CA_TZ=@@31@@。计算得到CZ_TZ=@@32@@。计算得到MN_TZ=@@33@@。计算得到MA_TZ=@@34@@。计算得到MZ_TZ=@@35@@。""",
    "天平系到风轴系": """1.为了计算天平弹性角在风轴系产生的迎角和侧滑角增量，需要进行@@1@@转换。风洞绝对轴向正方向（前方）为@@2@@，天平绝对轴向正方向（前方）为@@3@@。由于风轴系的绝对法向正方向（上方）、绝对侧向正方向（右侧）、绝对轴向正方向（前方）分别对应@@4@@、@@5@@、@@6@@；天平的绝对法向正方向（上方）、绝对侧向正方向（右侧）、绝对轴向正方向（前方）分别对应@@7@@、@@8@@、@@9@@，对应的变换矩阵为RP_TP_FD。变换矩阵RP_TP_FD为@@10@@。此矩阵可以使@@11@@的坐标轴方向与@@12@@的坐标方向相同。这里需要先将天平旋转到平正位置，再考虑坐标方向的统一。因此RP_TP_FD的变换应该在天平旋转变换之后进行。

2.天平处于平置状态时，当产生γ_tp>0的安装角，应该绕天平的绝对@@13@@旋转。对应到坐标轴，应该绕坐标轴@@14@@。绕该轴旋转方向为@@15@@，旋转γ_tp角度，对应的旋转矩阵为R_γ_tp。对应的旋转矩阵R_γ_tp为@@16@@。

3.必须先变换姿态，之后再变换坐标轴对应关系。写出[XFD,YFD,ZFD]^T=R1*R2*[XTP,YTP,ZYP]^T中的R1和R2。R1和R2从前面得到的矩阵代号中选择。式中R1=@@17@@。式中R2=@@18@@。

4.在天平系中，天平沿天平绝对轴向正方向（前方）的单位向量为@@19@@。当天平产生α_e、β_e、γ_e时，天平轴向分别朝@@20@@弯曲、朝@@21@@弯曲。同时沿天平绝对轴向正方向（前方）看去发生@@22@@扭转。三个角度对应的旋转轴分别为@@23@@、@@24@@、@@25@@。形变后天平轴向单位向量在未变形原坐标系中的坐标为[XTP_e,YTP_e,ZTP_e]。其中XTP_e为@@26@@。其中YTP_e为@@27@@。其中ZTP_e为@@28@@。由于弹性角较小，可以用sinθ=θ、cosθ=1来近似。此时天平沿轴向的单位向量在原坐标系中的坐标为@@29@@。

5.将变形后的天平轴向单位向量坐标带入公式，计算得到XFD=@@30@@。计算得到YFD=@@31@@。计算得到ZFD=@@32@@。

6.确定Δα_e、Δβ_e如何用XFD、YFD、ZFD表示。这里采用sinθ=θ、cosθ=1简化。正理论迎角时，空速指向机体的@@39@@。在该空速方向和风轴系轴向、法向、侧向都已确定后，Δα_e应使用的风轴分量为@@33@@。正理论侧滑时，空速指向机体的@@40@@。在该空速方向和风轴系轴向、法向、侧向都已确定后，Δβ_e应使用的风轴分量为@@34@@。

7.这步推导Δα_e、Δβ_e的公式符号。γ_tp可能导致符号受α_e与β_e的数值影响。这里假设γ_tp=0，从边界条件判断符号。此时天平平置安装。当产生形变α_e、β_e、γ_e时，模型是@@35@@的。产生这些形变后，空速位于模型的@@36@@。空速同时位于模型的@@37@@。此时绕天平绝对轴向正方向（前方）的旋转方向与右手定则方向@@38@@。滚转角的定义是绕机体绝对轴向正方向（前方）的右手定则@@41@@为正。因此天平弹性角产生的增量中，Δα_e为@@42@@。天平弹性角产生的增量中，Δβ_e为@@43@@。天平弹性角产生的增量中，Δγ_e为@@44@@。此时同样能够判断坐标分量的正负，XFD为@@45@@。判断得到的YFD为@@46@@。判断得到的ZFD为@@47@@。当存在γ_tp时，Δα_e、Δβ_e的方向会基于这个初始状态，随α_e与β_e的数值变化。

8.用XFD、YFD、ZFD表示的公式模板为[角度符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在分式外面。禁止把正负号放进分式内部来代替外层符号。其中Δα_e的角度符号为@@48@@。Δα_e的分子应使用的风轴分量为@@50@@。根据该分量的符号，Δα_e分子的符号为@@49@@。Δα_e的分母应使用的风轴分量为@@52@@。根据该分量的符号，Δα_e分母的符号为@@51@@。Δβ_e的角度符号为@@53@@。Δβ_e的分子应使用的风轴分量为@@55@@。根据该分量的符号，Δβ_e分子的符号为@@54@@。Δβ_e的分母应使用的风轴分量为@@57@@。根据该分量的符号，Δβ_e分母的符号为@@56@@。求得Δα_e=@@58@@，这个空只能填风轴分量 XFD、YFD、ZFD 的分式，分子和分母都为正，负号写在整个分式之前，不能直接写含 α_e、β_e 或 γ_e 的弹性角式。求得Δβ_e=@@59@@。用α_e、β_e表示的公式为Δα_e=@@60@@。用α_e、β_e表示的公式为Δβ_e=@@61@@。由天平弹性角产生的滚转增量不随天平安装角γ_tp改变。符号由弹性滚转与理论滚转绕绝对轴向正方向（前方）是否同向决定，故Δγ_e=@@62@@。

9.Δα_V和Δβ_V是风洞气流偏心造成的迎角和侧滑角补偿。正方向定义是：气流偏心使机体的实际迎角或侧滑角增大时为正。补偿公式为α=α_t+Δα_e+Δα_V，β=β_t+Δβ_e+Δβ_V。补偿量Δα_V的符号为@@63@@。补偿量Δβ_V的符号为@@64@@。""",
    "力矩参考点": """1.天平系转到体轴系后，该体轴系的原点位于@@1@@处。工程上会给出力矩参考点，需要把体轴系上的力矩等效到力矩参考点。

2.天平嵌入模型内部。天平校心到模型顶点的距离为Xr，这个距离通过间接测量计算。天平校心到天平支杆底部的距离为L_r，模型总长为L，模型底部到天平支杆底部的距离为L_t，三者均已测得，且共线，校心位于顶点与支杆底部之间。由此Xr=@@2@@。

3.力矩参考点的工程约定坐标为[Xg,Yg,Zg]。工程约定与数学坐标的差异是：沿模型轴向的分量定义为参考点到模型顶点的距离，这个工程正方向指向模型后方，与绝对轴向正方向（前方）相反；其余两分量的正方向与它们所在体轴坐标轴的正方向相同，不改成绝对法向（上方）或绝对侧向（右侧）。由此，坐标分量@@3@@与体轴系数学坐标可能存在差异。换算到体轴系数学坐标后，力矩参考点的坐标是@@4@@。

4.从体轴系@@5@@到@@6@@的力矩等效，需要考虑@@7@@处的力对@@8@@处产生的额外力矩。

5.MCA_g=MCA+[MCN_CA]+[MCZ_CA]，MCZ_g=MCZ+[MCA_CZ]+[MCN_CZ]，MCN_g=MCN+[MCA_CN]+[MCZ_CN]。其中MCN_CA是原点处法向力在参考点处轴向产生的等效力矩系数，MCN_CZ是原点处法向力在参考点处侧向产生的等效力矩系数。每一项只允许写成模板C·L_c/L_RT，L_RT是固定参考长度。答案里不能保留字母C和L_c。C必须填入产生该项力矩的力系数，只能是CA、CZ、CN中的一个。L_c必须填入力臂，力臂只能由Xg、Yg、Zg、Xr组成。假设当前Xg、Yg、Zg、Xr都为正。符号放在整个式子最外侧，力臂内部必须为正，表示额外力矩的绝对值。在该假设下，等效力矩系数MCN_CA的符号为@@9@@。等效力矩系数MCZ_CA的符号为@@10@@。等效力矩系数MCA_CZ的符号为@@11@@。等效力矩系数MCN_CZ的符号为@@12@@。等效力矩系数MCA_CN的符号为@@13@@。等效力矩系数MCZ_CN的符号为@@14@@。因此等效力矩系数MCN_CA=@@15@@。等效力矩系数MCZ_CA=@@16@@。等效力矩系数MCA_CZ=@@17@@。等效力矩系数MCN_CZ=@@18@@。等效力矩系数MCA_CN=@@19@@。等效力矩系数MCZ_CN=@@20@@。""",
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
        {"id": "01", "title": "统一默认", "change": "全部取统一默认定义。", "spec": standard},
        {"id": "02", "title": "反装", "change": "安装方式改为反装，其余保持默认。", "spec": inverted},
        {"id": "03", "title": "理论迎角反向", "change": "理论迎角改为空速指向机体上方为正，机构迎角仍指向机体下方。", "spec": alpha_up},
        {"id": "04", "title": "理论侧滑反向", "change": "理论侧滑角改为空速指向机体左侧为正，机构侧滑角仍指向机体右侧。", "spec": beta_left},
        {"id": "05", "title": "滚转与安装角反向", "change": "机构滚转改为与绕模型绝对轴向正方向（前方）的右手定则相反，天平安装角与绕天平绝对轴向正方向（前方）的右手定则相反。", "spec": roll_install},
        {"id": "07", "title": "体轴左手且轴向朝后", "change": "体轴系改为左手系，XTZ沿机体轴向指向机体后方，YTZ指向机体上方。", "spec": body_left},
        {"id": "08", "title": "风轴侧向与法向换轴", "change": "风轴系YFD改为沿风洞侧向指向右侧，ZFD由右手系推断。", "spec": wind_swap},
        {"id": "09", "title": "弹性角与气流偏角反向", "change": "三个弹性角和两个气流偏角的正方向都改为与默认相反。", "spec": elastic_flow},
        {"id": "10", "title": "天平左手且轴向朝后", "change": "天平系改为左手系，XTP沿天平轴向指向后方；模型安装角改为与右手定则相反。", "spec": balance_left},
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
        "forward": "前方",
        "rearward": "后方",
        "up": "上方",
        "down": "下方",
        "left": "左侧",
        "right": "右侧",
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
        if target not in {"机体下方", "机体上方"}:
            raise RuntimeError(f"理论迎角的正方向不是上下：{target}")
        sign = "正" if target == "机体下方" else "负"
        return (
            "理论迎角 α：空速在飞行器对称面上的投影与机体系轴向正方向的夹角。"
            f"投影处于机体轴向下方时为{sign}。"
        )
    if name == BETA:
        if target not in {"机体右侧", "机体左侧"}:
            raise RuntimeError(f"理论侧滑角的正方向不是左右：{target}")
        sign = "正" if target == "机体右侧" else "负"
        return (
            "理论侧滑角 β：空速向量与飞行器对称面的夹角，"
            f"空速在飞行器右侧为{sign}。"
        )
    if kind == "airspeed":
        return f"{ANGLE_NAME[name]}：空速指向{target}时为正。"
    if kind == "wing":
        return f"{ANGLE_NAME[name]}：右翼指向{target}时为正。"
    raise KeyError(kind)


ROD_ORDER_PARAGRAPH = (
    "3.根据机构结构，推断第一个旋转角为@@16@@。第二个旋转角为@@17@@。"
    "第三个旋转角为@@18@@。\n\n"
)
WRIST_ORDER_PARAGRAPH = (
    "3.本题规定机构姿态角按偏航、俯仰、滚转的顺序执行。"
    "第一个旋转角为@@16@@。第二个旋转角为@@17@@。第三个旋转角为@@18@@。\n\n"
)


def _process_template(model: BodyFrameModel, process: str) -> str:
    """Stock subprocess text, with the wrist order stated when that mechanism is declared."""
    template = PROCESS_TEXT[process]
    if process == "风轴系到体轴系" and tuple(model.euler_order) == WRIST_ORDER:
        if template.count(ROD_ORDER_PARAGRAPH) != 1:
            raise RuntimeError("机构顺序原句没有恰好出现一次。")
        return template.replace(ROD_ORDER_PARAGRAPH, WRIST_ORDER_PARAGRAPH, 1)
    return template


def _mechanism_lines(model: BodyFrameModel) -> list[str]:
    """Angle-mechanism sentences. The wrist names an order; the rod leaves it implicit."""
    mechanism = model.mechanism_angles
    alpha_side = _balance_deflection(mechanism[ALPHA_M].positive_airspeed_toward)
    beta_side = _balance_deflection(mechanism[BETA_M].positive_airspeed_toward)
    gamma_word = (
        "相同"
        if roll_about_nose_sense(mechanism[GAMMA_M].right_wing_toward) == SENSE_SAME
        else "相反"
    )
    if tuple(model.euler_order) == WRIST_ORDER:
        return [
            "机构角由三自由度动轴球腕执行。三根转轴依次为偏航轴、俯仰轴、滚转轴，执行顺序规定为偏航、俯仰、滚转。偏航对应机构侧滑角 β_M，俯仰对应机构迎角 α_M，滚转对应机构滚转角 γ_M。",
            f"机构迎角 α_M：球腕的俯仰转角，绕绝对侧向轴。天平偏向风洞{alpha_side}时为正。",
            f"机构侧滑角 β_M：球腕的偏航转角，绕绝对法向轴。使天平偏向风洞{beta_side}时为正。",
            (
                "机构滚转角 γ_M：球腕的滚转转角，绕模型绝对轴向正方向（前方）。"
                f"角度为正时，与右手定则方向{gamma_word}。"
            ),
        ]
    return [
        (
            "机构迎角 α_M：功角支撑杆在铅垂面上运动，可视为绕风轴系的侧向轴运动，"
            "为支撑杆（天平轴向）与水平面的夹角。"
            f"天平偏向风洞{alpha_side}时为正。"
        ),
        (
            "机构侧滑角 β_M：侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，"
            "即可视为侧滑角机构基于功角机构在作动。"
            "β_M是侧滑角支撑杆与铅垂面的夹角。"
            f"使天平偏向风洞{beta_side}时为正。"
        ),
        (
            "机构滚转角 γ_M：机体绕侧滑角支撑杆的转角。"
            "角度为正时，以模型绝对轴向正方向（前方）旋转，与右手定则方向"
            f"{gamma_word}。"
        ),
    ]


def _frame_lines(title: str, names: tuple[str, ...], frame: dict) -> list[str]:
    lines = [title]
    for name in names:
        axis = frame[name]
        lines.append(
            f"- {name}：沿{AXIS_ZH[axis.absolute_axis]}，指向{_direction(axis.points_to)}（{SOURCE_ZH[axis.source]}）。"
        )
    return lines


def _declaration(model: BodyFrameModel) -> str:
    angles = model.angles
    elastic = model.elastic_angles
    flow = model.flow_angles
    lines = [
        "固定约定",
        "- 坐标系转换一律采用被动变换。机体姿态旋转一律为动轴内旋。",
        "- 力与力矩固定对应绝对轴：CN法向、CZ侧向、CA轴向；MZ俯仰对应侧向、MA滚转对应轴向、MN偏航对应法向。体轴和天平使用同一对应。",
        "- 体轴系原点在天平校心。模型顶点到模型底部为L，模型底部到支杆底部为L_t，天平校心到支杆底部为L_r。三者共线，校心位于模型顶点与支杆底部之间。天平校心到模型顶点的距离记为Xr。",
        "- 工程坐标里，轴向分量从顶点向后量到参考点，这个工程正方向与绝对轴向正方向（前方）相反。另外两个分量沿各自体轴坐标轴的正方向，不改成绝对法向（上方）或绝对侧向（右侧）。力作用在原点，力矩转到参考点。额外力矩只写成C·L_c/L_RT。",
        "",
        "绝对方向",
        "机体、风洞、天平各有三根绝对轴。三套绝对轴的正方向定义相同：轴向正方向是前方，法向正方向是上方，侧向正方向是右侧。六个绝对方向是这三根轴上的两个指向。",
        "下文凡写轴向正方向、法向正方向、侧向正方向，以及模型轴向的正方向、机体轴向的正方向、天平轴向正方向、风洞轴向的正方向，都是这三根绝对轴的正方向，不是坐标轴 X、Y、Z 的正方向。坐标轴声明只是指定每根坐标轴沿哪根绝对轴、指向哪一侧，可以与绝对正方向相同，也可以相反。",
        "三个系共用同一套指向关系：前方 × 右侧 = 下方。沿前方看去，右侧在右手侧，下方铅垂向下。这个关系不随定义改变。",
        "机体的三轴六方向相对机体本身，正装或反装都不改写它们：",
        "- 轴向：连接机尾与机头。正方向朝前，指向机头。后方指向机尾。",
        "- 法向：机体的上下轴。正方向朝上。下方是其反方向。",
        "- 侧向：机体的左右轴。正方向朝右。左侧是其反方向。",
        "风洞的三轴六方向相对风洞：",
        "- 轴向：风洞轴线。正方向朝前。前方是来流，与空速方向一致，是气流方向的反方向。后方是其反方向。",
        "- 法向：风洞铅垂轴。正方向朝上，上方铅垂向上。下方铅垂向下。",
        "- 侧向：风洞水平横轴。正方向朝右。左侧是其反方向。",
        "天平的三轴六方向与风洞重合，不另设指向，也不随模型正装或反装翻转。天平前方、后方、上方、下方、右侧、左侧分别就是风洞的前方、后方、上方、下方、右侧、左侧。天平轴向、法向、侧向分别就是风洞轴向、法向、侧向。天平绝对轴向正方向是前方，绝对法向正方向是上方，绝对侧向正方向是右侧。天平坐标轴 XTP、YTP、ZYP 的正方向由后面的坐标轴声明给出，可以与绝对正方向相同，也可以相反。",
        "",
        "场景",
        "初始位置时，模型平正处于风洞中，模型前方朝向来流方向。安装有正装和反装两种。正装时模型上方朝风洞上方，反装时模型上方朝风洞下方。天平处于模型内部，天平尾部通过一根支杆伸出模型，与风洞的角度机构相连。角度机构运动时，通过支杆带动天平和模型一起运动。正装时，平正状态下的模型上方和地面上方一致。反装时，平正状态下的模型上方和地面下方一致。天平由支杆支撑，会产生弹性形变，从而产生弹性角。由于风洞气流不能做到完全与风洞轴向重合，所以会产生气流偏角。天平弹性角和气流偏角均会对模型实际理论迎角、侧滑角、滚转角造成影响，需要进行校正。",
        "",
        "角度",
        _positive_sentence(ALPHA, angles[ALPHA].positive_airspeed_toward, "airspeed"),
        _positive_sentence(BETA, angles[BETA].positive_airspeed_toward, "airspeed"),
        _positive_sentence(GAMMA, angles[GAMMA].right_wing_toward, "wing"),
        *_mechanism_lines(model),
        "",
        "安装",
        (
            f"- 模型安装：{'正装' if model.placement == 'upright' else '反装'}。"
            "正装时机体上方与风洞上方相同，机体下方与风洞下方相同，机体左侧与风洞左侧相同，机体右侧与风洞右侧相同，机体前方与风洞前方相同。"
            "反装时机体上方朝向风洞下方，机体下方朝向风洞上方，机体左侧朝向风洞右侧，机体右侧朝向风洞左侧，机体前方仍与风洞前方相同。"
        ),
        (
            "- 天平安装角 γ_tp：法向偏离过天平轴向的铅垂面。"
            f"正方向与绕天平绝对轴向正方向（前方）的右手定则{'相同' if model.installation.sense == 'same' else '相反'}。"
        ),
        (
            "- 模型安装角 γ_model：法向偏离过机体轴向的铅垂面。"
            f"正方向与绕机体绝对轴向正方向（前方）的右手定则{'相同' if model.model_installation.sense == 'same' else '相反'}。"
        ),
        "",
        "弹性角与气流偏角",
        f"- 弹性迎角 α_e：天平轴向弯向{_direction(elastic[ALPHA_E].positive_toward)}时为正。",
        f"- 弹性侧滑角 β_e：天平轴向弯向{_direction(elastic[BETA_E].positive_toward)}时为正。",
        f"- 弹性滚转角 γ_e：天平侧向移向{_direction(elastic[GAMMA_E].positive_toward)}时为正。",
        "- 弹性迎角增量 Δα_e：天平变形后天平轴向在过风洞轴向铅垂面的投影与风洞轴向的夹角。",
        "- 弹性侧滑角增量 Δβ_e：天平变形后天平轴向与过风洞轴向的铅垂面的夹角。",
        f"- 气流偏角 α_V：模型正装且平置时，空速指向{_direction(flow[ALPHA_V].positive_airspeed_toward)}为正。",
        f"- 气流偏角 β_V：模型正装且平置时，空速指向{_direction(flow[BETA_V].positive_airspeed_toward)}为正。",
        "",
    ]
    lines.extend(_frame_lines(
        f"体轴系（{HAND_ZH[model.chirality]}）",
        BODY_AXES,
        model.body_frame,
    ))
    lines.append("")
    lines.extend(_frame_lines(
        f"风轴系（{HAND_ZH[model.wind_chirality]}）",
        WIND_VOCAB.coordinate_axes,
        model.wind_frame,
    ))
    lines.append("")
    lines.extend(_frame_lines(
        f"天平系（{HAND_ZH[model.balance_chirality]}）",
        BALANCE_VOCAB.coordinate_axes,
        model.balance_frame,
    ))
    return "\n".join(lines)


_DECLARATION_REQUIRED = (
    "理论迎角 α：",
    "理论侧滑角 β：",
    "理论滚转角 γ：",
    "机构迎角 α_M：",
    "机构侧滑角 β_M：",
    "机构滚转角 γ_M：",
    "模型安装：",
    "天平安装角 γ_tp：",
    "模型安装角 γ_model：",
    "弹性迎角 α_e：",
    "弹性侧滑角 β_e：",
    "弹性滚转角 γ_e：",
    "弹性迎角增量 Δα_e：天平变形后天平轴向在过风洞轴向铅垂面的投影与风洞轴向的夹角。",
    "弹性侧滑角增量 Δβ_e：天平变形后天平轴向与过风洞轴向的铅垂面的夹角。",
    "气流偏角 α_V：",
    "气流偏角 β_V：",
    "体轴系（",
    "- XTZ：",
    "- YTZ：",
    "- ZTZ：",
    "风轴系（",
    "- XFD：",
    "- YFD：",
    "- ZFD：",
    "天平系（",
    "- XTP：",
    "- YTP：",
    "- ZYP：",
)
_DECLARATION_FORBIDDEN = ("定义变更", "保持统一默认", "不计入", "改为", "不是可变定义")


def _check_declaration(declaration: str) -> None:
    missing = [item for item in _DECLARATION_REQUIRED if item not in declaration]
    if missing:
        raise RuntimeError(f"定义声明缺少：{'、'.join(missing)}")
    present = [item for item in _DECLARATION_FORBIDDEN if item in declaration]
    if present:
        raise RuntimeError(f"定义声明仍在对照变更：{'、'.join(present)}")


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
    if process == "力矩参考点" and item["manuscript_id"] == "2":
        return DISTANCE_FORMAT
    return EXPRESSION_FORMAT


def _question(process: str, item: dict) -> dict:
    output_format = _output_format(process, item)
    if item["options"] and item["answer"] not in item["options"]:
        raise RuntimeError(
            f"{process} ID {item['manuscript_id']} 的答案“{item['answer']}”不在选项中。"
        )
    question = {
        "process": process,
        "id": _public_id(process, item["manuscript_id"]),
        "local_id": item["manuscript_id"],
        "prompt": item["prompt"],
        "options": item["options"],
        "output_format": output_format,
        "format_code": "无" if output_format is None else FORMAT_CODE[output_format],
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
    options = [str(option) for option in (item.get("options") or []) if option and str(option) != "无"]
    if not options:
        return answer
    _letter, text = resolve_choice(answer, options)
    return text


def _review_slot(item: dict) -> str:
    mention = ""
    if item.get("rule_mention"):
        kind = "回忆" if item["rule_mention"] == "recall" else "方法提示"
        mention = f", 规则提及:{kind}/{item.get('rule_definition_name') or item.get('rule_definition')}"
    return (
        f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}, "
        f"标签:{item['label']}, 答案:{item['answer']}{mention}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}}}"


def _prepared_template(process: str, phrases: dict[str, str] | None) -> str:
    template = PROCESS_TEXT[process]
    for token, text in (phrases or {}).items():
        template = template.replace(token, text)
    return template


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
        if "。" in template[cursor:match.start()] and current:
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
        raise RuntimeError(f"{process} 题面有重复空。")
    if sorted(seen, key=int) != sorted(expected, key=int):
        missing = [item for item in expected if item not in set(seen)]
        extra = [item for item in seen if item not in set(expected)]
        raise RuntimeError(f"{process} 题面空与生成空不一致，缺 {missing}，多 {extra}。")


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
                period = tail.find("。")
                if period == -1:
                    raise RuntimeError(f"{process} ID {blank_id} 所在句没有句号。")
                if SLOT_MARK.search(tail[:period]):
                    raise RuntimeError(f"{process} ID {blank_id} 的句子里还有未纳入的空。")
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            if not keep_suffix:
                raise RuntimeError(f"{process} ID {blank_id} 出现在当前句的待填槽之后。")
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_phrase(item) if memory is None else memory.get(item["id"], ""))
        cursor = match.end()
    if open_ids:
        if not seen_open:
            missing = target if target is not None else "、".join(open_ids)
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
        return f"{blank_id}:{letter if example else '字母'}"
    if example:
        shown = DELTA_ALPHA_QUOTIENT_EXAMPLE if _is_delta_alpha_quotient(item) else FORMAT_EXAMPLE[item["format_code"]]
        return f'{blank_id}:"{shown}"'
    return f'{blank_id}:"答案"'


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
        "【输出】",
        "所有答案都放在 JSON 对象中。键是空的编号，值是答案，例如 {1:A}。",
        "本次是分步问答：一个问题句只输出一个 JSON 对象，这一句里的全部空都写在这个对象中。",
        "端到端问答才把整条过程的全部答案放进同一个 JSON 对象。本次只写当前这一句。",
        "选择题的值只写字母，不加引号。公式、矩阵、向量、表达式和距离必须用英文双引号包成字符串。",
    ]


def _is_delta_alpha_quotient(item: dict) -> bool:
    """天平系到风轴系的代入后 Δα_e。这个空要风轴分量分式，不接受弹性角式。"""
    return item.get("process") == "天平系到风轴系" and str(item.get("local_id")) == "58"


def _formula_note(item: dict) -> str:
    if _is_delta_alpha_quotient(item):
        return DELTA_ALPHA_QUOTIENT_NOTE
    return ANGLE_SYMBOL_NOTE


def _output_block(item: dict) -> str:
    lines = _output_preamble()
    lines.append(f"只输出一行 {_json_object([item])}。")
    if item.get("options"):
        lines.append("不要写选项原文、句号、解释、Markdown或LaTeX。")
    else:
        lines.extend([
            "双引号只表示 JSON 字符串，不是答案的一部分。不要输出引号以外的解释、Markdown或LaTeX。",
            _formula_note(item),
            DELTA_ALPHA_QUOTIENT_FORMAT if _is_delta_alpha_quotient(item) else FORMAT_TEXT[item["format_code"]],
        ])
    lines.append(f"格式示例（不是本题答案）：{_json_object([item], example=True)}")
    return "\n".join(lines)


def _output_block_many(by_id: dict[str, dict], blank_ids: list[str]) -> str:
    items = [by_id[blank_id] for blank_id in blank_ids]
    lines = _output_preamble()
    lines.extend([
        f"只输出一行 {_json_object(items)}。按编号顺序写在同一个对象里，不要拆成多个 JSON 对象。",
        "不要输出句子原文。",
    ])
    if not any(_is_delta_alpha_quotient(item) for item in items):
        lines.append(ANGLE_SYMBOL_NOTE)
    for item in items:
        if item.get("options"):
            lines.append(f"ID {item['id']} 的值是选项字母。")
            continue
        code = item["format_code"]
        if _is_delta_alpha_quotient(item):
            lines.append(
                f"ID {item['id']} 的值是{code}，必须用英文双引号包成字符串。"
                f"{DELTA_ALPHA_QUOTIENT_NOTE}{DELTA_ALPHA_QUOTIENT_FORMAT}"
            )
        else:
            lines.append(f"ID {item['id']} 的值是{code}，必须用英文双引号包成字符串。{FORMAT_TEXT[code]}")
    lines.append(f"格式示例（不是本题答案）：{_json_object(items, example=True)}")
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
            f"只填写当前这一句。本次是分步问答，只输出一个 JSON 对象 {_json_object([item])}。"
            "不要解释，不要填写其他空。"
        )
        output = _output_block(item)
    else:
        listed = "、".join(group)
        lead = (
            f"填写这一句里的全部待填槽，ID 为 {listed}。"
            "这些槽属于同一句话，一次把这句话答完。"
            f"本次是分步问答，只输出一个 JSON 对象 {_json_object([by_id[blank_id] for blank_id in group])}。"
            "这一句的全部空都放在这一个对象里，不要拆成多个对象。"
            "选择题的值是字母，不加引号。公式、矩阵、向量、表达式和距离的值用英文双引号包成字符串。"
            "不要输出句子原文，不要解释。"
        )
        output = _output_block_many(by_id, group)
    if keep_suffix:
        lead += "句号之后的后续正文只供参考，不要填写那些空。"
    return "\n".join([
        lead,
        "",
        "【定义声明】",
        declaration,
        "",
        "【物理常识】",
        PRIOR,
        "",
        "【子过程】",
        process,
        "",
        "【过程】",
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
        "本次是端到端问答。填写这个子过程里的全部待填槽，只调用这一次。"
        "只输出一个 JSON 对象，键是空的编号，值是答案。"
        "这个对象要包含本子过程的每一个编号，不要拆成多个对象，不要按句子分开作答。"
        "选择题的值是字母，不加引号。公式、矩阵、向量、表达式和距离的值用英文双引号包成字符串。"
        "不要输出句子原文，不要解释。"
    )
    output = "\n".join([
        "【输出】",
        "所有答案放在同一个 JSON 对象中。键是空的编号，值是答案。",
        '例如 {1:A, 2:C, 11:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}。示例不是本题答案。',
        "选择题的值只写字母，不加引号。公式、矩阵、向量、表达式和距离必须用英文双引号包成字符串。",
        "只输出这一行 JSON，不要解释、Markdown或LaTeX。",
    ])
    return "\n".join([
        lead,
        "",
        "【定义声明】",
        declaration,
        "",
        "【物理常识】",
        PRIOR,
        "",
        "【子过程】",
        process,
        "",
        "【过程】",
        _render_process(
            process, by_id, phrases=phrases, targets=blank_ids,
            keep_suffix=True, template=template,
        ),
        "",
        output,
    ])


def _render_markdown(record: dict) -> str:
    lines = [
        f"# 数据 {record['id']} {record['title']}",
        "",
        "本文件是审阅稿，展示一条数据的完整过程。评测时一次答一句话，不把整条过程一次送入。输入保留定义声明、物理常识、当前子过程名，以及从该子过程开头直到这一句的句号。这一句之前的空替换成选项所指的原文，例如 XFD、正装，不写字母。这一句里的空全部保留为待填槽，一起填写。句号之后的正文不送入。待填槽不含标签和答案。标签只用于计分。模型把这一句的全部空写成一个 JSON 对象，键是编号，值是答案。",
        "",
        "待填槽格式：{ID:编号, 选项:A：选项1；B：选项2, 输出格式:格式名}。没有选项时，选项写无。输出格式名的含义见 data/wind_tunnel/review/00_注入格式.md。",
        "",
        "## 定义声明",
        "",
        record["declaration"],
        "",
        "## 物理常识",
        "",
        PRIOR,
        "",
        "## 完整过程",
        "",
        "下面每个槽都带标签和标准答案，便于直接审阅。这是审阅字段，不是模型在待填空上看到的字段。",
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
            "风轴系到体轴系",
            {item["local_id"]: item for item in questions if item["process"] == "风轴系到体轴系"},
            "22",
            phrases,
        ),
    }


def _format_document(example: str) -> str:
    lines = [
        "# 注入到模型的正式数据格式",
        "",
        "一次调用评测一句话。这句话里有多个空时，这些空一起填写，不拆成互相独立的空。输入由五段组成，顺序固定。",
        "",
        "1. 作答要求：所有答案都放在 JSON 对象中，键是空的编号，值是答案，例如 {1:A}。分步问答的范围是一个问题句：这一句只有一个空时输出 {1:A}，有多个空时输出一个对象，例如 {1:A, 2:C}，不要拆成多个对象。端到端问答才把整条过程的全部答案放进同一个对象。选择题的值是字母，不加引号。公式、矩阵、向量、表达式和距离必须用英文双引号包成字符串，例如 {11:\"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]\"}。不输出句子原文，不解释。",
        "2. 【定义声明】该条数据的完整定义。",
        "3. 【物理常识】下面这段固定文字，每次都附上。",
        "4. 【子过程】当前句所属子过程的名称。",
        "5. 【过程】从该子过程开头写到当前句的句号。这一句之前的槽替换成选项所指的原文，例如 XFD、正装，不写 A、B 这样的字母。这一句里的每个槽都写成待填槽。句号之后的正文不出现。这是条件 A。计分用的标准答案仍是字母。",
        "",
        "待填槽只有三个字段：{ID:编号, 选项:A：选项1；B：选项2, 输出格式:格式名}。没有选项时写“选项:无”。待填槽不包含标签，也不包含答案。",
        "",
        "每次注入在过程之后附上【输出】。其中写明每个待填槽的 JSON 形式，并给出输出示例。示例只说明字符怎么排，不是该槽的答案。有选项时，待填槽把选项写成 A：选项1；B：选项2，模型输出 {编号:字母}。计分时取出 JSON 的值；正确字母、选项原文，以及能唯一对应正确选项的片段，都算命中。公式类答案去掉外层双引号后再比对。",
        "",
        "各条审阅稿展示的是同一条过程的全文。其中每个槽额外写出标签和答案，这两项是审阅字段。人可以据此理解：评测哪一句，就把那一句里的槽都换成待填槽，把更早的槽换成答案，并在句号处结束。",
        "",
        "四个逐句条件共用同一模板、同一句序，并在每个子过程开头重新开始。A：空前写入选项原文，句号后截断。B：空前换成模型上一句的原始输出，不纠正、不归一化；调用失败或空返回也原样放进下一句。C′：空前仍是选项原文，句号后的后续正文保留，其中的空继续待填、不预填答案。C″：空前是模型自己的原始输出，同时保留后续正文。A 与 B 只差前缀来源。A 与 C′ 只差有没有后续正文。",
        "",
        "空的编号按子过程顺序排列。风轴系到体轴系占用 1–81，其中 52、53、54 三道分矩阵展开式已删除。天平系到体轴系占用 82–116，其中 101、102、103 三道分矩阵展开式已删除。天平系到风轴系 117–180，力矩参考点 181–200。删掉的空号不前移。",
        "",
        "D 是端到端。每个子过程单独一次调用，正文从开头保留到结尾，每个空都是待填槽，不预填答案。模型只输出一个 JSON 对象，键使用上面的连续编号。",
        "",
        "## 物理常识",
        "",
        PRIOR,
        "",
        "## 输出格式名",
        "",
    ]
    for code, text in FORMAT_TEXT.items():
        lines.append(f"- {code}：{text} 示例：{FORMAT_EXAMPLE[code]}")
    lines.extend([
        "",
        "## 一次真实注入",
        "",
        "下面是数据 01、子过程“风轴系到体轴系”、ID 22 的完整输入。这是单独成句的 A1。其他句子按同一规则截取。",
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
                raise RuntimeError(f"{process} 把 A 与 C 放在同一句：{list(zip(sentence, kinds))}")
            if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
                raise RuntimeError(f"{process} 的 A 类空没有单独成句：{list(zip(sentence, kinds))}")

    numbers = [item["id"] for item in record["questions"]]
    removed = {str(number) for slots in REMOVED_SLOTS.values() for number in slots}
    expected_numbers = [str(number) for number in range(1, 201) if str(number) not in removed]
    if numbers != expected_numbers:
        raise RuntimeError("四个子过程的编号与保留空号不一致。")
    wind_count = PROCESS_SIZE["风轴系到体轴系"] - len(REMOVED_SLOTS["风轴系到体轴系"])
    if record["questions"][wind_count]["id"] != "82" or record["questions"][wind_count]["process"] != "天平系到体轴系":
        raise RuntimeError("天平系到体轴系没有从 82 接上。")
    if record["questions"][-1]["id"] != "200" or record["questions"][-1]["process"] != "力矩参考点":
        raise RuntimeError("力矩参考点没有排到 200。")
    group = {
        item["local_id"]: item
        for item in record["questions"]
        if item["process"] == "风轴系到体轴系"
    }

    def process_body(blank_id: str) -> str:
        text = _injection(record["declaration"], "风轴系到体轴系", group, blank_id, phrases)
        return text.split("【过程】\n", 1)[1].split("\n\n【输出】", 1)[0]

    sentence_22 = process_body("22")
    if not sentence_22.endswith("理论迎角的符号为{ID:22, 选项:A：正；B：负；C：不确定, 输出格式:无}。"):
        raise RuntimeError("ID 22 没有单独成句。")
    if "{ID:21," in sentence_22:
        raise RuntimeError("ID 22 的注入仍把 ID 21 留在同一句。")
    if process_body("23") == process_body("24"):
        raise RuntimeError("ID 23 与 ID 24 仍在同一次注入里，坐标系答案进不了转轴前缀。")
    if "这相当于中间体轴系平正时。" not in process_body("24") or "{ID:23," in process_body("24"):
        raise RuntimeError("ID 24 的前缀没有写上 ID 23 的中间体轴系答案。")
    if "坐标起点变为此时的中间体轴系，该坐标系依然沿用机体系的XTZ、YTZ、ZTZ，不做另行的符号设定。" not in process_body("16"):
        raise RuntimeError("姿态旋转前没有声明坐标起点变为中间体轴系。")
    if "旋转α_M角度" in process_body("23"):
        raise RuntimeError("ID 23 所在句仍带上了 A1 的转向。")
    sentence_25 = process_body("25")
    if not sentence_25.endswith("绕该轴旋转方向为{ID:25, 选项:A：正；B：负；C：不确定, 输出格式:无}，旋转α_M角度，对应的旋转矩阵为R_α_M。"):
        raise RuntimeError("ID 25 所在句没有写明旋转方向和矩阵名称。")
    if "{ID:26," in sentence_25 or "{ID:24," in sentence_25:
        raise RuntimeError("ID 25 的注入带进了相邻句的槽。")
    earlier = process_body("8")
    if "分别对应YFD、XFD、ZFD" not in earlier or "分别对应YTZ、XTZ、ZTZ" not in earlier:
        raise RuntimeError("空前的标准答案没有写成坐标轴原文。")
    if "分别对应A" in earlier or "分别对应D" in earlier:
        raise RuntimeError("空前的标准答案仍是字母。")
    leaked = _injection(
        record["declaration"], "风轴系到体轴系", group, "22", phrases, keep_suffix=True,
    )
    if "这相当于" not in leaked or "{ID:23," not in leaked:
        raise RuntimeError("C′ 没有保留 ID 22 之后的正文。")
    if "句号之后的后续正文只供参考" not in leaked:
        raise RuntimeError("C′ 没有说明后续空不要填写。")
    remembered = _injection(
        record["declaration"], "风轴系到体轴系", group, "22", phrases, memory={"19": "MEMTEST"},
    )
    if "MEMTEST" not in remembered or "MEMTEST" in sentence_22:
        raise RuntimeError("B 没有用模型自己的前序输出替换空前答案。")
    if group["19"]["answer"] in remembered.split("MEMTEST", 1)[0].split("模型采用", 1)[-1]:
        raise RuntimeError("B 仍把 ID 19 的标准答案留在前缀里。")
    both = _injection(
        record["declaration"], "风轴系到体轴系", group, "22", phrases,
        memory={"19": "MEMTEST"}, keep_suffix=True,
    )
    if "MEMTEST" not in both or "{ID:23," not in both:
        raise RuntimeError("C″ 没有同时保留自己的前序输出和后续正文。")
    end_to_end = _injection_end_to_end(record["declaration"], "风轴系到体轴系", group, phrases)
    if "本次是端到端问答" not in end_to_end or "分别对应YFD、XFD、ZFD" in end_to_end:
        raise RuntimeError("端到端仍预填了前面的答案。")
    if end_to_end.count("{ID:") != 78:
        raise RuntimeError(
            f"端到端没有把这个子过程的空全部留开，现在有 {end_to_end.count('{ID:')} 个。"
        )
    choice = _injection(record["declaration"], "风轴系到体轴系", group, "22", phrases)
    if "{22:字母}" not in choice or '{22:"' in choice:
        raise RuntimeError("选择题没有要求输出 {22:字母}。")
    several = _injection(record["declaration"], "风轴系到体轴系", group, "2", phrases)
    if "{2:字母, 3:字母, 4:字母}" not in several or "不要拆成多个对象" not in several:
        raise RuntimeError("同一句的多个空没有要求放进一个 JSON 对象。")
    matrix = _injection(record["declaration"], "风轴系到体轴系", group, "8", phrases)
    if '{8:"[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]"}' not in matrix:
        raise RuntimeError("矩阵没有要求用双引号放进 JSON。")


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
        "# 9条校核数据",
        "",
        "正式注入格式见 00_注入格式.md。每条审阅稿给出定义声明和四个子过程的完整正文，空留在原句中。",
        "",
    ]
    for record in records:
        path = OUT_DIR / f"sample_{record['id']}.md"
        path.write_text(_render_markdown(record), encoding="utf-8")
        index.append(f"- 数据 {record['id']} {record['title']}：{record['change']}")
    (OUT_DIR / "00_索引.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (OUT_DIR / "00_注入格式.md").write_text(_format_document(records[0]["injection_example"]), encoding="utf-8")
    print(f"Wrote {len(records)} samples, {len(records[0]['questions'])} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
