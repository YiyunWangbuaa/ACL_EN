# -*- coding: utf-8 -*-
"""Build the wind-tunnel × spherical-wrist review from the seven declarations.

Matrices, airflow angles, the wing tip, the inverse extraction and the nose
path all come from ArmModel. The checked declaration is compared with
data/archive/wind_arm/review/sample_01.md. A second declaration is rendered
beside it so a changed rotation order cannot keep the default collision story.

Run from the repository root::

    python -m domains.wind_arm.dataset_generate
"""
from __future__ import annotations

import argparse
import math
import re

import sympy as sp

from domain_layout import SETS
from domains.wind_arm.model import (
    ABS,
    LIMITS,
    Q,
    ROLE_POSITIVE,
    ArmModel,
    ArmSpec,
    _align,
    _cross,
    fmt_matrix,
    fmt_vector,
    rot_about,
)
from domains.wind_arm.reach import (
    ELL,
    WING_RADIUS,
    format_body_vector,
    format_interval,
    format_scalar,
    format_wind_vector,
    joint_axis,
    max_abs_ratio,
    nose_ranges,
    perpendicular_square,
    principal_rotation,
    recover_joints,
    snap_ratio,
    theta_text,
    wall_reach,
    wing_offset,
)


GOLD = SETS["cross.wind_arm"]["review"] / "sample_01.md"
GENERATED = SETS["cross.wind_arm"]["root"] / "generated"

DIR_ZH = {
    "forward": "前方", "rearward": "后方", "up": "上方", "down": "下方",
    "right": "右侧", "left": "左侧",
}
ROLE_ZH = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
ORDINAL = ("第一根", "第二根", "第三根")
AXES = ("XFD", "YFD", "ZFD", "XMD", "YMD", "ZMD", "XTZ", "YTZ", "ZTZ")
DIRECTED = AXES + tuple(f"-{name}" for name in AXES)
SIX = ("前方", "后方", "上方", "下方", "右侧", "左侧")
SLOT_RE = re.compile(r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}")


class Builder:
    def __init__(self):
        self.lines: list[str] = []
        self.number = 0

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def ask(self, stem: str, answer: str, label: str, options: list[str] | None = None, fmt: str = "无", tail: str = "") -> None:
        self.number += 1
        if options is None:
            shown = "无"
            stored = answer
        else:
            if answer not in options:
                options = [answer, *[item for item in options if item != answer]]
            unique = []
            for item in options:
                if item not in unique:
                    unique.append(item)
            options = unique
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            shown = "；".join(f"{letters[i]}：{item}" for i, item in enumerate(options))
            stored = letters[options.index(answer)]
        self.lines.append(
            f"{self.number}.{stem}"
            f"{{ID:{self.number}, 选项:{shown}, 输出格式:{fmt}, 标签:{label}, 答案:{stored}}}"
            f"{tail}。"
        )

    def expect(self, number: int) -> None:
        if self.number != number:
            raise RuntimeError(f"题号停在 {self.number}，这里应当是 {number}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _changed_definitions(spec: ArmSpec) -> list[str]:
    base = ArmSpec()
    changed = []
    if spec.wind_declared != base.wind_declared or spec.wind_chirality != base.wind_chirality:
        changed.append("风轴系")
    if spec.wrist_declared != base.wrist_declared or spec.wrist_chirality != base.wrist_chirality:
        changed.append("球腕系")
    if spec.body_declared != base.body_declared or spec.body_chirality != base.body_chirality:
        changed.append("体轴系")
    if spec.order != base.order:
        changed.append("绝对方向轴旋转顺序")
    if spec.joint_sign != base.joint_sign:
        changed.append("机械臂旋转角度正方向")
    if spec.install_up != base.install_up:
        changed.append("零位安装")
    if (spec.alpha_toward, spec.beta_toward, spec.gamma_sense) != (base.alpha_toward, base.beta_toward, base.gamma_sense):
        changed.append("理论气流角正方向")
    return changed


def is_checked(spec: ArmSpec) -> bool:
    base = ArmSpec()
    return (
        spec.order == base.order
        and spec.install_up == base.install_up
        and spec.joint_sign == base.joint_sign
        and spec.wind_declared == base.wind_declared
        and spec.wind_chirality == base.wind_chirality
        and spec.wrist_declared == base.wrist_declared
        and spec.wrist_chirality == base.wrist_chirality
        and spec.body_declared == base.body_declared
        and spec.body_chirality == base.body_chirality
        and spec.alpha_toward == base.alpha_toward
        and spec.beta_toward == base.beta_toward
        and spec.gamma_sense == base.gamma_sense
        and spec.role_deg == base.role_deg
    )


def _order_text(spec: ArmSpec) -> str:
    return "、".join(ROLE_ZH[role] for role in spec.order)


def _gamma_word(spec: ArmSpec) -> str:
    return "相同" if spec.gamma_sense > 0 else "相反"


def _pointing(model: ArmModel, kind: str, axis: str) -> str:
    basis = {"wind": model.wind, "wrist": model.wrist, "body": model.body}[kind]
    owner = {"wind": "风洞", "wrist": "球腕物理", "body": "机体"}[kind]
    column = basis[:, "XYZ".index(axis)]
    for name, vector in ABS.items():
        if column == vector:
            return f"{owner}{DIR_ZH[name]}"
    raise ValueError(axis)


def _frame_lines(model: ArmModel, kind: str) -> list[str]:
    spec = model.spec
    declared = {"wind": spec.wind_declared, "wrist": spec.wrist_declared, "body": spec.body_declared}[kind]
    basis = {"wind": model.wind, "wrist": model.wrist, "body": model.body}[kind]
    prefix = {"wind": "FD", "wrist": "MD", "body": "TZ"}[kind]
    owner = {"wind": "风洞", "wrist": "球腕物理", "body": "机体"}[kind]
    role_name = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
    lines = []
    for axis in "XYZ":
        column = basis[:, "XYZ".index(axis)]
        pointing = next(name for name, vector in ABS.items() if column == vector)
        role = {"forward": "axial", "rearward": "axial", "up": "normal", "down": "normal", "right": "lateral", "left": "lateral"}[pointing]
        source = "声明给出" if axis in declared else "由手性推断"
        lines.append(f"- {axis}{prefix}：沿{owner}{role_name[role]}，指向{owner}{DIR_ZH[pointing]}（{source}）。")
    return lines


def _header() -> list[str]:
    return [
        "# 数据 01 球腕正装",
        "",
        "本文件是审阅稿，包含两条独立样本。正运动学样本由关节角推出姿态和三个气流角，再计算翼尖、平移和碰壁。逆运动学样本单独送入，只给出目标气流角和安装矩阵，不给出关节角，也不给出正运动学的计算过程。两条样本的定义和正文互不可见。",
        "",
        "审阅说明、标签、答案、层间对应和生成注记都不送入模型。输出格式规范没有标准答案，作为公共提示送入。送入模型的还有当前样本的定义声明、物理常识、当前子过程名，以及从该子过程开头直到当前句号的正文。当前句之前的空写成答案原文，不写字母。当前句里的空保持为待填槽。每句只有一个待填槽。句号之后的正文不送入。待填槽不含标签和答案。标签只用于计分。",
        "",
        "待填槽在审阅稿中写成 {ID:编号, 选项:A：选项1；B：选项2, 输出格式:格式名, 标签:标签, 答案:标准答案}。没有选项时，选项写无。模型看不到标签和答案。模型把当前句的空写成一个 JSON 对象，键是编号字符串，值是字符串。选择题的值是字母，例如 {\"1\":\"B\"}。矩阵、向量、气流角、公式、关节角和表达式都是字符串。不输出句子原文，不解释。",
        "",
        "评分分开看三件事：JSON 与字段格式是否合规；数学表达式是否等价；数值、符号、向量或矩阵是否正确。等价表达式算数理正确。本例滚转角写成 arcsin(1/2) 或 atan2(1, sqrt(3)) 都对。不要只因反三角函数的写法不同就记成数理错误。",
        "",
        "正运动学样本编号为 1–140，有五个子过程。后一个子过程看不到前面子过程的正文。右翼尖前缘分两层：先写体轴系坐标，再写风轴系坐标。体轴原点的平移单独成层，先算机体顶点到球腕原点的距离，再算平移向量。风轴系这一层在本层开头给出体轴坐标和平移向量，不从上一层的填空里读取。碰壁检测单独成层，求机体顶点的三个可达区间，并用三个端点到球腕原点的距离判定是否存在碰壁姿态。正运动学样本不再包含由气流角反求关节角。",
        "",
        "逆运动学样本编号从 1 重新开始，只有一个子过程。它与正运动学样本配对，但送入时必须独立。正运动学定义里的关节角是正问题的已知量。逆运动学定义里不写这组关节角，也不写 D 的数值。不要把正运动学里出现过的关节角，当成独立逆解能力。",
        "",
        "## 输出格式规范",
        "",
        "- 无：选择题。值只写选项字母。示例：A",
        "- 矩阵：一行 3×3 矩阵，形如 [[a, b, c], [d, e, f], [g, h, i]]。逗号后恰好一个空格。元素可以是整数、分式、sqrt(3)，或 cos(q1)、sin(q1)、cos(q2)、sin(q2)、cos(q3)、sin(q3)。括号内不留空格。负号写在元素前。不要换行，不要使用 LaTeX，不要写度数。示例：[[cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1]]",
        "- 向量：一行三个分量，形如 [a, b, c]。逗号后恰好一个空格。分量顺序是该系坐标轴 X、Y、Z。分量可以含 sqrt、整数、分式，以及 L、L_测、x_r、b、Λ、Γ、tan、sin、cos、cos(q2)、sin(q2)。乘号用 *。不要换行。示例：[L + L_测, 0, 0]",
        "- 气流角：按迎角、侧滑角、滚转角的顺序写三项，相邻两项之间是逗号加一个空格。可以使用 arctan、arcsin、atan2 和 sqrt。不写约等，不写度数。示例：arctan(2/3), arcsin(1/5), atan2(-1, sqrt(3))",
        "- 公式：按 q1、q2、q3 的顺序写三项，分号后恰好一个空格。函数只使用 atan2 和 sqrt。矩阵元素写成 h11 这种形式，行号在前，列号在后。平方写成 ^2。乘号用 *。不写 q1=。示例：atan2(h21, h11); atan2(-h31, sqrt(h11^2 + h21^2)); atan2(h32, h33)",
        "- 关节角：按 q1、q2、q3 的顺序写成 [a, b, c]，单位是度，不写角度符号。逗号后恰好一个空格。示例：[10, -20, 30]",
        "- 距离：一行长度。只使用 L 与 L_测，加号两侧各一个空格。不写单位。示例：L + L_测",
        "- 范围：闭区间 [min, max]。逗号后恰好一个空格。负号写在下界或上界之前。下界和上界可以含 L、L_测、sqrt(3)、分式和乘号 *。不写单位，不写度数。示例：[-(L + L_测), L + L_测]",
        "- 表达式：一行。乘号用 *。方括号表示列向量。D*[a, b, c] 表示矩阵 D 左乘该列向量。可以保留矩阵名 D，可以含 sqrt、^2、max 和比较号。比较号写成 > 或 >=，两侧各一个空格。不要展开 D，不要换行，不要使用 LaTeX。示例：L + L_测 >= L_风/2",
        "",
    ]


def _definition(model: ArmModel, forward: bool) -> list[str]:
    spec = model.spec
    order = _order_text(spec)
    alpha = "下方" if spec.alpha_toward == "down" else "上方"
    beta = "右侧" if spec.beta_toward == "right" else "左侧"
    gamma = _gamma_word(spec)
    if not forward:
        return _inverse_definition(model, order, alpha, beta, gamma)
    q_text = "，".join(f"q{index + 1}={model.q_degrees[index]}°" for index in range(3))
    role_text = "，".join(f"{ROLE_ZH[role]}轴为 {spec.role_deg[role]}°" for role in ("normal", "lateral", "axial"))
    lines = [
        "## 定义声明",
        "",
        "固定约定",
        "- 向量用列向量表示，变换矩阵左乘列向量。父坐标系中的分量等于姿态矩阵乘子坐标系中的分量。坐标分量按该系 X、Y、Z 的顺序书写。",
        "- 绝对方向与坐标轴分开声明。后文写前方、后方、上方、下方、右侧、左侧，以及轴向正方向、法向正方向、侧向正方向，都是绝对方向，不是坐标轴 X、Y、Z 的正方向。",
        "- 球腕姿态旋转采用动轴内旋。三个角度按绝对方向轴的旋转顺序依次记为 q1、q2、q3。关节矩阵记为 R(q1)、R(q2)、R(q3)。最先转动的矩阵写在乘积最左侧。Rz、Ry、Rx 分别表示绕坐标轴 Z、Y、X 的旋转，行和列按 X、Y、Z 排列。这三个记号不携带绝对方向。关节角的正方向绑定在所绕的绝对方向轴上，按该绝对正方向的右手定则，不随坐标轴反向而翻转。绝对方向轴的旋转顺序是可变定义，固定约定不指定先后。",
        "- 定义声明不给出 Rx、Ry、Rz 的元素。元素由转动后绝对方向的去向、坐标轴是否与该绝对正方向同向、以及该绝对方向上的右手定则共同确定。",
        "- 关节范围绑定在绝对方向轴上，不随旋转顺序改到另一根轴上。法向轴的转角范围是 -60° 到 60°，在这个范围内余弦为正。侧向轴和轴向轴的转角范围都是 -90° 到 90°。",
        "- 计算时零位安装只由安装矩阵 S 表示，不再另乘安装矩阵。",
        "- 本例忽略弹性变形和气流偏角，按刚体、准静态运动学计算。",
        "- 迎角和侧滑角只由体轴系中的空速分量计算。滚转角按理论气流角中的零参考，由机体前方、机体上方和风洞上方计算。三个角并列，合称理论气流角。关节角本身不是气流角。",
        "",
        "绝对方向",
        "球腕的前方、上方、右侧固连在球腕上，随球腕转动。后文称球腕物理前方、球腕物理上方、球腕物理右侧。风洞的前方、上方、右侧固定在风洞上。球腕处于初始位置时，球腕物理前方、物理上方、物理右侧分别与风洞前方、上方、右侧重合。关节转动之后，球腕物理方向离开这一重合。机体的绝对方向相对机体本身，不随这个初始重合而改写。",
        "风洞、球腕、机体各有三根方向轴。三套轴的正方向定义相同：轴向正方向是前方，法向正方向是上方，侧向正方向是右侧。六个方向是这三根轴上的两个指向。",
        "下文凡写轴向正方向、法向正方向、侧向正方向，以及机体轴向的正方向、风洞轴向的正方向，都是这三根轴的正方向，不是坐标轴 X、Y、Z 的正方向。球腕物理轴向、球腕物理法向、球腕物理侧向也是这三根轴，只是固连在球腕上。坐标轴声明只是指定每根坐标轴沿哪根轴、指向哪一侧，可以与正方向相同，也可以相反。",
        "三个系共用同一套指向关系：前方 × 右侧 = 下方。沿前方看去，右侧在右手侧。风洞下方铅垂向下。机体和球腕的上下是各自固连方向，不称为铅垂向下。这个指向关系不随定义改变。",
        "机体的三轴六方向相对机体本身：",
        "- 轴向：连接机体尾部与机体头部。正方向朝前，指向机体头部。后方指向机体尾部。",
        "- 法向：机体的上下轴。正方向朝上。下方是其反方向。",
        "- 侧向：机体的左右轴。正方向朝右。左侧是其反方向。",
        "球腕的三轴六方向是球腕物理方向，相对球腕本身：",
        "- 轴向：球腕物理前后轴。正方向是球腕物理前方。后方是其反方向。",
        "- 法向：球腕物理上下轴。正方向是球腕物理上方。下方是其反方向。",
        "- 侧向：球腕物理左右轴。正方向是球腕物理右侧。左侧是其反方向。",
        "三轴球腕的三根关节轴并不始终固连在同一套末端方向上。第 i 关节所称的法向、侧向、轴向，是该关节动作前中间随动参考系中的对应方向。末端球腕系只用来描述关节全部转完之后的最终姿态。",
        "风洞的三轴六方向相对风洞，固定不动：",
        "- 轴向：风洞轴线。正方向朝前。前方是来流，与空速方向一致，是气流方向的反方向。后方是其反方向。",
        "- 法向：风洞铅垂轴。正方向朝上，上方铅垂向上。下方铅垂向下。",
        "- 侧向：风洞水平横轴。正方向朝右。左侧是其反方向。",
        "",
        "场景",
        "三转动关节球腕的三根轴线交于固定点 O。O 是球腕系原点，也是风轴系原点。球腕中心不移动。球腕末端连接一根支杆，支杆顶部从机体底部插入机体内部一段，机体刚性装在这根支杆上。角度机构运动时，球腕带动支杆和机体一起转动。体轴系原点在机体顶点，不在质心，也不在球腕原点。转动之后，机体顶点离开风轴系原点，两点之间的位置要按姿态另行计算。球腕原点在风洞横截面正中央。机械臂不能自由穿过通道壁。",
        "",
        "可变定义",
        "下列七项可以单独改写，绝对方向正文不随它们改变：风轴系、球腕系、体轴系、绝对方向轴旋转顺序、机械臂旋转角度正方向、零位安装、理论气流角正方向。"
        + (
            "本条给出的是默认声明。"
            if not _changed_definitions(spec)
            else "本条给出的是这一份样本的声明，已改动" + "、".join(_changed_definitions(spec)) + "。"
        )
        + "绝对方向轴旋转顺序是人为规定的扰动，只改动轴内旋的先后，不改各轴的正方向和转角范围。",
        "",
        f"风轴系（{'右手系' if spec.wind_chirality == 'right' else '左手系'}）",
        *_frame_lines(model, "wind"),
        "",
        f"球腕系（{'右手系' if spec.wrist_chirality == 'right' else '左手系'}）",
        *_frame_lines(model, "wrist"),
    ]
    if model.wrist == model.wind:
        lines.append("关节角都为 0、且尚未计入安装矩阵时，XMD、YMD、ZMD 分别与 XFD、YFD、ZFD 同向。")
    else:
        lines.append("关节角都为 0、且尚未计入安装矩阵时，球腕坐标轴与风轴坐标轴按上面两条声明对齐，不默认同向。")
    lines += [
        "",
        f"体轴系（{'右手系' if spec.body_chirality == 'right' else '左手系'}）",
        *_frame_lines(model, "body"),
        "体轴系原点在机体顶点。",
        "",
        "原点与支杆",
        "- 球腕系原点与风轴系原点重合，都是 O。O 不随关节角移动。",
        "- 体轴系原点在机体顶点。",
        "- L 是机体全长，从机体顶点量到机体底部。",
        "- 支杆顶部插入机体的那一段位于机体全长之内。",
        "- L_测 是机体底部到球腕原点的距离，不含已经插入机体的那一段。",
        "- O 位于机体底部后方。沿机体前方，顺序是 O、机体底部、机体顶点。三点沿机体轴向共线。",
        "- 球腕转动后，体轴系原点相对风轴系原点产生平移。平移向量是体轴系原点在风轴系中的位置，随姿态变化。定义不给出顶点到 O 的距离和平移向量的成品，由过程算出。",
        "",
        "风洞通道",
        "- 球腕原点位于风洞横截面正中央。",
        "- 风洞等效为沿风轴轴向的柱体通道。横截面是正方形，边长为 L_风。正方形的边与风洞上方、右侧平行。",
        "- 碰壁检测检查机体顶点、右翼顶点、左翼顶点。右翼顶点是右翼尖前缘。左翼顶点与右翼顶点的前后、高低相同，左右相反。",
        "- 支杆、机翼表面和机体其他部位不参加检测。检测扫过声明的关节范围，不用单独某一组关节角代替整个范围。",
        "- 接触壁面即算碰壁。定义不给出半宽、顶点可达区间和最终判据，由过程算出。",
        "",
        "绝对方向轴旋转顺序",
        "这是可变定义，也是扰动项。它规定动轴内旋时先转哪根绝对方向轴，不规定坐标轴 X、Y、Z 的名称顺序。三根轴各出现一次。",
        f"本例顺序是{order}。",
    ]
    for index, role in enumerate(spec.order):
        step = ("最先转", "其次转", "最后转")[index]
        lines.append(f"- q{index + 1} {step}，绕球腕绝对{ROLE_ZH[role]}轴。该轴的正方向是{DIR_ZH[ROLE_POSITIVE[role]]}。")
    lines += [
        f"因此本例 q1 使用{ROLE_ZH[spec.order[0]]}轴的范围，q2 使用{ROLE_ZH[spec.order[1]]}轴的范围，q3 使用{ROLE_ZH[spec.order[2]]}轴的范围。",
        "",
        "机械臂旋转角度正方向",
        "这是可变定义。正方向一律按所绕绝对方向轴的右手定则，不按坐标轴 X、Y、Z 的指向另行规定。",
    ]
    for role in ("normal", "lateral", "axial"):
        sense = "相同" if spec.joint_sign[role] > 0 else "相反"
        lines.append(f"- 绕{ROLE_ZH[role]}轴时，正方向与绕{DIR_ZH[ROLE_POSITIVE[role]]}的右手定则{sense}。")
    lines += [
        "",
        "矩阵",
        "- R 是球腕系相对风轴系的姿态。它把球腕系分量映射到风轴系分量。",
        "- S 是体轴系相对球腕系的固定安装矩阵。它把体轴系分量映射到球腕系分量。",
        "- D 是体轴系相对风轴系的姿态。",
        "",
        "零位安装",
        f"关节角都为 0 时，机体轴向正方向与风轴轴向正方向重合。机体法向正方向与{DIR_ZH[spec.install_up]}重合。",
        "",
        "空速",
        "空速方向与实际气流运动方向相反。前方来流时，空速指向风洞轴向正方向。空速在 XFD、YFD、ZFD 上的分量由风轴系声明决定。",
        "",
        "理论气流角",
        "迎角、侧滑角、滚转角并列，合称理论气流角。三者的正方向是可变定义。",
        f"- 理论迎角 α：空速在机体对称面上的投影与机体轴向正方向的夹角。投影处于机体轴向{alpha}时为正。",
        f"- 理论侧滑角 β：空速向量与机体对称面的夹角，空速在机体{beta}为正。",
        f"- 理论滚转角 γ：记机体前方单位向量为 t，机体上方单位向量为 n，风洞上方单位向量为 e_U。参考上方向 u 是 e_U 在垂直于 t 的平面上的投影再归一化，u=(e_U-(e_U·t)*t)/||e_U-(e_U·t)*t||。令 r=t×u。γ=atan2(n·r, n·u)。正方向与绕机体绝对轴向正方向（前方）的右手定则{gamma}，反方向为负。当机体前方与风洞上方向平行时，该滚转角未定义。单独变换滚转角时，旋转轴是机体绝对轴向。",
        "",
        "气动外形",
        "位置参考点是右翼尖前缘点。翼尖后缘点、左翼尖前缘点和翼根前缘点都不是参考点。",
        "外形参数如下。",
        "- b：翼展，左右翼尖前缘在机体绝对侧向上的间距。左右翼尖前缘的前后、高低相同，因此两点之间的距离也是 b。从对称面到一侧翼尖的侧向坐标是 b/2。这个侧向坐标不随上反角改变。",
        "- Λ：前缘在机体轴向—侧向平面投影中的后掠角。右翼尖前缘位于翼根前缘后方时为正。前后偏移等于该侧向坐标乘 tan(Λ)，再取负号。",
        "- Γ：上反角。翼尖相对翼根的法向偏移与侧向偏移之比为 tan(Γ)。右翼尖高于翼根前缘时为正。增大上反角会增大斜向长度，侧向间距保持为 b。",
        "- x_r：翼根前缘沿机体绝对轴向的坐标，从机体顶点量起，指向机体前方为正。它写入绝对轴向正方向所对应的那根机体坐标轴。翼根前缘的绝对法向坐标和绝对侧向坐标都是 0。",
        "右翼尖前缘点的前后位置使用前缘后掠角和侧向坐标。它的高低使用上反角和侧向坐标。前后、左右、高低分别沿绝对轴向、绝对侧向、绝对法向，再按坐标轴声明落入 XTZ、YTZ、ZTZ。",
        "",
        "本例关节角",
        f"{q_text}。{role_text}。",
    ]
    return lines


def _inverse_limits(spec: ArmSpec) -> str:
    if spec.order == ("normal", "lateral", "axial"):
        return "关节范围绑定在绝对方向轴上。法向轴 q1 的范围是 -60° 到 60°，余弦为正。侧向轴 q2 和轴向轴 q3 的范围都是 -90° 到 90°。"
    parts = []
    for index, role in enumerate(spec.order):
        low, high = LIMITS[role]
        extra = "，余弦为正" if role == "normal" else ""
        parts.append(f"{ROLE_ZH[role]}轴 q{index + 1} 的范围是 {low}° 到 {high}°{extra}")
    return "关节范围绑定在绝对方向轴上。" + "。".join(parts) + "。"


def _inverse_branch(spec: ArmSpec) -> str:
    low, high = LIMITS[spec.order[1]]
    if (low, high) == (-90, 90):
        return "当前球腕逆解使用非奇异分支 -90°<q2<90°。完整机械限位仍允许 q2=±90°，但边界奇异姿态不纳入唯一逆解任务。"
    return f"当前球腕逆解使用非奇异分支 {low}°<q2<{high}°。q2 的机械范围不到达 ±90°，边界奇异姿态不在该轴的行程内。"


def _inverse_joint_sentence(spec: ArmSpec) -> str:
    if all(spec.joint_sign[role] > 0 for role in spec.order):
        return "- 关节矩阵记为 R(q1)、R(q2)、R(q3)。正方向都与绕所对应绝对正方向的右手定则相同。定义不给出这三个矩阵的元素。R=R(q1)*R(q2)*R(q3)。"
    parts = []
    for index, role in enumerate(spec.order):
        word = "相同" if spec.joint_sign[role] > 0 else "相反"
        parts.append(f"q{index + 1} 与绕{DIR_ZH[ROLE_POSITIVE[role]]}的右手定则{word}")
    body = "，".join(parts)
    return f"- 关节矩阵记为 R(q1)、R(q2)、R(q3)。{body}。定义不给出这三个矩阵的元素。R=R(q1)*R(q2)*R(q3)。"


def _inverse_definition(model: ArmModel, order: str, alpha: str, beta: str, gamma: str) -> list[str]:
    spec = model.spec
    wind = fmt_vector(model.airspeed_wind)
    row = "，也就是 D 的第一行" if wind == "[1, 0, 0]" else ""
    low, high = LIMITS[spec.order[1]]
    middle = ROLE_ZH[spec.order[1]]
    if _in_operating_range(model):
        given = "当前给定量满足这些条件。"
    else:
        given = "当前给定量不全部满足这些条件。"
    return [
        "## 定义声明",
        "",
        "固定约定",
        "- 向量用列向量表示，变换矩阵左乘列向量。父坐标系中的分量等于姿态矩阵乘子坐标系中的分量。坐标分量按该系 X、Y、Z 的顺序书写。",
        f"- 风轴系：XFD 指向{_pointing(model, 'wind', 'X')}，YFD 指向{_pointing(model, 'wind', 'Y')}，ZFD 指向{_pointing(model, 'wind', 'Z')}。",
        f"- 球腕系：XMD 指向{_pointing(model, 'wrist', 'X')}，YMD 指向{_pointing(model, 'wrist', 'Y')}，ZMD 指向{_pointing(model, 'wrist', 'Z')}。球腕物理方向固连在球腕上。初始位置时，它们分别与风洞前方、上方、右侧重合。",
        f"- 体轴系：XTZ 指向{_pointing(model, 'body', 'X')}，YTZ 指向{_pointing(model, 'body', 'Y')}，ZTZ 指向{_pointing(model, 'body', 'Z')}。",
        f"- 球腕姿态旋转采用动轴内旋。绝对方向轴旋转顺序是可变定义。本例顺序是{order}，角度依次为 q1、q2、q3。最先转动的矩阵写在最左侧。",
        _inverse_joint_sentence(spec),
        "- 各关节所称的法向、侧向、轴向，是该关节动作前中间随动参考系中的对应方向。末端球腕系描述最终姿态。三根关节轴并不始终固连在同一套末端方向上。",
        f"- {_inverse_limits(spec)}",
        f"- {_inverse_branch(spec)}",
        f"- 关节角都为 0 时，机体轴向正方向与风轴轴向正方向重合。机体法向正方向与{DIR_ZH[spec.install_up]}重合。S 是体轴系相对球腕系的安装矩阵。",
        "- D 是体轴系相对风轴系的姿态。",
        f"- 前方来流时，空速指向风洞前方。按 XFD、YFD、ZFD 的顺序，风轴系中的空速单位向量是 {wind}。它变到体轴系后，等于 D 的转置乘这个列向量{row}。",
        "- 本样本不给出关节角，也不给出 D 的数值。",
        "",
        "理论气流角",
        f"- 理论迎角 α：空速在机体对称面上的投影与机体轴向正方向的夹角。投影处于机体轴向{alpha}时为正。",
        f"- 理论侧滑角 β：空速向量与机体对称面的夹角，空速在机体{beta}为正。",
        f"- 理论滚转角 γ：记机体前方单位向量为 t，机体上方单位向量为 n，风洞上方单位向量为 e_U。参考上方向 u 是 e_U 在垂直于 t 的平面上的投影再归一化，u=(e_U-(e_U·t)*t)/||e_U-(e_U·t)*t||。令 r=t×u。γ=atan2(n·r, n·u)。正方向与绕机体绝对轴向正方向（前方）的右手定则{gamma}，反方向为负。当机体前方与风洞上方向平行时，该滚转角未定义。",
        f"本例给定的三个气流角是 {model.airflow_text()}。",
        f"本样本只在 -90°<α<90°、-90°<β<90°、机体前方轴不与风洞上方向平行、且{middle}轴处在非奇异分支 {low}°<q2<{high}° 内时，由三个气流角求关节角。{given}",
    ]


def _sense(model: ArmModel) -> str:
    spec = model.spec
    return (
        "轴向正方向、法向正方向、侧向正方向分别是前方、上方、右侧。机体轴向的正方向、风洞轴向的正方向都是这个绝对前方，不是坐标轴 X、Y、Z 的正方向。"
        "球腕物理前方、物理上方、物理右侧固连在球腕上，初始位置与风洞前方、上方、右侧重合，转动后离开这一重合。"
        "坐标轴可以与正方向相同，也可以相反。前方 × 右侧 = 下方，这个关系不随坐标轴声明改变。"
        "单独变换迎角时，旋转轴是绝对侧向轴。单独变换侧滑角时，旋转轴是绝对法向轴。单独变换滚转角时，旋转轴是绝对轴向轴。"
        f"滚转角从风洞上方在垂直于机体前方的平面上的投影量起，正方向与绕机体绝对轴向正方向的右手定则{_gamma_word(spec)}，反方向为负。"
        "机体前方与风洞上方向平行时滚转角未定义。关节角的正方向按所绕绝对方向轴的右手定则，不随坐标轴反向而翻转。"
        f"动轴内旋的先后由绝对方向轴旋转顺序规定，本例是{_order_text(spec)}。"
        "关节范围绑定在绝对方向轴上：法向轴是 -60° 到 60°，侧向轴和轴向轴都是 -90° 到 90°。"
        "理论气流角的正方向以可变定义为准。"
        f"零位时机体轴向正方向与风轴轴向正方向重合。机体法向正方向与{DIR_ZH[spec.install_up]}重合。"
        "球腕关节转动通过支杆带动机体作刚体转动。本例忽略弹性变形，也忽略气流偏角。"
        "空速与实际气流运动方向相反。前方来流时，空速指向风洞轴向正方向。"
        "同一个物理向量在不同坐标系中的分量不同。向量用列向量表示，姿态矩阵左乘列向量，把子坐标系的分量映射到父坐标系。"
        "父坐标系的分量映射回子坐标系时，使用该姿态矩阵的转置。"
        "迎角和侧滑角的正方向以理论气流角的定义为准。滚转角的零参考和正负以理论气流角的定义为准。"
        "前缘后掠使翼尖相对翼根靠后。上反使翼尖相对翼根升高。翼展按侧向间距定义时，上反不减小侧向投影，斜向长度随上反角增大。"
        "球腕系原点与风轴系原点重合。体轴系原点在机体顶点。O 在机体底部后方，沿前方依次是 O、底部、顶点。"
        "机体顶点到球腕原点要沿机体轴向把机体全长和底部到球腕的距离接起来。支杆插入机体的部分已经在机体全长里。"
        "姿态变化后，体轴原点在风轴系中的位置由姿态矩阵乘这个轴向偏移得到，两个原点不再重合。"
        "风洞通道的壁限制上下和左右，不限制轴向。接触壁面即算碰壁。"
    )


def _inverse_sense(model: ArmModel) -> str:
    low, high = LIMITS[model.spec.order[1]]
    middle = ROLE_ZH[model.spec.order[1]]
    if model.spec.order[0] == "normal":
        cosine = "q1 在声明范围内余弦为正。"
    else:
        cosine = f"q1 是{ROLE_ZH[model.spec.order[0]]}轴，是否余弦为正按该轴的范围判断。"
    return (
        "向量用列向量表示，姿态矩阵左乘列向量。姿态矩阵的转置把父坐标系分量映射回子坐标系。"
        f"滚转角从风洞上方在垂直于机体前方的平面上的投影量起，正方向与绕机体前方轴的右手定则{_gamma_word(model.spec)}，反方向为负。"
        "机体前方与风洞上方向平行时滚转角未定义。"
        f"本例动轴内旋顺序是{_order_text(model.spec)}。{cosine}{middle}轴的唯一逆解只使用 {low}°<q2<{high}°。"
    )


def _sign(value) -> str:
    number = sp.simplify(value)
    if number == 0:
        return "零"
    if number.is_real and bool(number.is_negative):
        return "负"
    if number.is_real and bool(number.is_positive):
        return "正"
    return "负" if sp.sstr(number).startswith("-") else "正"


def _opposite(direction: str) -> str:
    return {"forward": "rearward", "rearward": "forward", "up": "down", "down": "up", "right": "left", "left": "right"}[direction]


def _axis_positive_zh(model: ArmModel, direction: str) -> str:
    """Positive sense of the coordinate axis that carries this absolute direction."""
    name = model.axis_name("body", direction)
    pointed = direction if not name.startswith("-") else _opposite(direction)
    return DIR_ZH[pointed]


def _classic_airspeed_rows(model: ArmModel) -> bool:
    """The default row formulas assume wind X is forward and body XYZ is forward, up, right."""
    return (
        model.wind == sp.eye(3)
        and model.body == sp.eye(3)
        and model.spec.alpha_toward == "down"
        and model.spec.beta_toward == "right"
        and model.spec.gamma_sense > 0
    )


def _abs_component_vector(model: ArmModel) -> sp.Matrix:
    alpha, beta = model.alpha, model.beta
    forward = sp.cos(alpha) * sp.cos(beta)
    if model.spec.alpha_toward == "down":
        up = -sp.sin(alpha) * sp.cos(beta)
    else:
        up = sp.sin(alpha) * sp.cos(beta)
    if model.spec.beta_toward == "right":
        right = sp.sin(beta)
    else:
        right = -sp.sin(beta)
    return sp.simplify(sp.Matrix([forward, up, right]))


def _abs_component_text(model: ArmModel) -> str:
    up = "-sin(α)*cos(β)" if model.spec.alpha_toward == "down" else "sin(α)*cos(β)"
    right = "sin(β)" if model.spec.beta_toward == "right" else "-sin(β)"
    return f"[cos(α)*cos(β), {up}, {right}]"


def _in_operating_range(model: ArmModel) -> bool:
    alpha = float(model.alpha)
    beta = float(model.beta)
    if not (-math.pi / 2 < alpha < math.pi / 2 and -math.pi / 2 < beta < math.pi / 2):
        return False
    low, high = model.limit_of_q(1)
    if not (low < model.q_degrees[1] < high):
        return False
    forward = sp.simplify(model.D * (model.body.T * ABS["forward"]))
    upward = model.wind.T * ABS["up"]
    return sp.simplify(_cross(forward, upward)) != sp.Matrix([0, 0, 0])


def _axis_direction(basis: sp.Matrix, index: int) -> str:
    column = basis[:, index]
    for name, value in ABS.items():
        if column == value:
            return name
    raise ValueError("坐标轴没有落到绝对方向上")


def _wrist_column(model: ArmModel, direction: str) -> str:
    components = model.body.T * ABS[direction]
    return fmt_vector(sp.simplify(model.S * components))


def _combined_degree(model: ArmModel):
    install = principal_rotation(model.S)
    axis, epsilon = joint_axis(model, 2)
    if install is None or install[0] != axis:
        return None
    return install[1] + epsilon * model.q_degrees[2]


def _factored_pose(model: ArmModel) -> str:
    combined = _combined_degree(model)
    if combined is None or model.q_degrees[0] != 0:
        return fmt_matrix(model.D)
    q2_axis, epsilon = joint_axis(model, 1)
    q3_axis, _epsilon = joint_axis(model, 2)
    names = {0: "X", 1: "Y", 2: "Z"}
    q2_coordinate = epsilon * model.q_degrees[1]
    return f"绕 {names[q2_axis]} 的 {q2_coordinate}° 乘绕 {names[q3_axis]} 的 {combined}°"


def _cross_name(first: str, second: str) -> str:
    vector = _cross(ABS[first], ABS[second])
    for name, value in ABS.items():
        if vector == value:
            return name
    raise ValueError("叉积不是六个绝对方向之一")


def _trig_text(value) -> str:
    exact = snap_ratio(float(value))
    if exact is None:
        raise ValueError(f"三角函数取值不能化简：{value}")
    if exact == 0:
        return "0"
    if exact == 1:
        return "1"
    if exact == -1:
        return "-1"
    negative = exact < 0
    magnitude = sp.simplify(-exact if negative else exact)
    body = sp.sstr(magnitude).replace("**", "^")
    return f"-{body}" if negative else body


def _trig_bounds(kind: str, low: int, high: int):
    samples = [low, high]
    if kind == "sin":
        for hit in (-90, 90):
            if low <= hit <= high:
                samples.append(hit)
        values = [sp.sin(sp.rad(item)) for item in samples]
    else:
        if low <= 0 <= high:
            samples.append(0)
        values = [sp.cos(sp.rad(item)) for item in samples]
    return min(values), max(values)


def _shift(model: ArmModel) -> str:
    return format_body_vector(model.vertex_body(ELL))


def _wing_radius(model: ArmModel) -> str:
    total = wing_offset(model)
    square = sum((sp.simplify(total[index] ** 2) for index in range(3)), sp.Integer(0))
    span, sweep, height, axial = sp.symbols("b Lambda Gamma x_r")
    target = (ELL + axial - (span / 2) * sp.tan(sweep)) ** 2 + (span / 2) ** 2 * (1 + sp.tan(height) ** 2)
    if sp.simplify(sp.expand(square - target)) == 0:
        return WING_RADIUS
    parts = " + ".join(f"({format_body_component(total[index])})^2" for index in range(3))
    return f"sqrt({parts})"


def format_body_component(expr) -> str:
    from domains.wind_arm.reach import format_body_component as _format

    return _format(expr)


def render(model: ArmModel, reached: float | None = None) -> str:
    checked = is_checked(model.spec)
    builder = Builder()
    builder.lines.extend(_header())
    builder.add("# 正运动学样本")
    builder.add("")
    builder.lines.extend(_definition(model, True))
    builder.add("")
    builder.add("## 物理常识")
    builder.add("")
    builder.add(_sense(model))
    builder.add("")
    builder.add("## 完整过程")
    builder.add("")
    builder.add("下面每个槽都带标签和标准答案，便于直接审阅。这是审阅字段，不是模型在待填空上看到的字段。")
    builder.add("")
    _forward(builder, model, checked, reached)
    forward_count = builder.number
    builder.add("# 逆运动学样本")
    builder.add("")
    builder.add("本样本单独送入。定义中不出现正运动学样本的关节角，也不出现 D 的数值。")
    builder.add("")
    builder.number = 0
    builder.lines.extend(_definition(model, False))
    builder.add("")
    builder.add("## 物理常识")
    builder.add("")
    builder.add(_inverse_sense(model))
    builder.add("")
    builder.add("## 完整过程")
    builder.add("")
    _inverse(builder, model, checked)
    builder.lines.extend(_notes(model, checked, reached))
    text = builder.text()
    old = "正运动学样本编号为 1–140"
    new = f"正运动学样本编号为 1–{forward_count}"
    if old not in text:
        raise RuntimeError("标题里没有正运动学编号")
    return text.replace(old, new, 1)


def _forward(builder: Builder, model: ArmModel, checked: bool, reached: float | None) -> None:
    spec = model.spec
    right_handed = spec.wind_chirality == spec.wrist_chirality == spec.body_chirality == "right"
    builder.add("### 机体姿态与气流角")
    builder.add("")
    builder.add("风轴系、球腕系和体轴系都是右手系。" if right_handed else "每个系的手性以定义声明为准。")
    declared = (spec.wind_declared, spec.wrist_declared, spec.body_declared)
    if all(set(item) == {"X", "Y"} for item in declared):
        builder.add("每个系先声明轴向坐标轴和法向坐标轴的指向，侧向坐标轴由手性推断。")
    else:
        builder.add("每个系未声明的那根坐标轴由手性推断。")
    for frame, direction, stem in (
        ("wind", "up", "风轴系的绝对法向正方向（上方）对应"),
        ("wind", "forward", "风轴系的绝对轴向正方向（前方）对应"),
        ("wind", "right", "风轴系的绝对侧向正方向（右侧）对应"),
        ("wrist", "up", "球腕系的物理上方对应"),
        ("wrist", "forward", "球腕系的物理前方对应"),
        ("wrist", "right", "球腕系的物理右侧对应"),
        ("body", "up", "体轴系的绝对法向正方向（上方）对应"),
        ("body", "forward", "体轴系的绝对轴向正方向（前方）对应"),
        ("body", "right", "体轴系的绝对侧向正方向（右侧）对应"),
    ):
        builder.ask(stem, model.axis_name(frame, direction), "C1", list(DIRECTED))
    builder.add("下面每个关节的去向，都相对于该关节动作前的随动参考系，不是相对转动结束后的风洞固定方向。")
    roles = ["轴向", "法向", "侧向"]
    wrist_axes = ["XMD", "YMD", "ZMD", "-XMD", "-YMD", "-ZMD"]
    turns = ["上方", "下方", "左侧", "右侧"]
    for index, role in enumerate(spec.order):
        builder.ask(f"本例绝对方向轴旋转顺序的{ORDINAL[index]}是", ROLE_ZH[role], "C1", roles)
        builder.ask("它对应的当前球腕坐标轴是", model.axis_name("wrist", ROLE_POSITIVE[role]), "C2", wrist_axes)
        sense = "相同" if spec.joint_sign[role] > 0 else "相反"
        builder.ask(f"q{index + 1} 的正方向与绕这根绝对方向的右手定则", sense, "C1", ["相同", "相反"])
        if role == "axial":
            builder.ask(f"在该关节动作前的随动参考系里，q{index + 1}>0 时，球腕物理上方转向", model.turn_of_up(index), "A1", turns)
        else:
            builder.ask(f"在该关节动作前的随动参考系里，q{index + 1}>0 时，球腕物理前方转向", model.turn_of_forward(index), "A1", turns)
        builder.ask(
            f"R(q{index + 1}) 的行和列按球腕坐标轴 X、Y、Z 排列，定义不给出它的元素，R(q{index + 1})=",
            fmt_matrix(model.joint_matrices[index]), "A2", None, "矩阵",
        )
    builder.ask("零位时机体轴向正方向指向", "来流方向", "C1", ["来流方向", "风洞上方", "风洞右侧", "风洞下方"])
    builder.ask("零位时机体法向正方向与", DIR_ZH[spec.install_up], "C1", ["上方", "下方", "右侧", "左侧"], tail="重合")
    builder.add("零位时球腕物理方向与风洞方向重合。")
    builder.ask("机体上方因此与球腕的物理方向", DIR_ZH[spec.install_up], "C2", ["前方", "上方", "右侧", "下方"], tail="重合")
    builder.add("机体前方与风洞前方重合。")
    builder.add(f"机体上方与风洞{DIR_ZH[spec.install_up]}重合。")
    right_abs = _cross_name("forward", spec.install_up)
    builder.ask("按机体前方 × 机体上方 = 机体右侧，机体右侧指向风洞的绝对方向", DIR_ZH[right_abs], "A1", ["上方", "下方", "左侧", "右侧"])
    builder.ask("该方向因此与球腕的物理方向", DIR_ZH[right_abs], "C2", ["前方", "上方", "右侧", "下方"], tail="重合")
    builder.add("S 的列按 XTZ、YTZ、ZTZ 排列，行按 XMD、YMD、ZMD 排列。")
    columns = ["[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]", "[0, -1, 0]", "[0, 0, -1]", "[-1, 0, 0]"]
    for direction, stem in (
        ("forward", "机体前方轴在球腕坐标中的列是"),
        ("up", "机体上方轴在球腕坐标中的列是"),
        ("right", "机体右侧轴在球腕坐标中的列是"),
    ):
        builder.ask(stem, _wrist_column(model, direction), "A2", columns)
    builder.ask("因此安装矩阵 S 为", fmt_matrix(model.S), "A2", None, "矩阵")
    install = principal_rotation(model.S)
    axis_names = ["XMD", "YMD", "ZMD"]
    if install is None:
        builder.ask("由 S 的元素，它等于绕球腕坐标轴", "不能看成单轴旋转", "A2", axis_names + ["不能看成单轴旋转"])
        builder.ask("该转角是", "不能写成单角", "A2", ["0°", "90°", "-90°", "180°", "不能写成单角"])
    else:
        builder.ask("由 S 的元素，它等于绕球腕坐标轴", axis_names[install[0]], "A2", axis_names, tail="的旋转")
        builder.ask("该转角是", f"{install[1]}°", "A2", ["0°", "90°", "-90°", "180°"])
    builder.ask(
        "三个关节矩阵按动轴内旋相乘，最先转动的放在最左侧，球腕姿态 R(q) 为",
        "R(q1)*R(q2)*R(q3)", "C2",
        ["R(q1)*R(q2)*R(q3)", "R(q3)*R(q2)*R(q1)", "R(q2)*R(q1)*R(q3)"],
    )
    builder.ask("机体相对球腕的安装矩阵为 S，球腕相对风轴系的姿态为 R，机体相对风轴系的姿态矩阵 D 为", "R*S", "A2", ["R*S", "S*R"])
    unit = "单位阵" if model.q_degrees[0] == 0 else "不是单位阵"
    builder.ask(f"代入 q1={model.q_degrees[0]}° 后，R(q1) 是", unit, "A2", ["单位阵", "不是单位阵", "R(q2)", "R(q3)"])
    same_axis = _combined_degree(model) is not None
    builder.add("R(q3) 与 S 绕同一根坐标轴。" if same_axis else "R(q3) 与 S 不绕同一根坐标轴。")
    builder.add(f"q3={model.q_degrees[2]}°。")
    combined = _combined_degree(model)
    builder.ask(
        "该轴上 q3 与前面得到的安装转角的合成转角是" if same_axis else "q3 与安装转角不能合成，结果是",
        f"{combined}°" if combined is not None else "不能合成",
        "A2", ["30°", "-30°", "-150°", "90°", "不能合成"],
    )
    phrase = _factored_pose(model)
    if phrase.startswith("绕"):
        stem = f"q2={model.q_degrees[1]}°，且 R(q1) 是{unit}，R(q2) 与上一合成转角相乘后，机体姿态矩阵是"
    else:
        stem = "把关节角代入 R(q1)*R(q2)*R(q3)*S 后，机体姿态矩阵是"
    builder.ask(stem, phrase, "A2", [phrase, "绕 Y 的 30° 乘绕 X 的 30°", "绕 Z 的 30° 乘绕 X 的 -60°"])
    builder.add("在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。")
    for index, matrix in enumerate(model.joint_matrices, start=1):
        builder.ask(f"R(q{index})的展开式为", fmt_matrix(matrix), "A2", None, "矩阵")
    builder.ask("S的展开式为", fmt_matrix(model.S), "A2", None, "矩阵")
    builder.ask("由此算出的 D 为", fmt_matrix(model.D), "A2", None, "矩阵")
    builder.ask("空速指向风洞轴向正方向，按 XFD、YFD、ZFD 的顺序，风轴系中的空速单位向量是", fmt_vector(model.airspeed_wind), "C1", None, "向量")
    builder.ask("D 把体轴系的分量映射到风轴系，该风轴系空速坐标变到体轴系，应当使用", "D的转置乘风轴系空速坐标", "A2", ["D乘风轴系空速坐标", "D的转置乘风轴系空速坐标"])
    builder.ask("按 XTZ、YTZ、ZTZ 的顺序，空速单位向量在体轴系中的三个分量为", fmt_vector(model.airspeed_body), "A2", None, "向量")
    for component, axis_name in ((1, "YTZ"), (2, "ZTZ")):
        direction = _axis_direction(model.body, component)
        ordinal = "二" if component == 1 else "三"
        builder.ask(f"第{ordinal}分量是 {axis_name} 上的分量，{axis_name} 的正方向是绝对方向", DIR_ZH[direction], "C1", list(SIX))
        sign = _sign(model.airspeed_body[component])
        if sign == "负":
            toward = _opposite(direction)
            stem = "该分量为负表示空速指向绝对方向"
        elif sign == "正":
            toward = direction
            stem = "该分量为正表示空速指向绝对方向"
        else:
            toward = direction
            stem = "该分量为零，空速不偏向绝对方向"
        builder.ask(stem, DIR_ZH[toward], "A1", list(SIX))
    vertical = {"负": "下方", "正": "上方", "零": "没有上下偏置"}[_sign(model.airspeed_abs[1])]
    lateral = {"正": "右侧", "负": "左侧", "零": "没有左右偏置"}[_sign(model.airspeed_abs[2])]
    relation = f"{vertical}且{lateral}"
    builder.ask("因此空速相对机体指向", relation, "A1", ["下方且右侧", "上方且左侧", "下方且左侧", relation])
    builder.ask("按理论气流角的可变定义，迎角为正时，空速指向机体的绝对方向", DIR_ZH[spec.alpha_toward], "C1", list(SIX))
    builder.ask("侧滑角为正时，空速指向机体的绝对方向", DIR_ZH[spec.beta_toward], "C1", list(SIX))
    builder.ask("滚转角为负时，转向与绕机体绝对轴向正方向的右手定则", "相反" if spec.gamma_sense > 0 else "相同", "C1", ["相同", "相反"])
    builder.ask("单独变换滚转角时，旋转轴是机体的绝对", "轴向", "C1", ["轴向", "法向", "侧向"])
    builder.ask("该正方向与绕这根绝对轴的右手定则", "相同" if spec.gamma_sense > 0 else "相反", "C1", ["相同", "相反"])
    points_down = _sign(model.airspeed_abs[1]) == "负"
    alpha_positive = points_down if spec.alpha_toward == "down" else not points_down
    builder.ask(f"本例空速指向机体{vertical}，迎角的符号为", "正" if alpha_positive else "负", "A1", ["正", "负"])
    points_right = _sign(model.airspeed_abs[2]) == "正"
    beta_positive = points_right if spec.beta_toward == "right" else not points_right
    builder.ask(f"本例空速指向机体{lateral}，侧滑角的符号为", "正" if beta_positive else "负", "A1", ["正", "负"])
    right_index = _align(model.body, "right")[0]
    builder.ask("右翼沿机体绝对侧向正方向，它在风轴系中的坐标是 D 的第", ("一列", "二列", "三列")[right_index], "A1", ["一列", "二列", "三列"], tail="列")
    up_index, up_sign = _align(model.wind, "up")
    wing_up = model.D[up_index, right_index] * up_sign
    builder.ask("该列在风洞绝对法向正方向上的分量符号为", _sign(wing_up), "A1", ["正", "负"])
    builder.ask("因此右翼偏向风洞的绝对方向", "上方" if _sign(wing_up) == "正" else "下方", "A1", ["上方", "下方"])
    builder.ask("按理论滚转角的零参考和正方向，本例滚转角的符号为", _sign(model.gamma), "A1", ["正", "负"])
    builder.ask("按声明写出迎角、侧滑角和滚转角", model.airflow_text(), "A2", None, "气流角")
    builder.expect(65)

    builder.add("")
    builder.add("### 右翼尖前缘在体轴系")
    builder.add("")
    builder.add("这一层只求右翼尖前缘点在体轴系中的坐标，不把它映到风轴系。")
    builder.add("")
    body_axes = ["XTZ", "YTZ", "ZTZ", "-XTZ", "-YTZ", "-ZTZ"]
    for direction, stem in (
        ("forward", "体轴系的绝对轴向正方向（前方）对应"),
        ("up", "体轴系的绝对法向正方向（上方）对应"),
        ("right", "体轴系的绝对侧向正方向（右侧）对应"),
    ):
        builder.ask(stem, model.axis_name("body", direction), "C1", body_axes)
    builder.ask("本例位置参考点取", "右翼尖前缘点", "C1", ["右翼尖前缘点", "右翼尖后缘点", "左翼尖前缘点", "翼根前缘点"])
    builder.ask("计算翼尖前缘相对翼根前缘的前后偏移时，使用的气动参数是", "前缘后掠角Λ", "C1", ["前缘后掠角Λ", "上反角Γ", "翼展b", "翼根前缘坐标x_r"])
    builder.ask("该参数为正时，翼尖前缘相对翼根前缘偏向绝对方向", "后方", "C1", ["前方", "后方"])
    builder.ask("前后偏移写入机体绝对轴向正方向对应的坐标轴", model.axis_name("body", "forward"), "C1", body_axes)
    builder.ask("该坐标轴的正方向是绝对方向", _axis_positive_zh(model, "forward"), "C1", list(SIX))
    forward_sign = _align(model.body, "forward")[1]
    builder.ask("所以这项偏移使该分量", "减小" if forward_sign > 0 else "增大", "A1", ["增大", "减小"])
    sweep = "-(b/2)*tan(Λ)" if forward_sign > 0 else "(b/2)*tan(Λ)"
    builder.ask("前后偏移的表达式是", sweep, "A2", ["(b/2)*tan(Λ)", "-(b/2)*tan(Λ)", "(b/2)*tan(Γ)", "-b*tan(Λ)"])
    builder.ask("翼根前缘自身的纵向坐标使用", "x_r", "C1", ["x_r", "翼展b", "前缘后掠角Λ", "上反角Γ"])
    builder.ask("右翼尖的左右位置写入机体绝对侧向正方向对应的坐标轴", model.axis_name("body", "right"), "C1", body_axes)
    builder.ask("该坐标轴的正方向是绝对方向", _axis_positive_zh(model, "right"), "C1", list(SIX))
    builder.ask("左右坐标使用的气动参数是", "翼展b", "C1", ["翼展b", "翼展b与上反角Γ", "前缘后掠角Λ", "翼根前缘坐标x_r"])
    lateral_text = format_body_component(model.tip_body()[_align(model.body, "right")[0]])
    builder.ask("左右坐标的表达式是", lateral_text, "C1" if lateral_text == "b/2" else "A2", ["(b/2)*cos(Γ)", "(b/2)*sin(Γ)", "b/2", "(b/2)*tan(Γ)"])
    builder.ask("右翼尖的高低位置写入机体绝对法向正方向对应的坐标轴", model.axis_name("body", "up"), "C1", body_axes)
    builder.ask("该坐标轴的正方向是绝对方向", _axis_positive_zh(model, "up"), "C1", list(SIX))
    builder.ask("上反角为正时，翼尖相对翼根偏向绝对方向", "上方", "C1", ["上方", "下方"])
    height_text = format_body_component(model.tip_body()[_align(model.body, "up")[0]])
    builder.ask("高低偏移的表达式是", height_text, "A2", ["(b/2)*sin(Γ)", "-(b/2)*sin(Γ)", "(b/2)*tan(Γ)", "-(b/2)*cos(Γ)"])
    builder.ask("按 XTZ、YTZ、ZTZ 的顺序，右翼尖前缘点在体轴系中的坐标为", model.tip_text(), "A2", None, "向量")
    builder.expect(85)

    builder.add("")
    builder.add("### 体轴原点的平移")
    builder.add("")
    builder.add("这一层先算机体顶点到球腕原点的距离，再算体轴系原点在风轴系中的平移向量。不使用翼尖坐标。")
    builder.add("")
    builder.ask("体轴系的原点建立在", "机体顶点", "C1", ["机体顶点", "机体质心", "球腕原点", "机体底部"])
    builder.ask("球腕系原点与", "风轴系原点", "C1", ["风轴系原点", "机体顶点", "机体质心", "机体底部"], tail="重合")
    builder.ask("机体全长 L 量的是", "机体顶点到机体底部", "C1", ["机体顶点到机体底部", "机体底部到球腕原点", "机体顶点到球腕原点"])
    builder.ask("L_测 量的是", "机体底部到球腕原点", "C1", ["机体底部到球腕原点", "机体顶点到机体底部", "支杆插入机体内部的长度"])
    builder.ask("支杆顶部从机体底部插入机体内部的那一段", "已经含在机体全长 L 内，不再另加", "C1", ["已经含在机体全长 L 内，不再另加", "要加在 L 和 L_测 之外", "要从 L 中减去"])
    builder.ask("沿机体前方，球腕原点、机体底部、机体顶点的顺序是", "O、底部、顶点", "C1", ["O、底部、顶点", "顶点、底部、O", "底部、O、顶点"])
    builder.ask("因此机体顶点到球腕原点的距离为", "L + L_测", "A2", None, "距离")
    builder.ask("从球腕原点指向机体顶点的方向是机体绝对方向", "前方", "C1", ["前方", "后方", "上方", "下方"])
    builder.ask("该偏移写入的体轴坐标轴是", model.axis_name("body", "forward"), "C1", body_axes)
    builder.ask("另外两个分量为", "0", "C2", ["0", "L", "L_测", "L + L_测"])
    shift = _shift(model)
    builder.ask("按 XTZ、YTZ、ZTZ 的顺序，从球腕原点到体轴系原点的体轴分量为", shift, "A2", None, "向量")
    builder.ask("球腕具有姿态角后，该体轴向量映到风轴系应当使用", "D乘该向量", "A2", ["D乘该向量", "D的转置乘该向量", "R乘该向量", "不乘姿态矩阵"])
    builder.ask("不展开 D，方括号表示列向量，体轴系原点在风轴系中的平移向量为", f"D*{shift}", "A2", None, "表达式")
    builder.expect(98)

    builder.add("")
    builder.add("### 右翼尖前缘在风轴系")
    builder.add("")
    tip = model.tip_text()
    builder.add(f"这一层只求风轴系坐标。右翼尖前缘点在体轴系中、按 XTZ、YTZ、ZTZ 顺序的坐标是 {tip}。体轴系原点不在风轴系原点。体轴系原点在风轴系中的平移向量是 D*{shift}。方括号表示列向量。")
    builder.add("")
    builder.ask("该体轴坐标映到风轴系时应当使用", "D乘体轴坐标", "A2", ["D乘体轴坐标", "D的转置乘体轴坐标", "R乘体轴坐标"])
    builder.ask("风轴系位置还要把平移向量与这个旋转结果", "相加", "A2", ["相加", "只保留旋转结果", "只保留平移向量"])
    builder.ask("不展开 D，按 XFD、YFD、ZFD 的顺序，右翼尖前缘点在风轴系中的位置为", f"D*{shift} + D*{tip}", "A2", None, "表达式")
    builder.expect(101)
    _collision(builder, model, checked, tip, shift, reached)


def _collision(builder: Builder, model: ArmModel, checked: bool, tip: str, shift: str, reached: float | None) -> None:
    builder.add("")
    builder.add("### 碰壁检测")
    builder.add("")
    builder.add(
        "这一层求机体顶点在声明关节范围内的三个风洞坐标可达区间，并判定机体顶点、右翼顶点和左翼顶点是否存在碰壁姿态。"
        "接触壁面即算碰壁。不把翼尖的上下、左右展开成闭区间。球腕原点在截面正中。"
        f"右翼顶点取右翼尖前缘，它在体轴系中的坐标是 {tip}。机体顶点相对球腕原点的体轴分量是 {shift}。"
        "这一层看不到前面子过程的填空，也不用本例这一组关节角代替整个范围。判据只覆盖这三个端点。"
    )
    builder.add("")
    wind_axes = ["XFD", "YFD", "ZFD"]
    builder.ask("通道壁面限制的绝对方向是", "上方与下方、右侧与左侧", "C1", ["上方与下方、右侧与左侧", "前方与后方、上方与下方", "前方与后方、右侧与左侧"])
    builder.ask("沿风洞绝对轴向", "不设壁面", "C1", ["不设壁面", "在前方设壁面", "在后方设壁面"])
    builder.ask("上下壁限制的风洞坐标轴是", model.positive_name("wind", "up"), "C1", wind_axes)
    builder.ask("左右壁限制的风洞坐标轴是", model.positive_name("wind", "right"), "C1", wind_axes)
    builder.ask("截面边长为 L_风，球腕原点在正中，中心到一面壁的距离是", "L_风/2", "A2", None, "表达式")
    builder.ask("两侧壁位于这个距离的", "正负两侧", "A1", ["正负两侧", "只在正侧", "只在负侧"])
    builder.ask("右翼顶点相对球腕原点的体轴向量，由机体顶点偏移与右翼尖前缘坐标", "相加", "A2", ["相加", "相减", "只保留翼尖坐标"], tail="得到")
    builder.ask("按 XTZ、YTZ、ZTZ 的顺序，该向量为", format_body_vector(wing_offset(model)), "A2", None, "向量")
    builder.ask("左翼顶点与右翼顶点相比，", "前后与高低相同，左右取反", "A1", ["前后与高低相同，左右取反", "三个分量都取反", "前后取反，左右相同"])
    builder.ask("按 XTZ、YTZ、ZTZ 的顺序，左翼向量为", format_body_vector(wing_offset(model, -1)), "A2", None, "向量")
    builder.ask("这三个体轴向量映到风轴系应当使用", "D乘该向量", "A2", ["D乘该向量", "D的转置乘该向量", "R乘该向量", "不乘姿态矩阵"])
    if checked:
        _collision_checked(builder, model)
        builder.expect(140)
    else:
        _collision_general(builder, model, reached)


def _collision_checked(builder: Builder, model: ArmModel) -> None:
    nose = sp.simplify(model.nose_wind())
    middle = sp.simplify(model.joint_matrices[1] * model.joint_matrices[2] * model.S * model.vertex_body(ELL))
    ranges = nose_ranges(model)
    forward_index = _align(model.wind, "forward")[0]
    up_index = _align(model.wind, "up")[0]
    right_index = _align(model.wind, "right")[0]
    install = principal_rotation(model.S)
    degree = install[1]
    signed = f"+{degree}°" if degree > 0 else f"{degree}°"
    builder.add(f"q3 与安装转角绕同一根坐标轴，安装转角是 {signed}。")
    builder.ask("扫过范围时的合成角 θ 取", theta_text(model), "A2", ["q3+90°", "q3-90°", "q3", "q1+90°"])
    low, high = model.limit_of_q(2)
    epsilon = joint_axis(model, 2)[1]
    ends = sorted((epsilon * low + degree, epsilon * high + degree))
    builder.ask(f"q3 从 {low}° 到 {high}°，θ 的范围是", f"{ends[0]}°到{ends[1]}°", "A2", ["0°到180°", "-90°到90°", "-180°到0°", "0°到90°"])
    q1_axis = model.positive_name("wind", ROLE_POSITIVE[model.role_of_q(0)])
    builder.ask("记 v=R(q2)*R(θ)*r，v 是乘积中的中间向量，分量按 X、Y、Z 排列，尚未左乘 R(q1)，q1 的矩阵写在最左侧，绕风洞坐标轴", q1_axis, "C2", ["XFD", "YFD", "ZFD"])
    independent = not sp.simplify(nose[up_index]).has(Q[0])
    builder.ask("左乘 R(q1) 之后，端点的风洞上下坐标与 q1", "无关" if independent else "有关", "A1", ["无关", "有关"])
    builder.ask("因此该上下坐标等于 v 的第", ("一", "二", "三")[up_index], "A2", ["一", "二", "三"], tail="个分量")
    builder.ask("机体顶点的体轴向量不被 R(θ) 改变，再左乘 R(q2) 后，中间向量为", format_wind_vector(middle), "A2", None, "向量")
    up_text = format_scalar(middle[up_index])
    builder.ask("左乘 R(q1) 不改变这个第二分量，顶点的风洞上下坐标等于", up_text, "A2", [up_text, "(L + L_测)*cos(q1)", "-(L + L_测)*sin(q1)"])
    lateral_index = model.spec.order.index("lateral")
    lateral_low, lateral_high = model.limit_of_q(lateral_index)
    sin_low, sin_high = _trig_bounds("sin", lateral_low, lateral_high)
    builder.ask(
        f"q{lateral_index + 1} 是侧向轴，范围是 {lateral_low}° 到 {lateral_high}°，sin(q{lateral_index + 1}) 能取到的值覆盖",
        f"{_trig_text(sin_low)}到{_trig_text(sin_high)}",
        "A2", ["-1到1", "-1/2到1/2", "-sqrt(3)/2到sqrt(3)/2"],
    )
    normal_index = model.spec.order.index("normal")
    normal_low, normal_high = model.limit_of_q(normal_index)
    cos_low, _cos_high = _trig_bounds("cos", normal_low, normal_high)
    builder.ask(
        f"q{normal_index + 1} 是法向轴，范围是 {normal_low}° 到 {normal_high}°，cos(q{normal_index + 1}) 的最小值是",
        _trig_text(cos_low), "A2", ["1/2", "0", "sqrt(3)/2", "1"],
    )
    q2_low, q2_high = model.limit_of_q(1)
    q2_cos, _q2_top = _trig_bounds("cos", q2_low, q2_high)
    builder.ask(
        f"q2 在 {q2_low}° 到 {q2_high}° 内，cos(q2)",
        "不为负" if float(q2_cos) >= -1e-9 else "可以变号",
        "A1", ["不为负", "不为正", "恒为 1", "可以变号"],
    )
    normal_sin_low, normal_sin_high = _trig_bounds("sin", normal_low, normal_high)
    builder.ask(
        f"sin(q{normal_index + 1}) 的最大绝对值是",
        _trig_text(max(abs(normal_sin_low), abs(normal_sin_high))),
        "A2", ["sqrt(3)/2", "1/2", "1", "0"],
    )
    axial_text = format_scalar(nose[forward_index])
    lateral_coord = format_scalar(nose[right_index])
    builder.ask("顶点的风洞轴向坐标是", axial_text, "A2", [axial_text, "(L + L_测)*cos(q2)", "(L + L_测)*sin(q2)"])
    builder.ask("左右坐标是", lateral_coord, "A2", [lateral_coord, "(L + L_测)*sin(q2)", "-(L + L_测)*sin(q2)"])
    labels = ("轴向可达范围是", "上下可达范围是", "左右可达范围是")
    indexes = (forward_index, up_index, right_index)
    for stem, index in zip(labels, indexes):
        builder.ask(stem, format_interval(*ranges[index]), "A2", None, "范围")
    up_max = max_abs_ratio(*ranges[up_index])
    right_max = max_abs_ratio(*ranges[right_index])
    if up_max > right_max + 1e-6:
        larger, wall = "更大", "上下壁"
    elif right_max > up_max + 1e-6:
        larger, wall = "更小", "左右壁"
    else:
        larger, wall = "相等", "上下壁"
    builder.ask("顶点上下方向的最大绝对值比左右方向的最大绝对值", larger, "A1", ["更大", "更小", "相等"])
    builder.ask("当逐步增大顶点到球腕中心的距离时，首先达到的碰壁阈值对应", wall, "A1", ["上下壁", "左右壁", "轴向两端"])
    full = max(up_max, right_max) >= 1 - 1e-6
    builder.ask("接触壁面即算碰壁，顶点可能碰壁的条件是", "L + L_测 >= L_风/2" if full else _scaled_condition(max(up_max, right_max)), "A2", None, "表达式")
    builder.ask(
        "在声明的关节范围内，端点风洞上下坐标绝对值的最大可达值等于",
        "该端点到球腕原点的距离" if up_max >= 1 - 1e-6 else "该距离乘一个小于 1 的系数",
        "A2", ["该端点到球腕原点的距离", "该距离的一半", "该距离乘 sqrt(3)/2"],
    )
    builder.add("可以选取 θ，使 R(θ) 后的第三分量为 0，再用 q2 在 -90° 到 90° 内把向量转到 YFD 的正方向或负方向。")
    builder.add("q1 绕 YFD，不改变上下坐标。")
    builder.add("因此这个最大可达值等于该端点到球腕原点的距离。")
    builder.ask("端点到球腕原点的距离达到或超过中心到壁的距离，是存在接触壁面姿态的", "充要条件" if full else "只是必要不充分", "A2", ["充要条件", "只是充分条件", "只是必要不充分"])
    builder.ask("轴向可达区间", "不参与", "C1", ["不参与", "同样参与"], tail="碰壁判断")
    builder.ask("三个端点中只要有一个端点到球腕原点的距离达到或超过半宽，判定为", "可能碰壁", "C2", ["可能碰壁", "不可能碰壁"])
    builder.ask("某端点到球腕原点的距离小于半宽时，该端点", "不可能碰壁", "A2", ["不可能碰壁", "可能碰壁"])
    perp = perpendicular_square(model)
    builder.ask("右翼顶点体轴向量的后两个分量平方和是", perp, "A2", ["(b/2)^2", "(b/2)^2*(1 + (tan(Γ))^2)", "b^2", "(b/2)^2*(tan(Γ))^2"])
    radius = _wing_radius(model)
    builder.ask("右翼顶点到球腕原点的距离是", radius, "A2", None, "表达式")
    builder.ask("左翼顶点的这个距离与右翼", "相同", "A1", ["相同", "相反", "相差 L"])
    final = f"max(L + L_测, {radius}) >= L_风/2" if full else f"{radius} >= L_风/2 只给出长度上限"
    builder.ask("该判据只覆盖三个检测端点，在声明的关节范围内，至少有一个端点接触或越过壁面的条件是", final, "A2", None, "表达式")


def _scaled_condition(ratio: float) -> str:
    exact = snap_ratio(ratio)
    if exact is None or exact == 1:
        return "L + L_测 >= L_风/2"
    return f"{format_scalar(exact * ELL)} >= L_风/2"


def _coord_range_stem(model: ArmModel, direction: str) -> str:
    name = model.axis_name("wind", direction)
    if name.startswith("-"):
        axis = name[1:]
        sense = DIR_ZH[_opposite(direction)]
    else:
        axis = name
        sense = DIR_ZH[direction]
    return f"{axis} 的正方向是{sense}，该坐标的可达范围是"


def _collision_general(builder: Builder, model: ArmModel, reached: float | None) -> None:
    """Recompute the scan. Do not reuse θ=q3+90° or the default nose box."""
    theta = theta_text(model)
    install = principal_rotation(model.S)
    if theta is None:
        builder.add("q3 与安装转角不绕同一根坐标轴，扫描时不能把它们合成一个角。")
        builder.ask("扫过范围时的合成角 θ 取", "不能合成", "A2", ["不能合成", "仍写成 q3", "仍写成 q1 与安装角之和"])
    else:
        degree = install[1]
        signed = f"+{degree}°" if degree > 0 else f"{degree}°"
        builder.add(f"q3 与安装转角绕同一根坐标轴，安装转角是 {signed}。")
        builder.ask("扫过范围时的合成角 θ 取", theta, "A2", ["q3+90°", "q3-90°", "q3", "q1+90°", theta])
    nose = sp.simplify(model.nose_wind())
    ranges = nose_ranges(model)
    builder.ask("按当前旋转顺序和安装矩阵重新相乘，机体顶点在风轴系中的坐标为", format_wind_vector(nose), "A2", None, "向量")
    forward_index = _align(model.wind, "forward")[0]
    up_index = _align(model.wind, "up")[0]
    right_index = _align(model.wind, "right")[0]
    q1_axis = model.positive_name("wind", ROLE_POSITIVE[model.role_of_q(0)])
    independent = not sp.simplify(nose[up_index]).has(Q[0])
    builder.add(f"q1 绕 {q1_axis}。" + ("上下坐标不含 q1。" if independent else "上下坐标含有 q1，不能沿用绕风洞上方时上下与 q1 无关的结论。"))
    builder.ask("左乘最左侧关节之后，端点的风洞上下坐标与 q1", "无关" if independent else "有关", "A1", ["无关", "有关"])
    names = {
        "forward": _coord_range_stem(model, "forward"),
        "up": _coord_range_stem(model, "up"),
        "right": _coord_range_stem(model, "right"),
    }
    for direction, index in (
        ("forward", forward_index),
        ("up", up_index),
        ("right", right_index),
    ):
        builder.ask(names[direction], format_interval(*ranges[index]), "A2", None, "范围")
    up_max = max_abs_ratio(*ranges[up_index])
    right_max = max_abs_ratio(*ranges[right_index])
    if abs(up_max - right_max) <= 1e-6:
        larger, wall = "相等", "上下壁与左右壁同时"
    elif up_max > right_max:
        larger, wall = "更大", "上下壁"
    else:
        larger, wall = "更小", "左右壁"
    builder.ask("顶点上下方向的最大绝对值比左右方向的最大绝对值", larger, "A1", ["更大", "更小", "相等"])
    builder.ask("当逐步增大顶点到球腕中心的距离时，首先达到的碰壁阈值对应", wall, "A1", ["上下壁", "左右壁", "轴向两端"])
    vertex_full = max(up_max, right_max) >= 1 - 1e-6
    builder.ask("接触壁面即算碰壁，顶点可能碰壁的条件是", "L + L_测 >= L_风/2" if vertex_full else _scaled_condition(max(up_max, right_max)), "A2", None, "表达式")
    covered = reached is not None and reached >= 0.99
    if covered:
        builder.add("按该声明重新扫描关节范围后，采样到的机体系方向仍能贴到某一壁面法向。")
        verdict = "充要条件"
    else:
        builder.add("按该声明重新扫描后，有的机体系方向到不了壁面法向。半径达到半宽只是长度上限。")
        verdict = "只是必要不充分"
    builder.ask("端点到球腕原点的距离达到或超过中心到壁的距离，对当前声明是存在接触壁面姿态的", verdict, "A2", ["充要条件", "只是充分条件", "只是必要不充分"])
    radius = _wing_radius(model)
    builder.ask("右翼顶点到球腕原点的距离是", radius, "A2", None, "表达式")
    if covered:
        final = f"max(L + L_测, {radius}) >= L_风/2"
    else:
        final = f"{radius} >= L_风/2 只给出长度上限，不能当成该姿态可达"
    builder.add("该判据只覆盖三个检测端点。")
    builder.ask("按重新扫描得到的可达性，至少有一个端点接触或越过壁面的条件是", final, "A2", None, "表达式")


def _pose_at(model: ArmModel, degrees: list[int]) -> sp.Matrix:
    values = {Q[index]: sp.rad(degrees[index]) for index in range(3)}
    return sp.simplify(model.wrist_pose().subs(values))


def _singular_pair(model: ArmModel):
    _low, high = LIMITS[model.spec.order[1]]
    if high < 90:
        return None
    first_choices = ([0, 90, 0], [0, -90, 0])
    seconds = []
    for delta in (30, -30, 20, 45):
        seconds.append([delta, 90, -delta])
    for delta in (30, -30, 20, 45):
        seconds.extend(([delta, 90, delta], [delta, -90, -delta], [delta, -90, delta]))
    for first in first_choices:
        if not _inside(model, first):
            continue
        for second in seconds:
            if second[1] != first[1] or not _inside(model, second):
                continue
            if _pose_at(model, first) == _pose_at(model, second):
                return first, second
    return None


def _inside(model: ArmModel, degrees: list[int]) -> bool:
    for index, value in enumerate(degrees):
        low, high = model.limit_of_q(index)
        if value < low or value > high:
            return False
    return True


def _parallel_up(model: ArmModel, degrees: list[int]) -> bool:
    mapping = sp.simplify(_pose_at(model, degrees) * model.S)
    forward = sp.simplify(mapping * (model.body.T * ABS["forward"]))
    upward = sp.simplify(model.wind.T * ABS["up"])
    return sp.simplify(_cross(forward, upward)) == sp.Matrix([0, 0, 0])


def _triple(degrees: list[int]) -> str:
    return "(" + ", ".join(f"{item}°" for item in degrees) + ")"


def _branch_unique(model: ArmModel) -> bool:
    if not _in_operating_range(model):
        return False
    try:
        recovered = recover_joints(model)
    except (ValueError, TypeError):
        return False
    return all(abs(got - want) < 1e-6 for got, want in zip(recovered, model.q_degrees))


def _inverse(builder: Builder, model: ArmModel, checked: bool) -> None:
    builder.add("### 由三个气流角求关节角")
    builder.add("")
    builder.ask("若只给出迎角和侧滑角，不给出滚转角，姿态还少一个自由度，三关节能否保证唯一？答案为", "不能", "A2", ["能", "不能"])
    pair = _singular_pair(model)
    middle = ROLE_ZH[model.spec.order[1]]
    if pair is None:
        if LIMITS[model.spec.order[1]][1] < 90:
            builder.add(f"中间的{middle}轴到不了 ±90°，下面不再用边界姿态作为多解反例。")
        else:
            builder.add(f"中间的{middle}轴能够到达 ±90°。抽查的边界组合里没有两组姿态相同，下面不把未验证的组合写成多解。")
        builder.ask("在声明范围内，边界奇异姿态", "不纳入唯一逆解", "A2", ["纳入唯一逆解", "不纳入唯一逆解"])
        builder.ask("机体前方与风洞上方向平行时，按本样本的使用条件", "不由三个气流角求关节角", "C1", ["仍由三个气流角求关节角", "不由三个气流角求关节角"])
    else:
        left, right = pair
        same = _pose_at(model, left) == _pose_at(model, right)
        builder.add(f"下面这组反例说明的是{middle}轴取到边界时，同一姿态可以对应多组关节角，不是两气流角少一个自由度。")
        builder.add(f"两组关节角都在声明范围内：q={_triple(left)} 和 q={_triple(right)}。")
        builder.ask("用声明的乘积检查，这两组的姿态矩阵", "相同" if same else "不同", "A2", ["相同", "不同"])
        parallel = _parallel_up(model, left) and _parallel_up(model, right)
        if parallel:
            builder.ask("这两组的机体前方轴与风洞上方向平行，按本样本的使用条件，这时", "不由三个气流角求关节角", "C1", ["仍由三个气流角求关节角", "不由三个气流角求关节角"])
        else:
            builder.ask("这两组姿态下，按本样本的使用条件", "不由三个气流角求关节角" if not parallel else "仍由三个气流角求关节角", "C1", ["仍由三个气流角求关节角", "不由三个气流角求关节角"])
    branch = f"{middle}轴处在非奇异分支内"
    if checked or _branch_unique(model):
        builder.ask(f"本例这组三个角满足 -90°<α<90°、-90°<β<90°，机体前方轴不与风洞上方向平行，且{branch}，这组角在声明的关节范围内对应的关节角能否保证唯一？答案为", "能", "A2", ["能", "不能"])
    else:
        builder.ask("按本例角度和关节范围，这组角能否套用非奇异分支的唯一逆解？答案为", "不能", "A2", ["能", "不能"])
    builder.ask("这三个气流角确定的是机体相对风轴系的姿态，对应的矩阵是", "D", "C1", ["D", "R", "S"])
    if _classic_airspeed_rows(model):
        builder.ask("记 a=cos(α)*cos(β)，bv=-sin(α)*cos(β)，c=sin(β)，D 的第一行是体轴系中的空速分量，等于", "[a, bv, c]", "C2", ["[a, bv, c]", "[a, c, bv]", "[bv, a, c]"])
        builder.ask("代入本例角度，这一行的数值是", fmt_vector(model.airspeed_body), "A2", None, "向量")
        row = "[-d/N, a*cos(γ)/N, -a*sin(γ)/N]"
        builder.ask("记 d=bv*cos(γ)-c*sin(γ)，N=sqrt(a^2+d^2)，在 a>0 的分支上，按理论滚转角的零参考和绕机体前方轴的右手定则正方向，第二行是", row, "A2", ["[-d/N, a*cos(γ)/N, -a*sin(γ)/N]", "[d/N, a*cos(γ)/N, a*sin(γ)/N]", "[-d/N, a*sin(γ)/N, a*cos(γ)/N]", row])
        builder.ask("第三行是第一行与第二行的叉积，由此算出的 D 为", fmt_matrix(model.D), "A2", None, "矩阵")
    else:
        absolute = _abs_component_vector(model)
        if sp.simplify(absolute - model.airspeed_abs) != sp.Matrix([0, 0, 0]) and not _matrix_zero(absolute - model.airspeed_abs):
            raise RuntimeError("迎角和侧滑角的绝对分量与空速不一致")
        builder.ask("按本例迎角和侧滑角的正方向，空速在绝对前方、上方、右侧上的分量依次是", _abs_component_text(model), "A2", None, "向量")
        builder.ask("代入本例角度，该绝对分量为", fmt_vector(model.airspeed_abs), "A2", None, "向量")
        builder.ask("按体轴系声明写入 XTZ、YTZ、ZTZ，空速分量为", fmt_vector(model.airspeed_body), "A2", None, "向量")
        builder.add("该体轴分量等于 D 的转置乘风轴系空速。")
        builder.ask("连同滚转角零参考确定的姿态矩阵 D 为", fmt_matrix(model.D), "A2", None, "矩阵")
    builder.ask("零位时机体轴向正方向指向", "来流方向", "C1", ["来流方向", "风洞上方", "风洞右侧", "风洞下方"])
    builder.ask("机体法向正方向与", DIR_ZH[model.spec.install_up], "C1", ["上方", "下方", "右侧", "左侧"], tail="重合")
    builder.add("机体前方与风洞前方重合。")
    builder.add(f"机体上方与风洞{DIR_ZH[model.spec.install_up]}重合。")
    right_abs = _cross_name("forward", model.spec.install_up)
    builder.ask("按机体前方 × 机体上方 = 机体右侧，机体右侧指向风洞的绝对方向", DIR_ZH[right_abs], "A1", ["上方", "下方", "左侧", "右侧"])
    builder.add("S 的列按 XTZ、YTZ、ZTZ 排列，行按 XMD、YMD、ZMD 排列。")
    builder.add("零位时球腕物理方向与风洞方向重合。")
    builder.ask("安装矩阵 S 为", fmt_matrix(model.S), "A2", None, "矩阵")
    builder.ask("为消去固定安装，球腕相对风轴系的姿态应为", "D*S^T", "A2", ["D*S^T", "S^T*D", "D*S"])
    low, high = LIMITS[model.spec.order[1]]
    builder.add("记上一步得到的球腕姿态为 H，其元素为 hij，行号在前，列号在后。")
    if (low, high) == (-90, 90) and model.spec.order == ("normal", "lateral", "axial"):
        branch_text = "-90°<q2<90°"
    else:
        branch_text = f"{low}°<q2<{high}°"
    builder.ask(f"在非奇异分支 {branch_text} 上，q1、q2、q3 的解析提取公式为", model.extraction(), "A2", None, "公式")
    builder.ask("由此得到的三个关节角为", "[" + ", ".join(str(item) for item in model.q_degrees) + "]", "A2", None, "关节角")
    if checked:
        builder.expect(16)


def _notes(model: ArmModel, checked: bool, reached: float | None) -> list[str]:
    if checked:
        return _checked_notes()
    covered = reached is not None and reached >= 0.99
    theta = theta_text(model) or "不能合成"
    nose = format_wind_vector(sp.simplify(model.nose_wind()))
    verdict = "半径达到半宽仍是碰到壁面的充要条件。" if covered else "半径达到半宽只给出长度上限，不能当成该姿态可达。"
    return [
        "# 层间对应",
        "",
        "这一节不送入模型。本份由当前声明重新生成，旋转顺序或安装与已核对样本不同，不套用已核对样本的层间题号。",
        "",
        "# 生成注记",
        "",
        "这一节不送入模型。",
        "",
        f"动轴内旋顺序是{_order_text(model.spec)}。零位法向与{DIR_ZH[model.spec.install_up]}重合。逆解提取公式由这个声明重新推导：{model.extraction()}。",
        "",
        f"合成扫描角是 {theta}。机体顶点的风轴坐标是 {nose}。{verdict}",
        "",
    ]


def _checked_notes() -> list[str]:
    return [
        "# 层间对应",
        "",
        "这一节不送入模型。外形换算、翼尖在风轴系中的位置、碰壁检测没有重构层或理论层对应题。统计时分开报告两项：三层共同核心任务的配对准确率，以及交叉层额外工程任务的准确率。不要把三层全文总准确率直接放在一起比较。正运动学样本测量的是带分步提示的条件能力。逆运动学样本单独送入，用来观察去掉关节角和正向过程之后的逆解。",
        "",
        "| 交叉层 | 重构层保持的计算 | 理论层中的对应操作 |",
        "|---|---|---|",
        "| 正样本 1–9 风轴系、球腕系、体轴系的方向对应哪根坐标轴 | 基座、球腕、工具各自声明坐标轴 | 不设绝对方向 |",
        "| 正样本 10–24 绝对方向轴旋转顺序、正方向、去向和矩阵。本例顺序是法向、侧向、轴向 | 直接使用已声明的乘积与安装矩阵 | 直接给出复合前的矩阵 |",
        "| 正样本 25–35 使用安装矩阵之前写出 S。零位轴向与风轴轴向重合，法向与右侧重合 | 由安装方向写出工具相对末端的矩阵 | 直接给出安装矩阵 |",
        "| 正样本 36–53 自行判断 D 与 R、S 的关系，先写各分矩阵展开式，再算空速。46 读取空速与风洞轴向的关系 | 用同一组关节角计算工具姿态和参考方向 | 模块A |",
        "| 正样本 54–65 三个气流角的正方向和数值。滚转角以风洞上方在垂直于机体前方的平面上的投影为零参考，正方向沿绕机体前方轴的右手定则 | 方向角及其正方向 | 模块B |",
        "| 正样本 66–85 右翼尖前缘在体轴系中的坐标。翼展是侧向间距 | 无对应题 | 无对应题 |",
        "| 正样本 86–98 顶点到球腕原点的距离，以及体轴原点在风轴系中的平移。沿前方的顺序是 O、底部、顶点 | 工具原点相对基座的平移 | 齐次变换的平移列 |",
        "| 正样本 99–101 右翼尖前缘在风轴系中的坐标，含平移 | 无对应题 | 无对应题 |",
        "| 正样本 102–140 顶点可达区间，以及三个端点按半径作出的碰壁判据。接触壁面即算碰壁。该判据只对当前旋转顺序和当前零位安装成立 | 无对应题 | 无对应题 |",
        "| 逆样本 1–4 两气流角缺少自由度，与侧向轴边界多解分开设问。逆样本 5–9 由三个角构造 D | 由方向角恢复姿态 | 模块B与模块C的可解条件 |",
        "| 逆样本 10–13 使用 S 之前按零位安装写出矩阵 | 由安装方向写出工具相对末端的矩阵 | 直接给出安装矩阵 |",
        "| 逆样本 14 先单独问消去安装后的姿态。逆样本 15–16 把该姿态记为 H 后再提取关节角 | 由完整姿态反求关节角 | 模块C |",
        "",
        "# 生成注记",
        "",
        "这一节不送入模型。",
        "",
        "注入以句号为界，每句一个待填槽。不另设 step_id。当前句之前的空填入选项原文。同一句的后文对当前空可见，所以答案不写在该句后文。",
        "",
        "逆样本必须使用独立会话。不要把整份审阅稿作为附件同时送入，不要携带答案、评分说明、层间对应和本节。同一底层实例的正样本和逆样本按实例分组划分，不要拆到训练集和测试集两侧。",
        "",
        "逆解提取公式只在当前默认下成立：动轴内旋顺序是法向、侧向、轴向，非奇异分支是 -90°<q2<90°。完整机械限位仍允许 q2=±90°，边界奇异姿态不纳入唯一逆解任务。旋转顺序改变后必须重推提取公式，不能只替换轴名和关节编号。",
        "",
        "碰壁可达性和最终判据只在当前默认下成立：动轴内旋顺序是法向、侧向、轴向，零位安装是机体轴向与风轴轴向重合、机体法向与右侧重合。因此 q3 与安装角同轴，θ=q3+90°，最左侧矩阵绕 YFD。改变安装后，不一定能把 q3 与安装角直接合并。改变顺序后，最左侧矩阵未必绕 YFD，端点也未必能在给定限位内达到某壁面法向。此时半径达到半宽通常只给出理论长度上限，不能直接当成该姿态可达。另生成样本时要重新推导可达性。",
        "",
    ]


def _slots(text: str) -> dict[str, tuple[str, str]]:
    found = {}
    for match in SLOT_RE.finditer(text):
        found[match.group(1)] = (match.group(5), match.group(2))
    return found


def _resolved(answer: str, options: str) -> str:
    if options == "无":
        return answer
    for piece in options.split("；"):
        letter, _, body = piece.partition("：")
        if letter == answer:
            return body
    return answer


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check(generated: str, gold: str) -> list[str]:
    errors = []
    for name, pattern in (("正运动学", r"# 正运动学样本(.*)# 逆运动学样本"), ("逆运动学", r"# 逆运动学样本(.*)# 层间对应")):
        new = re.search(pattern, generated, re.S)
        old = re.search(pattern, gold, re.S)
        if not new or not old:
            errors.append(f"缺少{name}样本")
            continue
        new_slots = _slots(new.group(1))
        old_slots = _slots(old.group(1))
        if set(new_slots) != set(old_slots):
            missing = sorted(set(old_slots) - set(new_slots), key=int)
            extra = sorted(set(new_slots) - set(old_slots), key=int)
            errors.append(f"{name} 题号不一致，缺 {missing[:8]}，多 {extra[:8]}")
        for key in sorted(set(old_slots) & set(new_slots), key=int):
            old_text = _resolved(*old_slots[key])
            new_text = _resolved(*new_slots[key])
            if _norm(old_text) != _norm(new_text):
                errors.append(f"{name} ID {key}: {new_text} != {old_text}")
    return errors


def validate(text: str) -> list[str]:
    errors = []
    for line in text.splitlines():
        if re.match(r"\d+\.", line) and re.search(r"\{ID:\d+,", line):
            if len(re.findall(r"\{ID:\d+,", line)) != 1 or line.count("。") != 1:
                errors.append(line[:80])
    return errors


def _variant_ok(text: str, model: ArmModel) -> list[str]:
    errors = []
    collision = text.split("### 碰壁检测", 1)[-1].split("# 逆运动学样本", 1)[0]
    theta = theta_text(model)
    for line in collision.splitlines():
        match = SLOT_RE.search(line)
        if match and "合成角" in line:
            answer = _resolved(match.group(5), match.group(2))
            if answer == "q3+90°" and theta != "q3+90°":
                errors.append("换序后仍把合成角答成 q3+90°")
    if theta != "q3+90°" and "安装转角是 +90°" in collision and model.spec.order != ("normal", "lateral", "axial"):
        errors.append("换序后仍把安装转角写成与 q3 直接相加的 +90°")
    if "q1 绕 YFD，不改变上下坐标" in collision:
        errors.append("换序后的碰壁段仍写了 q1 绕 YFD")
    if model.extraction() not in text:
        errors.append("换序样本没有写入重新推导的提取公式")
    recovered = recover_joints(model)
    for got, want in zip(recovered, model.q_degrees):
        if abs(got - want) > 1e-6:
            errors.append(f"提取公式回代得到 {recovered}，声明是 {model.q_degrees}")
            break
    return errors


def fully_changed_spec() -> ArmSpec:
    """All seven declarations differ from the checked sample.

    Wind and wrist keep the same new axes. The injected R is the joint product,
    so those two frames stay aligned at zero joints and D=R*S remains the
    body-to-wind attitude.
    """
    return ArmSpec(
        wind_declared={"X": "up", "Y": "right"},
        wind_chirality="left",
        wrist_declared={"X": "up", "Y": "right"},
        wrist_chirality="left",
        body_declared={"Y": "forward", "Z": "up"},
        body_chirality="left",
        order=("axial", "lateral", "normal"),
        joint_sign={"normal": -1, "lateral": -1, "axial": -1},
        install_up="up",
        alpha_toward="up",
        beta_toward="left",
        gamma_sense=-1,
        role_deg={"normal": 30, "lateral": 30, "axial": -60},
    )


def _matrix_zero(matrix: sp.Matrix) -> bool:
    return all(sp.simplify(matrix[index]) == 0 for index in range(len(matrix)))


def _audit_changed(model: ArmModel) -> list[str]:
    """Independent product for this one declaration. Do not reuse it for other specs."""
    errors = []
    changed = _changed_definitions(model.spec)
    if changed != ["风轴系", "球腕系", "体轴系", "绝对方向轴旋转顺序", "机械臂旋转角度正方向", "零位安装", "理论气流角正方向"]:
        errors.append(f"七项声明没有全部改动：{changed}")
    if model.wind != model.wrist:
        errors.append("风轴系与球腕系不同向，当前 D=R*S 不能当成体轴到风轴")
    install = sp.Matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    if not _matrix_zero(model.S - install):
        errors.append(f"安装矩阵与手算不同：{fmt_matrix(model.S)}")
    pose = rot_about(2, sp.rad(-60)) * rot_about(1, sp.rad(-30)) * rot_about(0, sp.rad(-30))
    hand = sp.simplify(pose * install)
    if not _matrix_zero(hand - model.D):
        errors.append("D 与 Rz(-60°)*Ry(-30°)*Rx(-30°)*S 不一致")
    absolute = sp.simplify(model.body * (hand.T * (model.wind.T * ABS["forward"])))
    if not _matrix_zero(absolute - model.airspeed_abs):
        errors.append("空速绝对分量与手算 D 不一致")
    expected = sp.Matrix([sp.Rational(3, 4), -sp.Rational(1, 2), sp.sqrt(3) / 4])
    if not _matrix_zero(absolute - expected):
        errors.append(f"空速绝对分量不是 [3/4, -1/2, sqrt(3)/4]：{absolute}")
    if model.airflow_text() != "-arctan(2/3), -arcsin(sqrt(3)/4), -arcsin(7*sqrt(61)/61)":
        errors.append(f"气流角文本与手算不符：{model.airflow_text()}")
    if not _in_operating_range(model):
        errors.append("这组角度不在声明的唯一逆解范围内")
    body_forward = sp.simplify(hand * (model.body.T * ABS["forward"]))
    body_up = sp.simplify(hand * (model.body.T * ABS["up"]))
    wind_up = sp.simplify(model.wind.T * ABS["up"])
    projected = sp.simplify(wind_up - wind_up.dot(body_forward) * body_forward)
    reference = sp.simplify(projected / sp.sqrt(projected.dot(projected)))
    side = _cross(body_forward, reference)
    geometric = sp.atan2(sp.simplify(body_up.dot(side)), sp.simplify(body_up.dot(reference)))
    declared = sp.simplify(-geometric)
    if abs(float(declared - model.gamma)) > 1e-9:
        errors.append("滚转角与零参考定义不一致")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="生成风洞与机械臂交叉审阅稿")
    parser.add_argument("--no-write", action="store_true", help="只核对，不写 generated 目录")
    args = parser.parse_args()
    if not GOLD.exists():
        raise SystemExit(f"找不到对照稿 {GOLD}")
    model = ArmModel(ArmSpec())
    text = render(model)
    errors = check(text, GOLD.read_text(encoding="utf-8"))
    errors.extend(f"句式 {item}" for item in validate(text))
    print(f"默认声明对照 sample_01.md：{len(errors)} 处不一致")
    for item in errors[:40]:
        print(item)
    variant = ArmModel(ArmSpec(order=("axial", "normal", "lateral")))
    reached = wall_reach(variant)
    variant_text = render(variant, reached)
    variant_errors = _variant_ok(variant_text, variant)
    print(f"顺序改为轴向、法向、侧向后：壁面贴合比例 {reached:.4f}，问题 {len(variant_errors)} 个")
    for item in variant_errors:
        print(item)
    print("顶点风轴坐标", format_wind_vector(sp.simplify(variant.nose_wind())))
    print("提取公式", variant.extraction())
    changed = ArmModel(fully_changed_spec())
    changed_errors = _audit_changed(changed)
    print(f"七项声明全部改动后的手算核对：{len(changed_errors)} 个问题")
    for item in changed_errors:
        print(item)
    reached_changed = wall_reach(changed) if not changed_errors else None
    changed_text = render(changed, reached_changed) if reached_changed is not None else ""
    if changed_text:
        changed_errors.extend(_variant_ok(changed_text, changed))
        changed_errors.extend(f"句式 {item}" for item in validate(changed_text))
        if "不是默认的第一行写法" in changed_text:
            changed_errors.append("逆解仍写了无法核对的第一行答案")
        if "当前给定量不全部满足这些条件" in changed_text:
            changed_errors.append("这组角度被写成不满足唯一逆解条件")
    print(f"七项声明全部改动后：壁面贴合比例 {reached_changed}，问题 {len(changed_errors)} 个")
    for item in changed_errors:
        print(item)
    if errors or variant_errors or changed_errors:
        raise SystemExit(1)
    if args.no_write:
        return
    GENERATED.mkdir(parents=True, exist_ok=True)
    default_path = GENERATED / "sample_01.md"
    variant_path = GENERATED / "sample_axial_normal_lateral.md"
    changed_path = GENERATED / "sample_all_definitions_changed.md"
    default_path.write_text(text, encoding="utf-8", newline="\n")
    variant_path.write_text(variant_text, encoding="utf-8", newline="\n")
    changed_path.write_text(changed_text, encoding="utf-8", newline="\n")
    print(f"已写入 {default_path}")
    print(f"已写入 {variant_path}")
    print(f"已写入 {changed_path}")


if __name__ == "__main__":
    main()
