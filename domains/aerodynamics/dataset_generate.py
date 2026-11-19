# -*- coding: utf-8 -*-
"""Review samples for the aerodynamics ground-to-body subprocess.

Storage stays under data/flight/basic/ground_to_body.

Run::

    python -m domains.aerodynamics.dataset_generate
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from cloze_score import answer_letter, format_choices, resolve_choice
from domain_layout import process_dirs
from flight_frame_model import (
    ANGLE_NAMES,
    COORDINATE_AXES,
    DECLARED,
    FIXED,
    INFERRED,
    OWNERS,
    PHI,
    PSI,
    STANDARD_SPEC,
    THETA,
    FlightFrameModel,
)
from domains.aerodynamics.ground_body_generate import BLANK_COUNT, PROCESS, generate
from domains.aerodynamics.numbering import TOTAL_BLANKS, public_id, public_span


OUT_DIR = process_dirs("aerodynamics", "ground_to_body")["review"]
SLOT_MARK = re.compile(r"@@(\d+)@@")
MATRIX_FORMAT = (
    "输出一行3×3矩阵，形如[[a, b, c], [d, e, f], [g, h, i]]。"
    "元素为整数，或由本题已出现的符号组成的表达式。乘号用*。"
    "不要换行，不要使用LaTeX。"
)
FORMAT_TEXT = {"矩阵": MATRIX_FORMAT}
FORMAT_EXAMPLE = {"矩阵": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}
FORMAT_CODE = {MATRIX_FORMAT: "矩阵"}
HAND_ZH = {"right": "右手系", "left": "左手系"}
SOURCE_ZH = {DECLARED: "声明", INFERRED: "由手性推断", FIXED: "固定"}
ROLE_ZH = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
OWNER_ZH = {
    "ground": "地面",
    "body": "机体",
    "airflow": "气流",
    "stability": "稳定",
    "track": "航迹",
}
POINT_ZH = {
    "forward": "前方",
    "rearward": "后方",
    "up": "上方",
    "down": "下方",
    "left": "左侧",
    "right": "右侧",
}
BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
PRIOR = (
    "轴向正方向、法向正方向、侧向正方向分别是前方、下方、右方。"
    "地面轴向、机体轴向、气流轴向、稳定轴向、航迹轴向的正方向都是这个绝对前方，不是坐标轴的正方向。"
    "侧向正方向由法向叉乘轴向得到。"
    "单独变换偏航时，旋转轴是地面的绝对法向轴。单独变换俯仰时，旋转轴是地面的绝对侧向轴。单独变换滚转时，旋转轴是地面的绝对轴向轴。"
    "单独变换航迹方位时，旋转轴是地面的绝对法向轴。单独变换航迹倾斜时，旋转轴是地面的绝对侧向轴。单独变换航迹滚转时，旋转轴是航迹的绝对轴向轴。"
    "单独变换迎角时，旋转轴是机体的绝对侧向轴。单独变换侧滑时，旋转轴是稳定性坐标系的绝对法向轴。"
    "各角的正方向以定义声明为准。"
    "坐标系转换采用被动变换。姿态旋转、航迹旋转和气流角旋转采用动轴内旋。"
)
PROCESS_TEXT = """1.推导地面系向机体系的坐标变换，所有变换过程采用@@1@@变换。旋转使用@@2@@。在这种变换条件下最先旋转的角度对应的矩阵在矩阵乘法中处于最@@3@@。为了每次旋转不影响已经转动的旋转轴，应当以@@4@@。假设一开始机体系与地面系存在3个正的欧拉角度，以地面系为起点逐步向机体系转换，地面系每转换一次，认为地面系的坐标轴随动，转换过程中不再为中间坐标系重新定义新的符号，统一使用地面系的符号。

2.地面系中法向正方向对应@@5@@，侧向正方向对应@@6@@，轴向正方向对应@@7@@。机体系中法向正方向对应@@8@@，侧向正方向对应@@9@@，轴向正方向对应@@10@@。

3.偏航角为正时，机体轴向正方向在地面的投影在Xg的@@11@@侧。此时以@@12@@为转轴。@@13@@旋转ψ角度。对应的旋转矩阵R_ψ为@@14@@。

4.俯仰角为正时，机体在@@15@@。应当以@@16@@为转轴。@@17@@旋转θ角度。对应的旋转矩阵R_θ为@@18@@。

5.滚转角为正时，机体向机体的@@19@@滚转。应当以@@20@@为转轴。@@21@@旋转Φ角度。对应的旋转矩阵R_Φ为@@22@@。

6.当完全消除地面系与机体系的姿态角后，此时的地面系绝对方向轴与机体系重合，但是坐标轴方向与机体系不重合，需要一个RP_g的变换矩阵让此时的地面坐标轴与机体轴重合。RP_g=@@23@@。

7.[xb,yb,zb]=R1*R2*R3*R4[xg,yg,zg]公式中，R1到R4从前面得到的矩阵代号中选择。R1=@@24@@。R2=@@25@@。R3=@@26@@。R4=@@27@@。

在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R1的展开式为@@28@@。R2的展开式为@@29@@。R3的展开式为@@30@@。R4的展开式为@@31@@。

8.SθψΦ=R1*R2*R3*R4，计算SθψΦ=@@32@@。"""
_DECLARATION_REQUIRED = (
    "场景",
    "俯仰角 θ：",
    "偏航角 ψ：",
    "滚转角 Φ：",
    "地面系（",
    "- Xg：",
    "- Yg：",
    "- Zg：",
    "机体系（",
    "- xb：",
    "- yb：",
    "- zb：",
    "气流系（",
    "稳定性坐标系（",
    "航迹系（",
)
_DECLARATION_FORBIDDEN = (
    "定义变更", "保持统一默认", "不计入", "改为", "不是可变定义",
    "欧拉顺序", "航迹顺序", "气流顺序",
)


def _samples() -> list[dict]:
    standard = deepcopy(STANDARD_SPEC)
    psi_negative = deepcopy(STANDARD_SPEC)
    psi_negative[PSI] = {"sign": "negative"}
    theta_negative = deepcopy(STANDARD_SPEC)
    theta_negative[THETA] = {"sign": "negative"}
    phi_negative = deepcopy(STANDARD_SPEC)
    phi_negative[PHI] = {"sign": "negative"}
    body_left = deepcopy(STANDARD_SPEC)
    body_left["body_frame"] = {
        "chirality": "left",
        "Xb": {"along": "body_axial", "points": "body_forward"},
        "Yb": {"along": "body_lateral", "points": "body_right"},
    }
    ground_lateral = deepcopy(STANDARD_SPEC)
    ground_lateral["ground_frame"] = {
        "chirality": "right",
        "Yg": {"along": "ground_lateral", "points": "ground_left"},
    }
    return [
        {"id": "01", "title": "统一默认", "change": "五个坐标系和八个角度都取当前默认定义。", "spec": standard},
        {"id": "02", "title": "偏航取负", "change": "偏航角改为机头向地面右方偏时为负。", "spec": psi_negative},
        {"id": "03", "title": "俯仰取负", "change": "俯仰角改为机头向地面上方偏时为负。", "spec": theta_negative},
        {"id": "04", "title": "滚转取负", "change": "滚转角改为向机体右侧滚转时为负。", "spec": phi_negative},
        {"id": "05", "title": "机体左手", "change": "机体系改为左手系，xb 沿机体轴向指向机头，yb 沿机体侧向指向机体右侧。", "spec": body_left},
        {"id": "06", "title": "地面侧向改轴", "change": "地面系改为由 Yg 沿地面侧向指向地面左方，Zg 由右手系推断。", "spec": ground_lateral},
    ]


def _direction(points_to: str) -> str:
    owner, point = points_to.split("_", 1)
    return f"{OWNER_ZH[owner]}{POINT_ZH[point]}"


def _axis_name(owner: str, name: str) -> str:
    if owner == "body":
        return BODY_SHOW[name]
    return name


def _frame_lines(model: FlightFrameModel, owner: str) -> list[str]:
    title = "稳定性坐标系" if owner == "stability" else f"{OWNER_ZH[owner]}系"
    lines = [f"{title}（{HAND_ZH[model.chirality[owner]]}）"]
    for name in COORDINATE_AXES[owner]:
        axis = model.frames[owner][name]
        role = axis.absolute_axis.removeprefix(owner + "_")
        lines.append(
            f"- {_axis_name(owner, name)}：沿{OWNER_ZH[owner]}{ROLE_ZH[role]}，"
            f"指向{_direction(axis.points_to)}（{SOURCE_ZH[axis.source]}）。"
        )
    return lines


def _declaration(model: FlightFrameModel) -> str:
    lines = [
        "固定约定",
        "- 坐标系转换采用被动变换。姿态旋转、航迹旋转和气流角旋转采用动轴内旋。最先旋转的角度，其矩阵写在连乘式最右侧。",
        "- 地面坐标轴 Xg 沿地面轴向，指向地面前方。",
        "",
        "绝对方向",
        "地面系、机体系、气流系、稳定性坐标系、航迹系各有三根绝对轴。五套绝对轴的正方向定义相同：轴向正方向是前方，法向正方向是下方，侧向正方向是右方。六个绝对方向是这三根轴上的两个指向。",
        "下文凡写轴向正方向、法向正方向、侧向正方向，以及地面轴向、机体轴向、气流轴向、稳定轴向、航迹轴向的正方向，都是这三根绝对轴的正方向，不是 Xg、Yg、Zg，xb、yb、zb，Xa、Ya、Za，Xs、Ys、Zs，Xp、Yp、Zp 这些坐标轴的正方向。坐标轴声明只是指定每根坐标轴沿哪根绝对轴、指向哪一侧，可以与绝对正方向相同，也可以相反。",
        "五个系共用同一套指向关系：法向 × 轴向 = 侧向。沿轴向正方向看去，右方在右手侧。嵌入是前方 × 几何右方 = 下方。沿前方看去，几何右方在右手侧，下方铅垂向下。这个关系不随定义改变。五个坐标系的右方都指向几何右方。",
        "地面系的三轴六方向相对地面：",
        "- 轴向：地平面上的前后轴。正方向朝前，是 Xg 的正方向。后方是其反方向。",
        "- 法向：地面铅垂轴。正方向朝下，下方铅垂向下。上方是其反方向，铅垂向上。",
        "- 侧向：地面水平横轴。正方向由法向叉乘轴向得到，称为地面右方。左方是其反方向。",
        "机体系的三轴六方向相对机体本身：",
        "- 轴向：连接机尾与机头。正方向朝前，指向机头。后方指向机尾。",
        "- 法向：机体对称面内、垂直于理论纵轴的上下轴。正方向朝下，指向机体下方。上方是其反方向，指向机体上方。",
        "- 侧向：垂直于机体对称面的左右轴。正方向由法向叉乘轴向得到，称为机体右方。左方是其反方向。",
        "气流系的三轴六方向相对空速：",
        "- 轴向：与空速重合。正方向朝前，指向空速方向。后方是其反方向。",
        "- 法向：飞行器对称面内、垂直于空速的上下轴。正方向朝下。上方是其反方向，在飞行器对称面内、垂直于空速向上。",
        "- 侧向：由法向叉乘轴向得到，称为气流右方。左方是其反方向。",
        "稳定性坐标系的三轴六方向相对机体对称面内的空速投影：",
        "- 轴向：空速在飞行器对称面内的投影。正方向朝前，指向机头。后方是其反方向。",
        "- 法向：飞行器对称面内、垂直于轴向的上下轴。正方向朝下，指向机体下方。上方是其反方向，指向机体上方。",
        "- 侧向：垂直于飞行器对称面。正方向由法向叉乘轴向得到，称为稳定右方。左方是其反方向。",
        "航迹系的三轴六方向相对空速：",
        "- 轴向：与空速重合。正方向朝前，指向空速方向。后方是其反方向。",
        "- 法向：过空速的铅垂面内、垂直于空速的上下轴。正方向朝下，指向地面下方。上方是其反方向，在过空速的铅垂面内、垂直于空速，指向离开地面的方向。",
        "- 侧向：由法向叉乘轴向得到，称为航迹右方。左方是其反方向。",
        "",
        "场景",
        "这个场景考察飞行动力学中飞机、地面和气流之间的关系。机体系与地面系用来描述飞机姿态，对应偏航、俯仰和滚转。机体系与气流系用来描述气流和飞机的关系，中间经过稳定性坐标系，用来度量迎角和侧滑。航迹系与地面系用来描述飞行方向和地面的关系，对应方位和倾斜。航迹系与气流系之间还有航迹滚转。",
        "其中有一类特殊任务，初始状态下，机体系与地面系的绝对方向完全重合，并且气流方向与地面系的轴向重合，吹向机头。机体尾部固定在了一个机械结构装置上，该装置会采取一种全新的欧拉角变换方式将机体变换到一种特殊姿态。变换的顺序与正常变换顺序不同，必须先理解机构角度的变换特性，然后推断出正确的角度变换顺序。这些角度的定义方式已在下方的角度定义给出。这些角度把初始与地面系重合的机体系转到一个特殊的区别于传统姿态角位置的机体系。在这个位置下，需要再按理论迎角、侧滑角的定义重新计算该位置下的理论气流角。",
        "这个过程中有如下限制：",
        "- 必须先变换坐标轴对应关系，之后再变换姿态。写成 [xb,yb,zb]^T=R1*R2*R3*R4*R5*[Xg,Yg,Zg]^T。连乘从右向左先写前置矩阵，再写机构角矩阵。最先旋转的机构角，其矩阵紧挨着前置矩阵。",
        "- 地面系向机体系、平正位置处的前置变换矩阵是 RP1。该矩阵使机体系的坐标轴与自身绝对方向的关系，与地面系的坐标轴与自身绝对方向的关系一致。",
        "- 安装只考虑正装。正装时，平正状态下的机体上方和地面上方一致。此时机体绝对法向正方向（上方）和绝对侧向正方向（右侧）与地面的绝对法向正方向和绝对侧向正方向相同。如果反装时机体上方与地面上方相反，所以还需要另一个前置变换矩阵 RP2。该矩阵使机体系的坐标轴方向与地面系的坐标方向相同。正装时 RP2 为单位阵。",
        "",
        "角度",
    ]
    for name in ANGLE_NAMES:
        record = model.binding(name)
        lines.append(f"{record.title} {record.symbol}：{record.definition}。")
    lines.extend([
        "机构角",
        "机构迎角 α_M：功角支撑杆在铅垂面上运动，可视为绕地面系的侧向轴运动，为支撑杆（机体轴向）与水平面的夹角。机体偏向地面上方时为正。",
        "机构侧滑角 β_M：侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，即可视为侧滑角机构基于功角机构在作动。β_M是侧滑角支撑杆与铅垂面的夹角。使机体偏向地面右侧时为正。",
        "机构滚转角 γ_M：机体绕侧滑角支撑杆的转角。角度为正时，以模型绝对轴向正方向（前方）旋转，与右手定则方向相同。",
        "",
        "坐标系",
    ])
    for owner in OWNERS:
        lines.extend(_frame_lines(model, owner))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _check_declaration(declaration: str) -> None:
    missing = [item for item in _DECLARATION_REQUIRED if item not in declaration]
    if missing:
        raise RuntimeError(f"定义声明缺少：{'、'.join(missing)}")
    present = [item for item in _DECLARATION_FORBIDDEN if item in declaration]
    if present:
        raise RuntimeError(f"定义声明仍在对照变更：{'、'.join(present)}")


def _question(item: dict) -> dict:
    output_format = None if item["options"] else MATRIX_FORMAT
    if item["options"] and item["answer"] not in item["options"]:
        raise RuntimeError(f"ID {item['manuscript_id']} 的答案不在选项中。")
    if output_format and not str(item["answer"]).startswith("[["):
        raise RuntimeError(f"ID {item['manuscript_id']} 不是矩阵。")
    return {
        "process": PROCESS,
        "id": public_id(PROCESS, item["manuscript_id"]),
        "local_id": item["manuscript_id"],
        "prompt": item["prompt"],
        "options": item["options"],
        "output_format": output_format,
        "format_code": "无" if output_format is None else FORMAT_CODE[output_format],
        "label": item["label"],
        "answer": answer_letter(item["answer"], item.get("options") or []),
    }


def _options_text(item: dict) -> str:
    return format_choices([str(option) for option in (item.get("options") or [])])


def _gold_phrase(item: dict) -> str:
    answer = str(item["answer"])
    options = [str(option) for option in (item.get("options") or []) if option and str(option) != "无"]
    if not options:
        return answer
    _letter, text = resolve_choice(answer, options)
    return text


def _review_slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}, "
        f"标签:{item['label']}, 答案:{item['answer']}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}}}"


def _sentence_groups() -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    cursor = 0
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        if "。" in PROCESS_TEXT[cursor:match.start()] and current:
            groups.append(current)
            current = []
        current.append(match.group(1))
        cursor = match.end()
    if current:
        groups.append(current)
    return groups


def _render_process(
    by_id: dict[str, dict],
    target: str | None = None,
    targets: list[str] | None = None,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
) -> str:
    if targets is None and target is not None:
        targets = next(group for group in _sentence_groups() if target in group)
    open_ids = list(targets or [])
    open_set = set(open_ids)
    last_open = open_ids[-1] if open_ids else None
    pieces: list[str] = []
    cursor = 0
    seen: list[str] = []
    seen_open = False
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        blank_id = match.group(1)
        seen.append(blank_id)
        pieces.append(PROCESS_TEXT[cursor:match.start()])
        item = by_id[blank_id]
        if not open_set:
            pieces.append(_review_slot(item))
        elif blank_id in open_set:
            seen_open = True
            pieces.append(_open_slot(item))
            if blank_id == last_open and not keep_suffix:
                tail = PROCESS_TEXT[match.end():]
                period = tail.find("。")
                if period == -1:
                    raise RuntimeError(f"ID {blank_id} 所在句没有句号。")
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_phrase(item) if memory is None else memory.get(item["id"], ""))
        cursor = match.end()
    pieces.append(PROCESS_TEXT[cursor:])
    if open_ids and not seen_open:
        raise KeyError(f"No sentence contains {target}.")
    if not open_ids and seen != [str(number) for number in range(1, BLANK_COUNT + 1)]:
        raise RuntimeError(f"Blank order {seen} does not match 1 through {BLANK_COUNT}.")
    return "".join(pieces).strip()


def _json_object(items: list[dict], example: bool = False) -> str:
    fields = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_index = 0
    for item in items:
        if item.get("options"):
            letter = letters[choice_index % len(letters)]
            choice_index += 1
            fields.append(f"{item['id']}:{letter if example else '字母'}")
        elif example:
            fields.append(f'{item["id"]}:"{FORMAT_EXAMPLE[item["format_code"]]}"')
        else:
            fields.append(f'{item["id"]}:"答案"')
    return "{" + ", ".join(fields) + "}"


def _injection(
    declaration: str,
    by_id: dict[str, dict],
    target: str,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
) -> str:
    group = next(item for item in _sentence_groups() if target in item)
    items = [by_id[blank_id] for blank_id in group]
    if len(group) == 1:
        lead = f"只填写当前这一句。本次是分步问答，只输出一个 JSON 对象 {_json_object(items)}。不要解释，不要填写其他空。"
    else:
        listed = "、".join(item["id"] for item in items)
        lead = (
            f"填写这一句里的全部待填槽，ID 为 {listed}。"
            f"本次是分步问答，只输出一个 JSON 对象 {_json_object(items)}。"
            "不要拆成多个对象。选择题的值是字母，不加引号。矩阵的值用英文双引号包成字符串。"
        )
    if keep_suffix:
        lead += "句号之后的后续正文只供参考，不要填写那些空。"
    output = [
        "【输出】",
        "所有答案都放在 JSON 对象中。键是空的编号，值是答案。",
        "选择题的值只写字母，不加引号。矩阵必须用英文双引号包成字符串。",
        f"只输出一行 {_json_object(items)}。",
        f"格式示例（不是本题答案）：{_json_object(items, example=True)}",
    ]
    return "\n".join([
        lead,
        "",
        "【定义声明】",
        declaration.strip(),
        "",
        "【物理常识】",
        PRIOR,
        "",
        "【子过程】",
        PROCESS,
        "",
        "【过程】",
        _render_process(by_id, targets=group, memory=memory, keep_suffix=keep_suffix),
        "",
        "\n".join(output),
    ])


def _render_markdown(record: dict) -> str:
    lines = [
        f"# 数据 {record['id']} {record['title']}",
        "",
        f"本文件是空气动力学领域、地面系到机体系子过程的审阅稿。三个子过程的空连续编号，从 1 到 {TOTAL_BLANKS}。这一子过程是 {public_span(PROCESS)[0]} 到 {public_span(PROCESS)[1]}。评测时一次答一句话。这一句之前的空替换成选项原文，这一句里的空保留为待填槽。",
        "",
        "## 定义声明",
        "",
        record["declaration"].strip(),
        "",
        "## 物理常识",
        "",
        PRIOR,
        "",
        "## 完整过程",
        "",
        "下面每个槽都带标签和标准答案，便于直接审阅。这是审阅字段，不是模型在待填空上看到的字段。",
        "",
        f"### {PROCESS}",
        "",
        record["process_text"],
        "",
    ]
    return "\n".join(lines)


def build_record(sample: dict) -> dict:
    model = FlightFrameModel().declare(deepcopy(sample["spec"]))
    payload = generate(deepcopy(sample["spec"]))
    questions = [_question(item) for item in payload["items"]]
    if len(questions) != BLANK_COUNT:
        raise RuntimeError(f"{sample['id']} produced {len(questions)} questions.")
    by_id = {item["local_id"]: item for item in questions}
    declaration = _declaration(model)
    _check_declaration(declaration)
    return {
        "id": sample["id"],
        "title": sample["title"],
        "change": sample["change"],
        "spec": sample["spec"],
        "declaration": declaration,
        "prior": PRIOR,
        "process": PROCESS,
        "process_text": _render_process(by_id),
        "questions": questions,
        "domain": "空气动力学",
    }


def _check_sentences(record: dict) -> None:
    by_id = {item["local_id"]: item for item in record["questions"]}
    covered = [blank_id for sentence in _sentence_groups() for blank_id in sentence]
    if covered != [str(number) for number in range(1, BLANK_COUNT + 1)]:
        raise RuntimeError(f"句子分组没有覆盖 1 到 {BLANK_COUNT}：{covered}")
    for sentence in _sentence_groups():
        kinds = [by_id[blank_id]["label"] for blank_id in sentence]
        families = {kind[0] for kind in kinds}
        if "A" in families and "C" in families:
            raise RuntimeError(f"A 与 C 在同一句：{list(zip(sentence, kinds))}")
        if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
            raise RuntimeError(f"A 类空没有单独成句：{list(zip(sentence, kinds))}")
    opened = _injection(record["declaration"], by_id, "11").split("【过程】\n", 1)[1]
    process = opened.split("【输出】", 1)[0].rstrip()
    if not process.endswith("侧。") or "{ID:11, 选项:A：左；B：右, 输出格式:无}" not in process or "为转轴" in process:
        raise RuntimeError("ID 11 没有单独收在句号处。")
    earlier = _injection(record["declaration"], by_id, "12").split("【过程】\n", 1)[1]
    if "在Xg的右侧" not in earlier or "在Xg的A侧" in earlier:
        raise RuntimeError("空前的标准答案没有写成选项原文。")
    matrix = _injection(record["declaration"], by_id, "14")
    if '{14:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}' not in matrix:
        raise RuntimeError("矩阵没有要求用双引号放进 JSON。")
    shared = _injection(record["declaration"], by_id, "5")
    if "{5:字母, 6:字母, 7:字母}" not in shared:
        raise RuntimeError("同一句的三个坐标轴没有放进一个 JSON 对象。")
    remembered = _injection(record["declaration"], by_id, "11", memory={"7": "MEMTEST"})
    if "MEMTEST" not in remembered:
        raise RuntimeError("前序空没有换成模型自己的输出。")


def _format_document(example: str) -> str:
    return "\n".join([
        "# 空气动力学注入格式",
        "",
        f"领域是空气动力学。三个子过程的空连续编号，从 1 到 {TOTAL_BLANKS}。当前子过程是地面系到机体系，编号 {public_span(PROCESS)[0]} 到 {public_span(PROCESS)[1]}。",
        "",
        "一次调用评测一句话。输入顺序是作答要求、【定义声明】、【物理常识】、【子过程】、【过程】、【输出】。选择题的值是字母。矩阵用英文双引号包成字符串。",
        "",
        "A 类空单独成句，不和 C 类空放在同一句。A：空前写入选项原文，句号后截断。B：空前换成模型上一句的原始输出。C′：空前是选项原文，并保留句号后的正文。C″：空前是模型自己的原始输出，同时保留后续正文。",
        "",
        "## 物理常识",
        "",
        PRIOR,
        "",
        f"- 矩阵：{MATRIX_FORMAT} 示例：{FORMAT_EXAMPLE['矩阵']}",
        "",
        "## 一次真实注入",
        "",
        "下面是数据 01、地面系到机体系、ID 11 的完整输入。",
        "",
        "```text",
        example,
        "```",
        "",
    ])


def _index(samples: list[dict]) -> str:
    lines = [
        "# 空气动力学审阅样本",
        "",
        f"子过程：地面系到机体系。每条 {BLANK_COUNT} 个空，公共编号 {public_span(PROCESS)[0]} 到 {public_span(PROCESS)[1]}。数据在 data/flight/basic/ground_to_body/review。",
        "",
        "| 编号 | 标题 | 说明 |",
        "| --- | --- | --- |",
    ]
    for sample in samples:
        lines.append(f"| {sample['id']} | {sample['title']} | {sample['change']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    records = [build_record(sample) for sample in samples]
    _check_sentences(records[0])
    for sample, record in zip(samples, records):
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(_render_markdown(record), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(
        json.dumps(
            [{key: value for key, value in record.items()} for record in records],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    example = _injection(
        records[0]["declaration"],
        {item["local_id"]: item for item in records[0]["questions"]},
        "11",
    )
    (OUT_DIR / "00_注入格式.md").write_text(_format_document(example), encoding="utf-8")
    (OUT_DIR / "00_索引.md").write_text(_index(samples), encoding="utf-8")
    print(f"Wrote {len(records)} aerodynamics review samples, {BLANK_COUNT} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
