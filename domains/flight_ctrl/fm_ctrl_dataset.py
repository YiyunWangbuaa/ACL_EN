# -*- coding: utf-8 -*-
"""Injection for the flight-mechanics control. The cloze is one manuscript, like the wind tunnel."""
from __future__ import annotations

import re

from cloze_score import format_choices, resolve_choice
from domains.wind_tunnel.dataset_generate import FORMAT_EXAMPLE, FORMAT_TEXT

PROCESS = "机体系相对地面系"
PRIOR = ""
N_BLANKS = 81
SLOT_MARK = re.compile(r"@@(\d+)@@")

# 与风洞「风轴系到体轴系」1–81 同构。风轴系写成地面系，体轴系写成机体系，天平系写成气流系。
# 句号隔开的空分次作答；同一句里没有句号隔开的空一次答完。
PROCESS_TEXT = """1.初始状态，体轴保持平正，为了计算机构角α_M、β_M、γ_M对理论迎角α、β和滚转角的影响，应当进行@@1@@转换。

2.由于地面系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@2@@、@@3@@、@@4@@。机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@5@@、@@6@@、@@7@@。所以有地面系向机体系平正位置处的前置变换矩阵RP1为@@8@@。该矩阵能够使@@9@@的坐标轴与自身绝对方向的关系与@@10@@的坐标轴与自身绝对方向的关系一致。此外，模型安装方式为@@11@@。此时模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与地面的绝对法向正方向和绝对侧向正方向@@12@@。所以还需要另一个前置变换矩阵RP2为@@13@@。该矩阵使得@@14@@的坐标轴方向与@@15@@的坐标方向相同。

3.根据机构结构，推断第一个旋转角为@@16@@。第二个旋转角为@@17@@。第三个旋转角为@@18@@。

4.当α_M>0时，模型采用@@19@@。α_M>0时，空速偏机体的@@20@@。理论迎角>0时，空速偏机体的@@21@@。α_M>0时，理论迎角的符号为@@22@@。这相当于@@23@@平正时，绕@@24@@旋转。绕该轴旋转方向为@@25@@，旋转α_M角度。对应的旋转矩阵R_α_M为@@26@@。

5.当β_M>0时，模型采用@@27@@。β_M>0时，空速偏机体的@@28@@。理论侧滑角>0时，空速偏机体的@@29@@。β_M>0时，理论侧滑角的符号为@@30@@。这相当于@@31@@平正时，绕@@32@@旋转。绕该轴旋转方向为@@33@@，旋转β_M角度。对应的旋转矩阵R_β_M为@@34@@。

6.当γ_M>0时，以模型绝对轴向正方向（前方）为旋转轴，与右手定则@@35@@。此时右翼偏机体的@@36@@。由于理论滚转角>0时，以机体绝对轴向正方向（前方）为旋转轴，与右手定则@@37@@。此时右翼偏机体的@@38@@。γ_M>0时，理论滚转角的符号为@@39@@。这相当于@@40@@平正时，绕坐标轴@@41@@旋转。绕该轴旋转方向为@@42@@，旋转γ_M角度。对应的旋转矩阵R_γ_M为@@43@@。

7.此过程涉及到坐标系转换，应当采用坐标系的@@44@@。此外由于机体系是@@45@@，故应当在旋转时采用@@46@@。必须先变换坐标轴对应关系，之后再变换姿态。写出[xb,yb,zb]^T=R1*R2*R3*R4*R5*[Xg,Yg,Zg]^T。R1到R5从前面得到的矩阵代号中选择。公式中对应的R1矩阵为@@47@@。对应的R2矩阵为@@48@@。对应的R3矩阵为@@49@@。对应的R4矩阵为@@50@@。对应的R5矩阵为@@51@@。

8.在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R1的展开式为@@52@@。R2的展开式为@@53@@。R3的展开式为@@54@@。写出R1*R2*R3符号化简后的总矩阵R_t，R_t=@@55@@。

9.由于空速矢量在平置机体系中的坐标与地面系的坐标无关，故可以确定，RP2*RP1乘地面系中的空速单位向量，结果一定是@@56@@。因此只需要研究@@57@@中的空速坐标到姿态角变化后的机体系的坐标映射即可。

10.计算[xb,yb,zb]^T=R_t*[X_V,Y_V,Z_V]^T。此时平置坐标系中的空速单位向量[X_V,Y_V,Z_V]是@@58@@。代入公式计算得到的xb=@@59@@。计算得到的yb=@@60@@。计算得到的zb=@@61@@。

11.当仅α_M>0时，推断xb为@@62@@。推断yb为@@63@@。推断zb为@@64@@。

12.当仅β_M>0时，推断xb为@@65@@。推断yb为@@66@@。推断zb为@@67@@。

13.此时α的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。其中α的角度符号为@@68@@。α的反三角函数符号为@@69@@。α分子的符号为@@70@@。α的分子分量为@@71@@。α分母的符号为@@72@@。α的分母分量为@@73@@。

14.此时β的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。其中β的角度符号为@@74@@。β的反三角函数符号为@@75@@。β分子的符号为@@76@@。β的分子分量为@@77@@。β分母的符号为@@78@@。β的分母分量为@@79@@。

15.初始状态，体轴保持平正，已知来流在与地面系轴向重合，吹向机头。当机构产生α_M、β_M、γ_M变化之后，机体处于一个新的姿态位置。现在需要根据已知量α_M、β_M、γ_M，推导新姿态位置下，机体的理论气流角α、β的表达式。这里给出公式模版：理论迎角α的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])，β的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。禁止把正负号放进反三角函数内部来代替外层符号。求得理论迎角α=@@80@@。求得理论侧滑角β=@@81@@。"""


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


def _options_text(item: dict) -> str:
    options = [str(option) for option in (item.get("options") or [])]
    return format_choices(options)


def _gold_text(item: dict) -> str:
    options = [str(option) for option in (item.get("options") or []) if option]
    if not options:
        return str(item["answer"])
    _letter, text = resolve_choice(str(item["answer"]), options)
    return text


def _review_slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}, "
        f"标签:{item['label']}, 答案:{_gold_text(item)}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, 选项:{_options_text(item)}, 输出格式:{item['format_code']}}}"


def _json_field(item: dict, example: bool, letter: str) -> str:
    blank_id = item["id"]
    if item.get("options"):
        return f"{blank_id}:{letter if example else '字母'}"
    if example:
        return f'{blank_id}:"{FORMAT_EXAMPLE[item["format_code"]]}"'
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


def _output_block(item: dict) -> str:
    lines = _output_preamble()
    lines.append(f"只输出一行 {_json_object([item])}。")
    if item.get("options"):
        lines.append("不要写选项原文、句号、解释、Markdown或LaTeX。")
    else:
        lines.extend([
            "双引号只表示 JSON 字符串，不是答案的一部分。不要输出引号以外的解释、Markdown或LaTeX。",
            "公式里的角度符号使用本题的α_M、β_M、γ_M。示例中的θ只展示格式，答案里不要写θ。",
            FORMAT_TEXT[item["format_code"]],
        ])
    lines.append(f"格式示例（不是本题答案）：{_json_object([item], example=True)}")
    return "\n".join(lines)


def _output_block_many(by_id: dict, blank_ids: list[str]) -> str:
    items = [by_id[blank_id] for blank_id in blank_ids]
    lines = _output_preamble()
    lines.extend([
        f"只输出一行 {_json_object(items)}。按编号顺序写在同一个对象里，不要拆成多个 JSON 对象。",
        "不要输出句子原文。",
        "公式里的角度符号使用本题的α_M、β_M、γ_M。示例中的θ只展示格式，答案里不要写θ。",
    ])
    for item in items:
        if item.get("options"):
            lines.append(f"ID {item['id']} 的值是选项字母。")
        else:
            code = item["format_code"]
            lines.append(f"ID {item['id']} 的值是{code}，必须用英文双引号包成字符串。{FORMAT_TEXT[code]}")
    lines.append(f"格式示例（不是本题答案）：{_json_object(items, example=True)}")
    return "\n".join(lines)


def _render_process(by_id: dict, targets: list[str] | None = None, keep_suffix: bool = False, memory=None) -> str:
    open_ids = list(targets or [])
    open_set = set(open_ids)
    last_open = open_ids[-1] if open_ids else None
    pieces: list[str] = []
    cursor = 0
    seen: list[str] = []
    seen_open = False
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        blank_id = match.group(1)
        if blank_id not in by_id:
            raise KeyError(f"机体系相对地面系没有 ID {blank_id}。")
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
                if SLOT_MARK.search(tail[:period]):
                    raise RuntimeError(f"ID {blank_id} 的句子里还有未纳入的空。")
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            if not keep_suffix:
                raise RuntimeError(f"ID {blank_id} 出现在当前句的待填槽之后。")
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_text(item) if memory is None else str(memory.get(item["id"], "")))
        cursor = match.end()
    if open_ids and not seen_open:
        raise KeyError(f"过程里没有当前句 {open_ids}。")
    pieces.append(PROCESS_TEXT[cursor:])
    if not open_ids and seen != [str(number) for number in range(1, N_BLANKS + 1)]:
        raise RuntimeError(f"过程空的顺序是 {seen}。")
    return "".join(pieces).strip()


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix: bool = False) -> str:
    group = next(item for item in _sentence_groups() if target in item)
    items = [by_id[blank_id] for blank_id in group]
    if len(group) == 1:
        lead = (
            f"只填写当前这一句。本次是分步问答，只输出一个 JSON 对象 {_json_object(items)}。"
            "不要解释，不要填写其他空。"
        )
        output = _output_block(items[0])
    else:
        listed = "、".join(group)
        lead = (
            f"填写这一句里的全部待填槽，ID 为 {listed}。"
            "这些槽属于同一句话，一次把这句话答完。"
            f"本次是分步问答，只输出一个 JSON 对象 {_json_object(items)}。"
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
        declaration.strip(),
        "",
        "【子过程】",
        PROCESS,
        "",
        "【过程】",
        _render_process(by_id, targets=group, memory=memory, keep_suffix=keep_suffix),
        "",
        output,
    ])


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [blank_id for group in _sentence_groups() for blank_id in group]
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
        lead, "", "【定义声明】", chain["declaration"].strip(), "",
        "【子过程】", PROCESS, "", "【过程】",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
