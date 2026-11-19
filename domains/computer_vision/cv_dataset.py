# -*- coding: utf-8 -*-
"""Injection module for the computer-vision stereo-schlieren task.

One subprocess ("计算机视觉双目姿态估计") with 165 local slots. The
reconstruction process uses the same final-question format. Intermediate
slots stay self-contained. Slots that are also final-e2e answers share that
question text and are preceded by the earlier steps filled with gold.
"""
from __future__ import annotations

import json

PROCESS = "计算机视觉双目姿态估计"
# 重构稿只改说法，终态题的输出格式与交叉稿共用。
RECON_PROCESS = "纯机器视觉双目姿态估计"
_VISION_PROCESSES = {PROCESS, RECON_PROCESS}

PRIOR = (
    "针孔相机成像 x = (K·S)·[R|t]·X；图像直线 l = (K·S)^(-T)·n，n 为光心与空间直线张成平面的法向。\n"
    "极线约束 x2^T·F·x1 = 0，F = (K2·S2)^(-T)·E·(K1·S1)^(-1)，E = [t21]_×·R21。\n"
    "旋转矩阵用被动变换、动轴内旋；矩阵元素写全，不展开连乘。"
)

N_BLANKS = 165


def _span(start: int, stop: int) -> list[str]:
    return [str(index) for index in range(start, stop)]


# 与 final-e2e 同一次调用的终态空。键是这组的第一个局部编号。
_FINAL_GROUP = {
    "41": ["41"],
    "84": ["84"],
    "138": ["138"],
    "139": ["139"],
    "146": ["146"],
    "147": ["147"],
    "148": ["148", "149", "150"],
    "153": ["153", "154"],
    "158": ["158", "159"],
}
# 终态之前已经做完的步骤。分矩阵展开式算在该总矩阵的前缀里。
_CHAIN = {
    "41": _span(1, 41),
    "84": _span(44, 84),
    "138": [],
    "139": [],
    "146": _span(140, 146),
    "147": _span(140, 146),
    "148": _span(87, 136),
    "153": ["41", "84", "43", "86", "151", "152"],
    "158": [],
}
_FINAL_QUESTION = {
    "41": (
        "相机1的方位角、俯仰角、滚转角为 ψ1、θ1、φ1。"
        "现在需要根据这些角度和当前坐标轴定义，推导风轴系到相机1系的旋转矩阵。"
        "给出 R_cw1。"
    ),
    "84": (
        "相机2的方位角、俯仰角、滚转角为 ψ2、θ2、φ2。"
        "现在需要根据这些角度和当前坐标轴定义，推导风轴系到相机2系的旋转矩阵。"
        "给出 R_cw2。"
    ),
    "138": "给出直线公式中 (K1·S1)^(-T) 的展开式。不要与 [P_c1]_×、R_cw1、d_w 相乘。",
    "139": "给出直线公式中 (K2·S2)^(-T) 的展开式。不要与 [P_c2]_×、R_cw2、d_w 相乘。",
    "146": "给出直线公式中 [P_c1]_× 的展开式。不要与 (K1·S1)^(-T)、R_cw1、d_w 相乘。",
    "147": "给出直线公式中 [P_c2]_× 的展开式。不要与 (K2·S2)^(-T)、R_cw2、d_w 相乘。",
    "148": (
        "空速在风轴系中的方向向量d_wV。"
        "空速在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV。"
        "机体纵轴在机体系的单位向量为 e。它在风轴系中的方向向量为 d_wL。"
        "机体纵轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL。"
        "机体法向轴在机体系的单位向量为 e。它在风轴系中的方向向量为d_wN。"
        "机体法向轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN。"
        "写出 d_wV、d_wL、d_wN。"
    ),
    "153": (
        "相机1、相机2的光心在风轴系中的坐标分别为 C1=(x1,y1,z1)、C2=(x2,y2,z2)，"
        "姿态角为 ψ1、θ1、φ1 与 ψ2、θ2、φ2。"
        "现在需要推导相机1系到相机2系的旋转 R21 和平移 t21。"
        "给出 R21 和 t21。"
    ),
    "158": (
        "相机1、相机2图像上激波点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
        "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
        "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
        "现在需要给出这两个射线方向各自的展开式,写成向量形式。"
        "给出 d1、d2。"
    ),
}


def _sentence_groups() -> list[list[str]]:
    """One slot per sentence, except final-e2e bundles that are one question."""
    merged = {group[0]: group for group in _FINAL_GROUP.values() if len(group) > 1}
    skip = {local_id for group in merged.values() for local_id in group[1:]}
    groups = []
    for index in range(1, N_BLANKS + 1):
        local_id = str(index)
        if local_id in skip:
            continue
        groups.append(merged.get(local_id, [local_id]))
    return groups


def _group_of(target: str) -> list[str]:
    for group in _sentence_groups():
        if target in group:
            return group
    raise KeyError(target)


def _render_prompt(item: dict) -> str:
    text = item["prompt"]
    options = item.get("options") or []
    if options:
        text += "\n选项：" + " ".join(f"{chr(65 + i)}.{opt}" for i, opt in enumerate(options))
    return text


def _filled(item: dict, memory: dict | None) -> str:
    if memory is not None:
        return str(memory.get(str(item["id"]), ""))
    options = [str(option) for option in (item.get("options") or [])]
    answer = str(item.get("answer") or "")
    if options and len(answer) == 1 and answer.isalpha():
        index = ord(answer) - 65
        if 0 <= index < len(options):
            return options[index]
    return answer


def _chain_text(by_id: dict, local_ids: list[str], memory: dict | None) -> str:
    parts: list[str] = []
    previous = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            parts.append("".join(buffer).strip())
            buffer.clear()

    for local_id in local_ids:
        item = by_id[local_id]
        key = (item.get("segment"), item.get("paragraph"))
        if previous is not None and key != previous:
            flush()
        previous = key
        lead = item.get("lead") or item.get("prompt") or ""
        tail = item.get("tail") or ("" if item.get("lead") else "。")
        buffer.append(f"{lead}{_filled(item, memory)}{tail}")
    flush()
    return "\n".join(part for part in parts if part)


# 值的字符与标准答案一致：整数和本题已出现的符号，不用 LaTeX。
_PLAIN_VALUE = (
    "值里只使用标准数据的写法：整数，以及本题已经出现的符号。"
    "乘号写*，除号写/。sin、cos、tan 必须带圆括号，写成 sin(θ1) 这种形式。"
    "不要使用 LaTeX，不要写反斜杠、\\frac、下标花括号或 $。"
    "不要写等号。"
)


def _value_shape(code: str) -> str:
    if code == "无":
        return "值只写选项字母，不加引号，不写选项原文。"
    if code == "矩阵":
        return (
            "值是一行 3×3 矩阵 [[a, b, c], [d, e, f], [g, h, i]]，用英文双引号包成字符串，元素写全。"
            "每个元素写成整数或已给出量的表达式。"
            "不要留下 P_c1x、P_c1y、P_c1z 或带下标花括号的未代入记号。"
        )
    if code == "向量":
        return "值是一行三个分量 [a, b, c]，用英文双引号包成字符串，不要写成矩阵。"
    return "值是一行表达式，用英文双引号包成字符串。"


def _json_ask(items: list[dict]) -> str:
    fields = []
    for item in items:
        if item["format_code"] == "无":
            fields.append(f"{item['id']}:字母")
        else:
            fields.append(f'{item["id"]}:"答案"')
    return "{" + ", ".join(fields) + "}"


def _requirement(items: list[dict], keep_steps: bool = False) -> str:
    shapes: list[str] = []
    seen: set[str] = set()
    for item in items:
        code = item["format_code"]
        if code not in seen:
            seen.add(code)
            shapes.append(_value_shape(code))
    text = (
        f"只输出一个 JSON 对象 {_json_ask(items)}。"
        "键必须使用给出的空号，一个空号对应一个值。"
        "这一句有多个空时，不要合成一个没有空号的式子。"
        + "".join(shapes)
        + _PLAIN_VALUE
    )
    if keep_steps:
        text += "不要复述已完成步骤。"
    return text


def _final_question(process: str, head: str) -> str:
    text = _FINAL_QUESTION[head]
    if process == RECON_PROCESS:
        from domains.computer_vision.cv_vision_context import rewrite

        return rewrite(text)
    return text


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix=False) -> str:
    del keep_suffix
    item = by_id[target]
    process = item.get("process")
    final_ids = {local_id for group in _FINAL_GROUP.values() for local_id in group}
    if process not in _VISION_PROCESSES or str(target) not in final_ids:
        prompt = _render_prompt(item)
        return f"{declaration.strip()}\n\n【问题】{prompt}\n【要求】{_requirement([item])}"
    group = _group_of(target)
    head = group[0]
    chain = _chain_text(by_id, _CHAIN[head], memory)
    asked = [by_id[local_id] for local_id in group]
    blocks = [declaration.strip(), ""]
    if chain:
        blocks.extend([
            "【已完成步骤】",
            "下面这些步骤已经按标准答案填好，用来完成本次终态。不要改写这些步骤，也不要再次填写它们。",
            chain,
            "",
        ])
    blocks.extend([
        "【问题】",
        _final_question(process, head),
        "【要求】",
        _requirement(asked, keep_steps=True),
    ])
    return "\n".join(blocks)


def _render_process(by_id: dict, targets: list[str], keep_suffix=False) -> str:
    return "\n".join(_render_prompt(by_id[target]) for target in targets)


def _json_object(items: list[dict], example: bool = False) -> str:
    if example:
        return json.dumps({item["id"]: "示例" for item in items}, ensure_ascii=False)
    return json.dumps({item["id"]: "?" for item in items}, ensure_ascii=False)


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [str(index) for index in range(1, N_BLANKS + 1)]
    items = [by_id[lid] for lid in local_ids]
    lead = (
        "本次是端到端问答。填写这个子过程里的全部待填槽，只调用这一次。"
        "只输出一个 JSON 对象，键是空的编号，值是答案。"
        "选择题的值是字母，不加引号。矩阵和向量的值用英文双引号包成字符串。"
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。"
        "不要输出句子原文，不要解释。"
        + _PLAIN_VALUE
    )
    output = "\n".join([
        "【输出】",
        "所有答案放在同一个 JSON 对象中。键是空的编号，值是答案。",
        "选择题的值只写字母，不加引号。矩阵和向量必须用英文双引号包成字符串。",
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。",
        _PLAIN_VALUE,
        f"只输出一行 {_json_object(items)}。",
        f"格式示例（不是本题答案）：{_json_object(items, example=True)}",
    ])
    return "\n".join([
        lead,
        "",
        "【定义声明】",
        chain["declaration"].strip(),
        "",
        "【物理常识】",
        PRIOR,
        "",
        "【子过程】",
        PROCESS,
        "",
        "【过程】",
        _render_process(by_id, targets=local_ids, keep_suffix=True),
        "",
        output,
    ])
