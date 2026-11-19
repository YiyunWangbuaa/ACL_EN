# -*- coding: utf-8 -*-
"""Injection module for the CV-CTRL machine-vision control (对照组)."""
from __future__ import annotations

import json

from domains.computer_vision.cv_dataset import _PLAIN_VALUE, _requirement

PROCESS = "世界系到相机"

PRIOR = (
    "针孔相机成像 x = (K·S)·[R|t]·X；图像直线 l = (K·S)^(-T)·n，n 为光心与空间直线张成平面的法向。"
    "极线约束 x2^T·F·x1 = 0，F = (K2·S2)^(-T)·E·(K1·S1)^(-1)，E = [t21]_×·R21。"
    "旋转矩阵用被动变换、动轴内旋；矩阵元素写全，不展开连乘。"
)

N_BLANKS = 66


def _sentence_groups() -> list[list[str]]:
    return [[str(i)] for i in range(1, N_BLANKS + 1)]


def _render_prompt(item: dict) -> str:
    text = item["prompt"]
    options = item.get("options") or []
    if options:
        text += "\n选项：" + " ".join(f"{chr(65 + i)}.{opt}" for i, opt in enumerate(options))
    return text


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix=False) -> str:
    del memory, keep_suffix
    item = by_id[target]
    prompt = _render_prompt(item)
    return f"{declaration.strip()}\n\n【问题】{prompt}\n【要求】{_requirement([item])}"


def _render_process(by_id: dict, targets: list[str], keep_suffix=False) -> str:
    return "\n".join(_render_prompt(by_id[target]) for target in targets)


def _json_object(items: list[dict], example: bool = False) -> str:
    if example:
        return json.dumps({item["id"]: "示例" for item in items}, ensure_ascii=False)
    return json.dumps({item["id"]: "?" for item in items}, ensure_ascii=False)


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [str(i) for i in range(1, N_BLANKS + 1)]
    items = [by_id[lid] for lid in local_ids]
    lead = (
        "本次是端到端问答。填写这个子过程里的全部待填槽，只调用这一次。"
        "只输出一个 JSON 对象，键是空的编号，值是答案。"
        "选择题的值是字母，不加引号。矩阵的值用英文双引号包成字符串。"
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。"
        "不要输出句子原文，不要解释。"
        + _PLAIN_VALUE
    )
    output = "\n".join([
        "【输出】",
        "所有答案放在同一个 JSON 对象中。键是空的编号，值是答案。",
        "选择题的值只写字母，不加引号。矩阵必须用英文双引号包成字符串。",
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。",
        _PLAIN_VALUE,
        f"只输出一行 {_json_object(items)}。",
        f"格式示例（不是本题答案）：{_json_object(items, example=True)}",
    ])
    return "\n".join([
        lead, "", "【定义声明】", chain["declaration"].strip(), "", "【物理常识】",
        PRIOR, "", "【子过程】", PROCESS, "", "【过程】",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
