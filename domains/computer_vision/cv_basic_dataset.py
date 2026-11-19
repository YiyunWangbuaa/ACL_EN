# -*- coding: utf-8 -*-
"""Injection for the basic-theory vision set.

One subprocess, 62 self-contained blanks (first-camera-as-world). Prompts use the
same slot format as the computer-vision sets.
"""
from __future__ import annotations

import json

from domains.computer_vision.cv_dataset import PRIOR, _PLAIN_VALUE, _injection, _render_process

PROCESS = "机器视觉基础理论"
N_BLANKS = 66


def _sentence_groups() -> list[list[str]]:
    return [[str(index)] for index in range(1, N_BLANKS + 1)]


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
        lead, "", "【定义声明】", chain["declaration"].strip(), "", "【物理常识】",
        PRIOR, "", "【子过程】", PROCESS, "", "【过程】",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
