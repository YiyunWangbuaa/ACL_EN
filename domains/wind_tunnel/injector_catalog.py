# -*- coding: utf-8 -*-
"""Injector-row catalog for older wind-tunnel checks. This is not the D matrix.

Stage 4 is ``domains.wind_tunnel.dmatrix``. Binary reach is ``signature``;
ambiguity groups are ``dmatrix_graph``. This module only lists injector rows.
It does not inject them or record the value a calculator would write.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from domains.wind_tunnel import slot_signature as SS
from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.multi_slot import (
    Assignment,
    _manuscript,
    branch_table,
)
from domains.wind_tunnel.contract_verifiers import validator_of

_ROOT = Path(__file__).resolve().parents[2]
_SAMPLES = _ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
_DEFAULT = "standard-default"
_SPECIALS = frozenset({"24", "32", "41", "26", "34", "43", "88"})
# 分量正负。题面只问符号，故障写在符号空上，不改写被读取的表达式。
_COMPONENT_SIGNS = frozenset({"62", "63", "64", "65", "66", "67", "161", "162", "163"})
# 等效力矩符号。路径必须是模板值，换路径会被计算器拒绝，不进入正负判定。
# 故障写在符号空上，不改 ID184，也不改 ID185–ID188。
_MOMENT_SIGNS = frozenset({"189", "190", "191", "192", "193", "194"})
# 力矩分子力系数、分子坐标、分母力臂。故障写在这道选择题上，注入另一个选项，不重算上游。
_MOMENT_PARTS = frozenset(str(number) for number in range(201, 214))
_SIGN_BLANKS = _COMPONENT_SIGNS | _MOMENT_SIGNS | _MOMENT_PARTS
# 右翼偏向由右手定则推出，与 ID35、ID37 的识别故障是同一条，不再单列。
_HAND_RULE_WING = frozenset({"36", "38"})
_SUFFIX = {"choice": "选择故障", "sign": "判断故障", "matrix": "执行故障",
           "vector": "执行故障", "expression": "执行故障"}


@dataclass(frozen=True)
class Row:
    id: str
    klass: str                      # DIR | RSN | EXE | ALG
    label: str                      # 语义行标
    question: tuple[str, ...]       # 该行所属题目的全部空
    injector: tuple                 # ((blank, alt), ...)
    variant: str                    # std | all-swap | single:<blank> | branch | chain

    @property
    def degree(self) -> int:
        return len(self.injector)


# ---------------------------------------------------------------- 题目与依赖

@lru_cache(maxsize=1)
def _questions() -> dict[str, dict]:
    for line in _SAMPLES.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("id") != _DEFAULT:
            continue
        return {str(q["id"]): q for q in row["questions"]}
    raise RuntimeError("找不到 standard-default")


def _bare(x) -> str:
    return str(x).removeprefix("ID")


def _kind(node: str) -> str:
    node_obj = NODE_BY_ID.get(node)
    return "" if node_obj is None else node_obj.answer_type


@lru_cache(maxsize=None)
def calc_inputs(node: str) -> tuple[str, ...]:
    """计算器真正读的输入（不是图上的父节点）。"""
    node = _bare(node)
    return tuple(_bare(i) for i in SS._inputs(node))


@lru_cache(maxsize=None)
def calc_ancestors(node: str) -> frozenset[str]:
    """沿**计算器输入**向上闭包——判断注入能否被感知。"""
    seen: set[str] = set()
    pending = list(calc_inputs(node))
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pending.extend(calc_inputs(item))
    return frozenset(seen)


@lru_cache(maxsize=None)
def graph_ancestors(node: str) -> frozenset[str]:
    """沿**图上依赖**向上闭包——判断结构上是否可达。"""
    seen: set[str] = set()
    pending = list(NODE_BY_ID[_bare(node)].depends_on)
    while pending:
        item = _bare(pending.pop())
        if item in seen or item not in NODE_BY_ID:
            continue
        seen.add(item)
        pending.extend(NODE_BY_ID[item].depends_on)
    return frozenset(seen)


# ---------------------------------------------------------------- 行构造

def _label(prompt: str) -> str:
    text = re.sub(r"[？?。\s]+$", "", str(prompt)).strip()
    text = re.sub(r"^为计算[^，]*，", "", text)
    if len(text) > 40:
        text = text.split("。")[0]
    return text + "契约识别失败"


def e1_rows() -> tuple[Row, ...]:
    """DIR = 去重后的直接识别入口。一个题干一行，注入该题的全部空。"""
    from domains.wind_tunnel.fault_instance import e1_instances

    std = _manuscript(_DEFAULT)
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in e1_instances():
        for blank, _v in item.injector:
            grouped[str(_questions()[_bare(blank)]["prompt"]).strip()].add(_bare(blank))
    rows = []
    for prompt, blanks in grouped.items():
        ordered = tuple(sorted(blanks, key=int))
        injector = tuple((b, std.get(b, "")) for b in ordered)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        rows.append(Row(f"{DIR}:{digest}", DIR, _label(prompt), ordered, injector, "std"))
    rows.sort(key=lambda r: int(r.question[0]))
    return tuple(rows)


def application_questions() -> dict[str, dict]:
    """执行题：有计算器、且不是 DIR 直接识别空。"""
    e1_blanks = {b for row in e1_rows() for b in row.question}
    found = {}
    for node, q in _questions().items():
        if node in e1_blanks:
            continue
        if validator_of(node) is not None or node in _SPECIALS:
            found[node] = q
    return found


def e3_rows(sample: str, scope: str = "core") -> tuple[Row, ...]:
    """EXE = 执行题。每题给：单槽行（full 模式）+ 一条「全换」行。

    符号空、以及力矩的力系数、坐标分量和分母，只写这道题自己的另一侧，不走全换。
    """
    std = _manuscript(_DEFAULT)
    gold = _manuscript(sample)
    rows: list[Row] = []
    for node in sorted(application_questions(), key=int):
        q = _questions()[node]
        label = f"{q['role']}{_SUFFIX.get(str(q['answer_type']), '执行故障')}"
        if node in _HAND_RULE_WING:
            continue
        if node in _SIGN_BLANKS:
            alt = _other_side(node, gold.get(node, ""), sample)
            if alt and alt != gold.get(node, ""):
                kind = "choice" if node in _MOMENT_PARTS else "sign"
                digest = hashlib.sha1(f"{node}|{kind}|{alt}".encode("utf-8")).hexdigest()[:8]
                rows.append(Row(f"{EXE}:{node}|{kind}|{digest}", EXE, label,
                                (node,), ((node, alt),), kind))
            continue
        inputs = calc_inputs(node)
        if not inputs:
            continue
        if scope == "full":
            for blank in inputs:
                for alt in _alternatives(blank, sample):
                    if alt == gold.get(blank):
                        continue
                    digest = hashlib.sha1(f"{node}|{blank}|{alt}".encode("utf-8")).hexdigest()[:8]
                    rows.append(Row(f"{EXE}:{node}|{blank}|{digest}", EXE, label,
                                    (node,), ((blank, alt),), f"single:{blank}"))
        # 全换：所有输入同时取"另一侧"
        swapped = []
        for blank in inputs:
            alt = _other_side(blank, gold.get(blank, ""), sample)
            if alt is None:
                swapped = []
                break
            swapped.append((blank, alt))
        if swapped and all(alt != gold.get(b) for b, alt in swapped):
            digest = hashlib.sha1(f"{node}|all".encode("utf-8")).hexdigest()[:8]
            rows.append(Row(f"{EXE}:{node}|all|{digest}", EXE, label,
                            (node,), tuple(swapped), "all-swap"))
    return tuple(rows)


def _alternatives(blank: str, sample: str) -> tuple[str, ...]:
    """该空的可替换值：优先同族兄弟的答案，否则取 standard 值。"""
    std = _manuscript(_DEFAULT)
    gold = _manuscript(sample)
    found = []
    label = TARGETS.get(blank, ("", ""))[0]
    for other, node in NODE_BY_ID.items():
        if other == blank or TARGETS.get(other, ("", ""))[0] != label:
            continue
        if node.answer_type != NODE_BY_ID[blank].answer_type:
            continue
        value = gold.get(other, "")
        if value and value != gold.get(blank):
            found.append(value)
    standard = std.get(blank, "")
    if standard and standard != gold.get(blank):
        found.append(standard)
    return tuple(dict.fromkeys(found))


def _other_side(blank: str, current: str, sample: str) -> str | None:
    options = tuple(_questions()[blank].get("options") or ())
    if options:
        for option in options:
            if option and option != current:
                return option
        return None
    alts = [a for a in _alternatives(blank, sample) if a != current]
    return alts[0] if alts else None


def e2_rows(sample: str) -> tuple[Row, ...]:
    rows = []
    for name, assignment in branch_table(sample).items():
        if not assignment.slots:
            continue
        rows.append(Row(f"{RSN}:{name}", RSN, f"{name} 推理识别",
                        (), tuple(assignment.slots), "branch"))
    return tuple(rows)


def e4_rows(sample: str) -> tuple[Row, ...]:
    """ALG = 矩阵、向量、公式题。行名停在「角色数理错误」。分量符号不在这里。"""
    from domains.wind_tunnel.fault_instance import instances

    rows = []
    for item in instances(sample):
        if item.klass != ALG:
            continue
        node = item.injector[0][0]
        rows.append(Row(item.id, ALG, item.scope, (node,), tuple(item.injector), "algebra"))
    return tuple(rows)


def rows(sample: str, scope: str = "core") -> tuple[Row, ...]:
    return e1_rows() + e2_rows(sample) + e3_rows(sample, scope) + e4_rows(sample)


if __name__ == "__main__":
    print("这里不是 D 矩阵。阶段4入口是 domains.wind_tunnel.dmatrix。")
