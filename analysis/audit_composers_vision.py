# -*- coding: utf-8 -*-
"""Exposure audit for vision basic composers."""
from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.computer_vision.answer_form import answers_equal  # noqa: E402
from domains.computer_vision.contract_verifiers import (  # noqa: E402
    _pc2_from_r_pw,
    _product4,
    _t21_from_r,
)
from domains.computer_vision.manuscript import gold_texts, questions  # noqa: E402

GOLD = gold_texts()
Q = questions()


def _bind(node: str, values: tuple[str, ...]) -> dict:
    return {"节点": node, "绑定": (node, values), "状态": "已填"}


def latent(table: dict) -> int:
    return sum(1 for slot in table.values() if slot.get("状态") == "独立")


def alias(table: dict, render) -> int:
    domains = [slot["绑定"][1] for slot in table.values()]
    answers = [render(*combo) for combo in product(*domains)]
    counts = Counter(answers)
    return sum(n * (n - 1) // 2 for n in counts.values())


FACTORS = {
    "R21/ID41": {
        "R1": _bind("37", (GOLD["37"], GOLD["20"])),
        "R2": _bind("38", (GOLD["38"], GOLD["32"])),
        "R3": _bind("39", (GOLD["39"], GOLD["26"])),
        "R4": _bind("40", (GOLD["40"], GOLD["32"])),
    },
    "t21/ID47": {
        "R21": _bind("41", (GOLD["41"], GOLD["20"])),
    },
    "Pc2/ID53": {
        "R21": _bind("41", (GOLD["41"], GOLD["20"])),
        "Pw": _bind("49", (GOLD["49"], "[L, 0, 0]")),
    },
}

PENDING: tuple[str, ...] = ()

RENDER = {
    "R21/ID41": lambda a, b, c, d: _product4(a, b, c, d),
    "t21/ID47": lambda r: _t21_from_r(r),
    "Pc2/ID53": lambda r, p: _pc2_from_r_pw(r, p),
}

TARGETS = {
    "R21/ID41": ("41", "矩阵"),
    "t21/ID47": ("47", "向量"),
    "Pc2/ID53": ("53", "向量"),
}


def check() -> None:
    for name, table in FACTORS.items():
        if latent(table) != 0:
            raise RuntimeError(f"{name} latent 应为 0")
        render = RENDER[name]
        if alias(table, render) < 0:
            raise RuntimeError(name)
        values = tuple(slot["绑定"][1][0] for slot in table.values())
        got = render(*values)
        target, fmt = TARGETS[name]
        if not answers_equal(got, GOLD[target], fmt):
            raise RuntimeError(f"{name} 金标合成不一致：{got!r} vs {GOLD[target]!r}")
    print(
        f"阶段7 测点：已填 {len(FACTORS)}，"
        + "，".join(f"{name} latent={latent(table)} alias={alias(table, RENDER[name])}" for name, table in FACTORS.items())
        + f"；未填 {len(PENDING)}。",
        flush=True,
    )


if __name__ == "__main__":
    check()
