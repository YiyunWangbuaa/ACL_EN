# -*- coding: utf-8 -*-
"""Exposure audit for vision scene / object+stereo composers."""
from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.computer_vision.scene_answer_form import answers_equal  # noqa: E402
from domains.computer_vision.scene_contract_verifiers import (  # noqa: E402
    _product4,
    _t_cw1_from_r,
    _t_cw2_from_r,
)
from domains.computer_vision.scene_manuscript import gold_texts, questions  # noqa: E402

GOLD = gold_texts()
Q = questions()


def _bind(node: str, values: tuple[str, ...]) -> dict:
    return {"node": node, "bound": (node, values), "status": "filled"}


def latent(table: dict) -> int:
    return sum(1 for slot in table.values() if slot.get("status") == "independent")


def alias(table: dict, render) -> int:
    domains = [slot["bound"][1] for slot in table.values()]
    answers = [render(*combo) for combo in product(*domains)]
    counts = Counter(answers)
    return sum(n * (n - 1) // 2 for n in counts.values())


FACTORS = {
    "R_cw1/ID41": {
        "R1": _bind("37", (GOLD["37"], GOLD["20"])),
        "R2": _bind("38", (GOLD["38"], GOLD["32"])),
        "R3": _bind("39", (GOLD["39"], GOLD["26"])),
        "R4": _bind("40", (GOLD["40"], GOLD["32"])),
    },
    "t_cw1/ID43": {
        "R_cw1": _bind("41", (GOLD["41"], GOLD["20"])),
    },
    "R_cw2/ID84": {
        "R1": _bind("80", (GOLD["80"], GOLD["63"])),
        "R2": _bind("81", (GOLD["81"], GOLD["75"])),
        "R3": _bind("82", (GOLD["82"], GOLD["69"])),
        "R4": _bind("83", (GOLD["83"], GOLD["75"])),
    },
    "t_cw2/ID86": {
        "R_cw2": _bind("84", (GOLD["84"], GOLD["63"])),
    },
}

PENDING: tuple[str, ...] = ()

RENDER = {
    "R_cw1/ID41": lambda a, b, c, d: _product4(a, b, c, d),
    "t_cw1/ID43": lambda r: _t_cw1_from_r(r),
    "R_cw2/ID84": lambda a, b, c, d: _product4(a, b, c, d),
    "t_cw2/ID86": lambda r: _t_cw2_from_r(r),
}

TARGETS = {
    "R_cw1/ID41": ("41", "matrix"),
    "t_cw1/ID43": ("43", "vector"),
    "R_cw2/ID84": ("84", "matrix"),
    "t_cw2/ID86": ("86", "vector"),
}


def check() -> None:
    parts = []
    for name, table in FACTORS.items():
        if latent(table) != 0:
            raise RuntimeError(f"{name} latent should be 0")
        render = RENDER[name]
        score = alias(table, render)
        if score < 0:
            raise RuntimeError(name)
        values = tuple(slot["bound"][1][0] for slot in table.values())
        got = render(*values)
        target, fmt = TARGETS[name]
        if not answers_equal(got, GOLD[target], fmt):
            raise RuntimeError(f"{name} gold composition disagrees: {got!r} vs {GOLD[target]!r}")
        parts.append(f"{name} latent={latent(table)} alias={score}")
    print(
        f"stage 7 probes: filled {len(FACTORS)}, " + ", ".join(parts) + f"; unfilled {len(PENDING)}.",
        flush=True,
    )


if __name__ == "__main__":
    check()
