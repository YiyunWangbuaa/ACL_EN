# -*- coding: utf-8 -*-
"""Exposure audit for robotics kinematics composers."""
from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.robotics.kin_answer_form import answers_equal  # noqa: E402
from domains.robotics.kin_contract_verifiers import (  # noqa: E402
    _joints_from_pose,
    _pose_from_factors,
    _tool_components,
    _translate_expr,
)
from domains.robotics.kin_manuscript import gold_texts, questions  # noqa: E402

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
    "pose/F45": {
        "R1": _bind("F41", (GOLD["F41"], GOLD["F42"])),
        "R2": _bind("F42", (GOLD["F42"], GOLD["F41"])),
        "R3": _bind("F43", (GOLD["F43"], GOLD["F41"])),
        "S": _bind("F44", (GOLD["F44"], GOLD["F41"])),
        "rule": _bind("F37", (GOLD["F37"], next(o for o in Q["F37"]["options"] if o != GOLD["F37"]))),
    },
    "tool_dir/F48": {
        "D": _bind("F45", (GOLD["F45"], GOLD["F45"].replace("sqrt(3)/2", "1/2"))),
        "rule": _bind("F47", (GOLD["F47"], next(o for o in Q["F47"]["options"] if o != GOLD["F47"]))),
    },
    "shift/F98": {
        "offset": _bind("F96", (GOLD["F96"], "[0, L + L_mount, 0]")),
    },
    "joints/I16": {
        "D": _bind("I9", (GOLD["I9"], GOLD["I9"].replace("sqrt(3)/2", "1/2"))),
        "erase": _bind("I14", (GOLD["I14"], next(o for o in Q["I14"]["options"] if o != GOLD["I14"]))),
        "S": _bind("I13", (GOLD["I13"], "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]")),
    },
}

PENDING: tuple[str, ...] = ()

RENDER = {
    "pose/F45": lambda a, b, c, d, rule: _pose_from_factors(a, b, c, d, rule),
    "tool_dir/F48": _tool_components,
    "shift/F98": _translate_expr,
    "joints/I16": lambda d, erase, install: _joints_from_pose(d, erase, install),
}

TARGETS = {
    "pose/F45": ("F45", "matrix"),
    "tool_dir/F48": ("F48", "vector"),
    "shift/F98": ("F98", "expression"),
    "joints/I16": ("I16", "joint-angles"),
}


def check() -> None:
    for name, table in FACTORS.items():
        if latent(table) != 0:
            raise RuntimeError(f"{name} latent should be 0")
        render = RENDER[name]
        if alias(table, render) < 0:
            raise RuntimeError(name)
        values = tuple(slot["bound"][1][0] for slot in table.values())
        got = render(*values)
        target, fmt = TARGETS[name]
        if not answers_equal(got, GOLD[target], fmt):
            raise RuntimeError(f"{name} gold composition disagrees: {got!r} vs {GOLD[target]!r}")
    print(
        f"stage 7 probes: filled {len(FACTORS)}, "
        + ", ".join(f"{name} latent={latent(table)} alias={alias(table, RENDER[name])}" for name, table in FACTORS.items())
        + f"; unfilled {len(PENDING)}.",
        flush=True,
    )


if __name__ == "__main__":
    check()
