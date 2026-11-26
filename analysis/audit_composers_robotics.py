# -*- coding: utf-8 -*-
"""Exposure audit for robotics theory composers."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.robotics.answer_form import answers_equal  # noqa: E402
from domains.robotics.contract_verifiers import (  # noqa: E402
    _angles_from_components,
    _joints_from_pose,
    _pose_from_factored,
    _tool_components,
    _translate_expr,
)
from domains.robotics.manuscript import gold_texts, questions  # noqa: E402

GOLD = gold_texts()
Q = questions()


def _bind(node: str, values: tuple[str, ...]) -> dict:
    return {"node": node, "bound": (node, values), "status": "filled"}


def latent(table: dict) -> int:
    return sum(1 for slot in table.values() if slot.get("status") == "independent")


def alias(table: dict, render) -> int:
    keys = list(table)
    domains = [slot["bound"][1] for slot in table.values()]
    # Cartesian product of factor assignments; count colliding outputs.
    from itertools import product

    answers = []
    for combo in product(*domains):
        answers.append(render(*combo))
    counts = Counter(answers)
    return sum(n * (n - 1) // 2 for n in counts.values())


FACTORS = {
    "pose/F4": {
        "factored": _bind("F3", (GOLD["F3"], next(o for o in Q["F3"]["options"] if o != GOLD["F3"]))),
    },
    "tool_dir/F7": {
        "D": _bind("F4", (GOLD["F4"], GOLD["F4"].replace("sqrt(3)/2", "1/2"))),
        "rule": _bind("F6", (GOLD["F6"], next(o for o in Q["F6"]["options"] if o != GOLD["F6"]))),
    },
    "angles/F11": {
        "components": _bind("F7", (GOLD["F7"], "[1, 0, 0]")),
    },
    "shift/F13": {
        "offset": _bind("F12", (GOLD["F12"], "[0, L, 0]")),
    },
    "joints/I10": {
        "D": _bind("I8", (GOLD["I8"], GOLD["I8"].replace("sqrt(3)/2", "1/2"))),
        "erase": _bind("I9", (GOLD["I9"], next(o for o in Q["I9"]["options"] if o != GOLD["I9"]))),
    },
}

PENDING: tuple[str, ...] = ()


def _render_pose(factored: str) -> str:
    return _pose_from_factored(factored)


def _render_tool(d: str, rule: str) -> str:
    return _tool_components(d, rule)


def _render_angles(components: str) -> str:
    return _angles_from_components(components)


def _render_shift(offset: str) -> str:
    return _translate_expr(offset)


def _render_joints(d: str, erase: str) -> str:
    return _joints_from_pose(d, erase)


RENDER = {
    "pose/F4": _render_pose,
    "tool_dir/F7": _render_tool,
    "angles/F11": _render_angles,
    "shift/F13": _render_shift,
    "joints/I10": _render_joints,
}

TARGETS = {
    "pose/F4": ("F4", "matrix"),
    "tool_dir/F7": ("F7", "vector"),
    "angles/F11": ("F11", "direction-angles"),
    "shift/F13": ("F13", "expression"),
    "joints/I10": ("I10", "joint-angles"),
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
