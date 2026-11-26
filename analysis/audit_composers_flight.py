# -*- coding: utf-8 -*-
"""Exposure audit for flight composers on blanks 1–131."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.aerodynamics.answer_form import answers_equal  # noqa: E402
from domains.aerodynamics.contract_verifiers import (  # noqa: E402
    _body_speed,
    _matrix_product,
    _matrix_transpose,
    _nose_ground,
    _track_speed,
)
from domains.aerodynamics.manuscript import gold_texts, questions  # noqa: E402


def _bind(question_id: str, options: tuple[str, ...]) -> dict:
    return {"bound": (question_id, options)}


def _hide(options: tuple[str, ...], source: str, repair: str) -> dict:
    if repair not in {"independent", "derived"}:
        raise RuntimeError(repair)
    return {"internal": (options, source, repair)}


def _kind(slot: dict) -> str:
    if set(slot) == {"bound"}:
        return "bound"
    if set(slot) == {"internal"}:
        return "internal"
    raise RuntimeError(slot)


def _options(slot: dict) -> tuple[str, ...]:
    kind = _kind(slot)
    if kind == "bound":
        return tuple(slot["bound"][1])
    return tuple(slot["internal"][0])


def latent(table: dict) -> int:
    if not table:
        raise RuntimeError("an empty table is unfilled")
    return sum(1 for slot in table.values() if _kind(slot) == "internal")


def sources(table: dict) -> tuple[tuple[str, str, str], ...]:
    found = []
    for name, slot in table.items():
        if _kind(slot) == "internal":
            _opts, source, repair = slot["internal"]
            found.append((name, source, repair))
    return tuple(found)


def alias(table: dict, render) -> int:
    if render is None:
        raise RuntimeError("alias is not 0 when there is no composer")
    counts: dict[str, int] = {}
    lists = [_options(slot) for slot in table.values()]
    for values in product(*lists):
        text = render(*values)
        counts[text] = counts.get(text, 0) + 1
    return sum(n * (n - 1) // 2 for n in counts.values())


GOLD = gold_texts()
Q = questions()

# Direct factors of the live composers. All bound — latent=0.
FACTORS = {
    "nose_ground/ID15": {
        "gRb": _bind("ID12", (GOLD["12"], next(o for o in Q["12"]["options"] if o != GOLD["12"]))),
        "body_forward": _bind("ID13", (GOLD["13"], "[0, 1, 0]")),
    },
    "track_speed/ID34": {
        "gRp": _bind("ID25", (GOLD["25"], next(o for o in Q["25"]["options"] if o != GOLD["25"]))),
    },
    "body_speed/ID45": {
        "bRa": _bind("ID42", (GOLD["42"], next(o for o in Q["42"]["options"] if o != GOLD["42"]))),
        "air_unit": _bind("ID43", (GOLD["43"], "[0, 1, 0]")),
    },
    "air_from_body/ID81": {
        "T_a_s": _bind("ID79", (GOLD["79"], GOLD["74"])),
        "T_s_b": _bind("ID80", (GOLD["80"], GOLD["77"])),
    },
    "body_from_air/ID83": {
        "T_a_b": _bind("ID81", (GOLD["81"], GOLD["74"])),
    },
}

PENDING: tuple[str, ...] = ()


def _render_nose(grb: str, forward: str) -> str:
    return _nose_ground({"12": grb, "13": forward})


def _render_track(grp: str) -> str:
    return _track_speed(grp)


def _render_body(bra: str, air: str) -> str:
    return _body_speed(bra, air)


def _render_air(left: str, right: str) -> str:
    return _matrix_product(left, right)


def _render_body_from_air(matrix: str) -> str:
    return _matrix_transpose(matrix)


RENDER = {
    "nose_ground/ID15": _render_nose,
    "track_speed/ID34": _render_track,
    "body_speed/ID45": _render_body,
    "air_from_body/ID81": _render_air,
    "body_from_air/ID83": _render_body_from_air,
}


def check() -> None:
    for name, table in FACTORS.items():
        if latent(table) != 0:
            raise RuntimeError(f"{name} latent should be 0")
        render = RENDER[name]
        if alias(table, render) < 0:
            raise RuntimeError(name)
        # Gold assignment must reproduce gold endpoint text.
        values = tuple(slot["bound"][1][0] for slot in table.values())
        got = render(*values)
        target = {
            "nose_ground/ID15": ("15", "vector"),
            "track_speed/ID34": ("34", "vector"),
            "body_speed/ID45": ("45", "vector"),
            "air_from_body/ID81": ("81", "matrix"),
            "body_from_air/ID83": ("83", "matrix"),
        }[name]
        if not answers_equal(got, GOLD[target[0]], target[1]):
            raise RuntimeError(f"{name} gold composition disagrees: {got!r} vs {GOLD[target[0]]!r}")
    print(
        f"stage 7 probes: filled {len(FACTORS)}, "
        + ", ".join(f"{name} latent={latent(table)} alias={alias(table, RENDER[name])}" for name, table in FACTORS.items())
        + f"; unfilled {len(PENDING)}.",
        flush=True,
    )


if __name__ == "__main__":
    check()
