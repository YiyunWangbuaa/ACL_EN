# -*- coding: utf-8 -*-
"""Exposure of composer factors: latent counts hidden factors, alias counts collisions.

Each target has a hand-written factor table. A factor is either

    {"bound": (question id, options)}
    {"internal": (options, source label)}

and never both. latent is the number of internal factors. alias is the number of
unordered pairs of assignments whose pasted strings are equal. An empty table
is unfilled: it is not latent=0, and it does not enter the completeness verdict.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.contract_verifiers import _compose_moment  # noqa: E402
from domains.wind_tunnel.ref_point_generate import (  # noqa: E402
    ARM_OPTIONS,
    DENOM_OPTIONS,
    FORCE_OPTIONS,
)


def _bind(question_id: str, options: tuple[str, ...]) -> dict:
    return {"bound": (question_id, options)}


def _hide(options: tuple[str, ...], source: str, repair: str) -> dict:
    """repair is independent (add a question) or derived (compute it from questions that already exist)."""
    if repair not in {"independent", "derived"}:
        raise RuntimeError(f"an internal factor is repaired only as independent or derived: {repair}")
    return {"internal": (options, source, repair)}


_SIGN = ("positive", "negative")

# Filled targets. The before-split table is a historical control that pins latent=3.
# It is not the current problem text.
FACTORS: dict[str, dict[str, dict]] = {
    "moment formula/after split": {
        "sign": _bind("ID190", _SIGN),
        "force coefficient": _bind("ID202", FORCE_OPTIONS),
        "coordinate": _bind("ID208", ARM_OPTIONS),
        "denominator": _bind("ID213", DENOM_OPTIONS),
    },
    "moment formula/before split": {
        "sign": _bind("ID190", _SIGN),
        "force coefficient": _hide(FORCE_OPTIONS, "moment name", "independent"),
        "coordinate": _hide(ARM_OPTIONS, "ID184", "independent"),
        "denominator": _hide(DENOM_OPTIONS, "constant L_RT", "independent"),
    },
    "attitude_target": {
        "frame": _bind("ID23/ID31/ID40", ("intermediate body frame", "balance frame", "wind frame")),
        "axis": _bind("ID24/ID32/ID41", ("XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP")),
        "sense": _bind("ID25/ID33/ID42", ("positive", "negative", "indeterminate")),
        "chirality": _hide(
            ("left", "right"),
            "read the declaration of the axis being rotated: body chirality is ID5/ID6/ID7, "
            "wind chirality is ID2/ID3/ID4, "
            "balance chirality is ID89/ID90/ID91; the triple is not passed in",
            "derived",
        ),
    },
    "roll_target": {
        "installation hand": _bind("ID83", ("positive", "negative")),
        "airframe hand": _bind("ID84", ("positive", "negative")),
        "axis": _bind("ID86", ("XTP", "YTP", "ZYP")),
        "sense": _bind("ID87", ("positive", "reverse")),
        "balance triple": _bind("ID89/ID90/ID91", ("XTP", "YTP", "ZYP", "-XTP", "-YTP", "-ZYP")),
        "chirality": _hide(
            ("left", "right"),
            "the axis stays in the balance frame, and matrix chirality still reads balance_chirality; "
            "ID89/ID90/ID91 are passed in and used only for the angle",
            "derived",
        ),
    },
    "_installation": {
        "axis": _bind("ID130", ("XTP", "YTP", "ZYP")),
        "sense": _bind("ID131", ("positive", "reverse")),
        "chirality": _hide(
            ("left", "right"),
            "the axis stays in the balance frame and reads balance_chirality; "
            "the inputs are only ID130/ID131, not ID89/ID90/ID91",
            "derived",
        ),
    },
    "_product": {
        "product code": _bind("ID47/ID48/ID49", ("RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M")),
        "convention": _bind("ID44/ID45/ID46", ("active", "passive", "moving axes", "fixed axes", "intrinsic", "extrinsic")),
    },
    "_alignment": {
        "row binding": _bind("ID5/ID6/ID7", ("XTZ", "YTZ", "ZTZ", "-XTZ", "-YTZ", "-ZTZ")),
        "column binding": _bind("ID2/ID3/ID4", ("XFD", "YFD", "ZFD", "-XFD", "-YFD", "-ZFD")),
    },
    "_mount": {
        "axis binding": _bind("ID5/ID6/ID7", ("XTZ", "YTZ", "ZTZ", "-XTZ", "-YTZ", "-ZTZ")),
        "installation relation": _bind("ID12/ID96", ("same", "opposite", "upright", "inverted")),
    },
}

# Empty tables do not enter the verdict. These composers are already in FACTORS.
PENDING: tuple[str, ...] = ()


def compose(sign: str, force: str, arm: str, denom: str) -> str:
    """Same paste as _compose_moment: optional minus, then force·arm/denom."""
    prefix = "-" if sign == "negative" else ""
    return f"{prefix}{force}·{arm}/{denom}"


RENDER = {
    "moment formula/after split": compose,
    "moment formula/before split": compose,
}


def _kind(slot: dict) -> str:
    if set(slot) == {"bound"}:
        return "bound"
    if set(slot) == {"internal"}:
        return "internal"
    raise RuntimeError(f"a factor is marked only bound or internal: {slot}")


def _options(slot: dict) -> tuple[str, ...]:
    kind = _kind(slot)
    if kind == "bound":
        _question, options = slot["bound"]
        return tuple(options)
    options, _source, _repair = slot["internal"]
    return tuple(options)


def latent(table: dict) -> int:
    """How many direct factors have no question of their own."""
    if not table:
        raise RuntimeError("an empty table is unfilled and is not latent=0")
    return sum(1 for slot in table.values() if _kind(slot) == "internal")


def sources(table: dict) -> tuple[tuple[str, str, str], ...]:
    """Internal factors, where the value comes from, and whether to add a question."""
    found = []
    for name, slot in table.items():
        if _kind(slot) == "internal":
            _options, source, repair = slot["internal"]
            found.append((name, source, repair))
    return tuple(found)


def bindings(table: dict) -> tuple[str, ...]:
    found = []
    for slot in table.values():
        if _kind(slot) == "bound":
            question, _options = slot["bound"]
            found.append(question)
    return tuple(found)


def alias(table: dict, render) -> int:
    """Unordered pairs of distinct assignments that paste the same string.

    The set notation (f, f') with f≠f' counts each collision twice. This
    function counts each collision once.
    """
    if not table:
        raise RuntimeError("an empty table is unfilled and is not alias=0")
    if render is None:
        raise RuntimeError("alias is not 0 when there is no composer")
    counts: dict[str, int] = {}
    lists = [_options(slot) for slot in table.values()]
    for values in product(*lists):
        text = render(*values)
        counts[text] = counts.get(text, 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def exposed(table: dict, render) -> bool:
    """Every direct factor has a question, and the pasted value tells them apart."""
    return latent(table) == 0 and alias(table, render) == 0


def audit() -> tuple[dict, ...]:
    """One row per filled target, then one unfilled row per reserved composer."""
    rows = []
    for name, table in FACTORS.items():
        render = RENDER.get(name)
        if render is None:
            rows.append({
                "target": name,
                "status": "filled",
                "latent": latent(table),
                "alias": None,
                "sources": sources(table),
                "exposed": None,
            })
            continue
        rows.append({
            "target": name,
            "status": "filled",
            "latent": latent(table),
            "alias": alias(table, render),
            "sources": sources(table),
            "exposed": exposed(table, render),
        })
    for name in PENDING:
        rows.append({
            "target": name,
            "status": "unfilled",
            "latent": None,
            "alias": None,
            "sources": (),
            "exposed": None,
        })
    return tuple(rows)


def _composer_agrees(table: dict) -> None:
    """The live moment composer pastes the same strings as the after-split fixture."""
    lists = [_options(slot) for slot in table.values()]
    for sign, force, arm, denom in product(*lists):
        got = _compose_moment(
            {"s": sign, "f": force, "a": arm, "d": denom},
            "s", "f", "a", "d",
        )
        if not got.ok or got.matrix != compose(sign, force, arm, denom):
            raise RuntimeError(f"pasted string disagrees with _compose_moment: {sign, force, arm, denom} -> {got}")


def check() -> None:
    after = FACTORS["moment formula/after split"]
    before = FACTORS["moment formula/before split"]
    if latent(after) != 0 or alias(after, compose) != 0 or not exposed(after, compose):
        raise RuntimeError("after split should be latent=0 and alias=0")
    if bindings(after) != ("ID190", "ID202", "ID208", "ID213"):
        raise RuntimeError(f"after-split question ids are wrong: {bindings(after)}")
    if sum(1 for _ in product(*(_options(slot) for slot in after.values()))) != 72:
        raise RuntimeError("after split should have 72 assignments")
    if latent(before) != 3:
        raise RuntimeError(f"before split latent should be 3, got {latent(before)}")
    expected = (
        ("force coefficient", "moment name", "independent"),
        ("coordinate", "ID184", "independent"),
        ("denominator", "constant L_RT", "independent"),
    )
    if sources(before) != expected:
        raise RuntimeError(f"before-split source labels are wrong: {sources(before)}")
    attitude = FACTORS["attitude_target"]
    if latent(attitude) != 1 or sources(attitude)[0][0] != "chirality" or sources(attitude)[0][2] != "derived":
        raise RuntimeError(f"chirality should be the only derived internal factor: {sources(attitude)}")
    if "ID2/ID3/ID4" not in sources(attitude)[0][1] or "ID89/ID90/ID91" not in sources(attitude)[0][1]:
        raise RuntimeError("the chirality source omitted the wind axes or the balance axes")
    for name in ("roll_target", "_installation"):
        found = sources(FACTORS[name])
        if latent(FACTORS[name]) != 1 or len(found) != 1 or found[0][2] != "derived":
            raise RuntimeError(f"{name} should have one derived chirality factor: {found}")
    for name in ("_product", "_alignment", "_mount"):
        if latent(FACTORS[name]) != 0:
            raise RuntimeError(f"{name} should have no internal factor among its direct factors")
        if name in RENDER:
            raise RuntimeError(f"{name} has no pasted string, so alias is not 0")
    if exposed(before, compose):
        raise RuntimeError("before split was recorded as fully exposed")
    _composer_agrees(after)
    filled = [row for row in audit() if row["status"] == "filled"]
    pending = [row for row in audit() if row["status"] == "unfilled"]
    if any(row["latent"] == 0 and row["status"] == "unfilled" for row in pending):
        raise RuntimeError("an unfilled target was treated as latent=0")
    if len(pending) != len(PENDING) or len(filled) != len(FACTORS):
        raise RuntimeError("audit rows did not separate filled from unfilled")


def _cell(value) -> str:
    return "—" if value is None else str(value)


def main() -> None:
    check()
    print("| target | status | latent | alias | sources |", flush=True)
    print("| --- | --- | ---: | ---: | --- |", flush=True)
    for row in audit():
        labels = ", ".join(f"{name}<-{label} ({repair})" for name, label, repair in row["sources"])
        print(
            f"| {row['target']} | {row['status']} | {_cell(row['latent'])} | "
            f"{_cell(row['alias'])} | {labels} |",
            flush=True,
        )
    filled = [row for row in audit() if row["status"] == "filled"]
    passed = [row["target"] for row in filled if row["exposed"] is True]
    hidden = [f"{row['target']} latent={row['latent']}" for row in filled if row["latent"]]
    unaliased = [row["target"] for row in filled if row["latent"] == 0 and row["alias"] is None]
    print(
        f"filled {len(filled)}: fully exposed {len(passed)} ({', '.join(passed) or 'none'}), "
        f"with internal factors {len(hidden)} ({', '.join(hidden) or 'none'}).",
        flush=True,
    )
    print(f"latent=0 but alias not computed: {', '.join(unaliased) or 'none'}.", flush=True)
    print(f"unfilled {len(PENDING)} are excluded from the full-table verdict.", flush=True)


if __name__ == "__main__":
    main()
