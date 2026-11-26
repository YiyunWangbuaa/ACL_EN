# -*- coding: utf-8 -*-
"""Ten strong perturbations. Every variable definition leaves the standard value.

Euler order is fixed by the angle mechanism and is not a definition, so it stays
alpha, then beta, then gamma. The three frames are left-handed, and both declared
axes of each frame lie on a different absolute axis from the standard frame.
The ten samples share that full reversal and differ in how the axes are assigned.

Run::

    python -m domains.wind_tunnel.stress_generate
"""

from __future__ import annotations

import json
from copy import deepcopy

from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    AXIS_ZH,
    DIRECTION_ZH,
    _format_document,
    _render_markdown,
    build_record,
)
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

OUT_DIR = DOMAINS["wind_tunnel"]["stress"]
POINTS = {
    "axial": ("forward", "rearward"),
    "normal": ("up", "down"),
    "lateral": ("left", "right"),
}
# X leaves the axial axis. Y leaves the normal axis. Signs select the two pointings.
LAYOUTS = [
    ("normal", 0, "lateral", 0),
    ("normal", 1, "lateral", 1),
    ("normal", 0, "lateral", 1),
    ("normal", 1, "lateral", 0),
    ("normal", 0, "axial", 0),
    ("normal", 1, "axial", 1),
    ("normal", 0, "axial", 1),
    ("normal", 1, "axial", 0),
    ("lateral", 0, "axial", 0),
    ("lateral", 1, "axial", 1),
]
FRAME_SLOTS = (
    ("body_frame", "body", "XTZ", "YTZ", True),
    ("wind_frame", "tunnel", "XFD", "YFD", False),
    ("balance_frame", "balance", "XTP", "YTP", False),
)


def _frame(owner: str, x_name: str, y_name: str, with_order: bool, layout: tuple) -> dict:
    x_axis, x_sign, y_axis, y_sign = layout
    if x_axis == "axial" or y_axis == "normal" or x_axis == y_axis:
        raise RuntimeError(f"Layout {layout} still uses a standard absolute axis.")
    spec = {
        "chirality": "left",
        x_name: {
            "along": f"{owner}_{x_axis}",
            "points": f"{owner}_{POINTS[x_axis][x_sign]}",
        },
        y_name: {
            "along": f"{owner}_{y_axis}",
            "points": f"{owner}_{POINTS[y_axis][y_sign]}",
        },
    }
    if with_order:
        spec["euler_order"] = ["alpha", "beta", "gamma"]
    return spec


def _scalars() -> dict:
    spec = deepcopy(STANDARD_SPEC)
    spec["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma"] = {"right_wing_toward": "body_up", "sign": "positive"}
    spec["alpha_m"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta_m"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    spec["placement"] = "inverted"
    spec["alpha_e"] = {"axial_bends": "balance_down", "sign": "positive"}
    spec["beta_e"] = {"axial_bends": "balance_right", "sign": "positive"}
    spec["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    spec["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta_v"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma_tp"] = {"sense": "opposite"}
    spec["gamma_model"] = {"sense": "opposite"}
    return spec


def _axis_phrase(frame: dict, name: str) -> str:
    axis = frame[name]
    return f"{name} along {AXIS_ZH[axis['along']]}, pointing {DIRECTION_ZH[axis['points']]}"


def _check_against_standard(spec: dict) -> None:
    if set(spec) != set(STANDARD_SPEC):
        raise RuntimeError("The stress spec does not cover the same definitions as the standard spec.")
    for key, value in spec.items():
        if value == STANDARD_SPEC[key]:
            raise RuntimeError(f"{key} is still the standard definition.")
    for key, _owner, x_name, y_name, _order in FRAME_SLOTS:
        frame = spec[key]
        standard = STANDARD_SPEC[key]
        if frame["chirality"] == standard["chirality"]:
            raise RuntimeError(f"{key} chirality was not changed.")
        if frame[x_name]["along"] == standard[x_name]["along"]:
            raise RuntimeError(f"{key} {x_name} is still on the standard absolute axis.")
        if frame[y_name]["along"] == standard[y_name]["along"]:
            raise RuntimeError(f"{key} {y_name} is still on the standard absolute axis.")
        if "euler_order" in frame and frame["euler_order"] != ["alpha", "beta", "gamma"]:
            raise RuntimeError("Euler order is fixed and must stay alpha, beta, gamma.")


def _samples() -> list[dict]:
    samples = []
    shared = (
        "Relative to the common default, every variable definition leaves its original value. "
        "The theoretical angle of attack is positive when the airspeed points upward on the airframe, the theoretical sideslip is positive when the airspeed points left on the airframe, and the theoretical roll is positive when the right wing points upward on the airframe. "
        "The mechanism angle of attack, mechanism sideslip, and mechanism roll are reversed in the same direction as those theoretical angles. "
        "Installation is changed to inverted. "
        "The elastic angle of attack is positive when the axis bends downward on the balance, the elastic sideslip is positive when the axis bends right on the balance, and the elastic roll is positive when the lateral axis moves upward on the balance. "
        "The flow-offset angle α_V is positive when the airspeed points upward on the airframe, and β_V is positive when the airspeed points left on the airframe. "
        "The balance installation angle is opposite the right-hand rule about the positive absolute balance axis (forward). The model installation angle is opposite the right-hand rule about the positive absolute airframe axis (forward). "
        "The body, wind, and balance frames are all left-handed. The first declared axis leaves the axial axis, and the second leaves the normal axis. "
        "The Euler order is fixed by the angle-rod mechanism and is not a variable definition."
    )
    for index in range(10):
        spec = _scalars()
        phrases = []
        for offset, (key, owner, x_name, y_name, with_order) in zip((0, 3, 6), FRAME_SLOTS):
            spec[key] = _frame(owner, x_name, y_name, with_order, LAYOUTS[(index + offset) % 10])
            phrases.append(_axis_phrase(spec[key], x_name) + "; " + _axis_phrase(spec[key], y_name))
        _check_against_standard(spec)
        change = shared + " Axes in this row: " + "; ".join(phrases) + "."
        samples.append({
            "id": f"{index + 1:02d}",
            "title": f"every definition reversed, axis layout {index + 1}",
            "change": change,
            "spec": spec,
        })
    return samples


def _standard_answers() -> list[tuple[str, str]]:
    record = build_record({
        "id": "00",
        "title": "common default",
        "change": "Common default.",
        "spec": deepcopy(STANDARD_SPEC),
    })
    return [(item["id"], item["answer"]) for item in record["questions"]]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    standard = _standard_answers()
    records = []
    for sample in _samples():
        record = build_record(sample)
        answers = [(item["id"], item["answer"]) for item in record["questions"]]
        if len(answers) != len(standard):
            raise RuntimeError(f"{sample['id']} question count does not match the standard set.")
        changed = sum(left != right for left, right in zip(answers, standard))
        record["changed_answers"] = changed
        records.append(record)
        print(f"sample {sample['id']}: {changed}/{len(answers)} answers differ from the standard")
    stored = []
    for record in records:
        stored.append({
            key: value
            for key, value in record.items()
            if key not in ("injection_example", "changed_answers")
        })
    (OUT_DIR / "samples.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index = [
        "# Ten strong perturbations",
        "",
        "Each row moves every variable definition off the common default, to look for program errors under combined definitions. The Euler order is fixed by the angle-rod mechanism and is not perturbed. Single-factor checks still use data/wind_tunnel/review and are not mixed with these rows.",
        "",
        "The formal injection format is in 00_injection_format.md. Each review manuscript gives the declaration and the full text of the four subprocesses.",
        "",
    ]
    for record in records:
        path = OUT_DIR / f"sample_{record['id']}.md"
        text = _render_markdown(record).replace(
            "data/wind_tunnel/review/00_injection_format.md",
            "data/wind_tunnel/stress/00_injection_format.md",
        )
        path.write_text(text, encoding="utf-8")
        index.append(
            f"- Data {record['id']} {record['title']}: "
            f"{record['changed_answers']} blanks differ from the common default."
        )
    (OUT_DIR / "00_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (OUT_DIR / "00_injection_format.md").write_text(
        _format_document(records[0]["injection_example"]),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} samples, {len(records[0]['questions'])} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
