# -*- coding: utf-8 -*-
"""One standard sample plus 100 declaration samples in tiers 25%, 50%, 75%, 100%.

A definition is one of 16 factors: 13 binary declarations that enter the cloze,
plus the body, wind, and balance frames. gamma_model is declared but no blank
reads it, so it stays at the standard value and is not one of the 16.
Euler order is fixed and is not a definition.

Each tier has 25 samples. A tier of k percent changes exactly 16*k/100 factors
away from the standard declaration. A changed frame is any of the 47 legal
frames other than the standard one.

Run::

    python -m domains.wind_tunnel.batch_generate
"""

from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    AXIS_ZH,
    DIRECTION_ZH,
    HAND_ZH,
    PROCESS_SIZE,
    REMOVED_SLOTS,
    build_record,
)
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

OUT_DIR = DOMAINS["wind_tunnel"]["batch"]
SEED = 20261102
PER_TIER = 25
TIERS = (25, 50, 75, 100)
BLANK_COUNT = sum(PROCESS_SIZE.values()) - sum(len(slots) for slots in REMOVED_SLOTS.values())
PILOT_COUNTS = {25: 3, 50: 3, 75: 2, 100: 2}
PILOT_SEED = 20261103

SCALAR_FLIP = {
    "alpha": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma": {"right_wing_toward": "body_up", "sign": "positive"},
    "alpha_m": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta_m": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma_m": {"right_wing_toward": "body_up", "sign": "positive"},
    "placement": "inverted",
    "alpha_e": {"axial_bends": "balance_down", "sign": "positive"},
    "beta_e": {"axial_bends": "balance_right", "sign": "positive"},
    "gamma_e": {"lateral_moves": "balance_up", "sign": "positive"},
    "alpha_v": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta_v": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma_tp": {"sense": "opposite"},
}
SCALAR_ZH = {
    "alpha": "theoretical angle of attack is positive when the airspeed points upward on the airframe",
    "beta": "theoretical sideslip is positive when the airspeed points left on the airframe",
    "gamma": "theoretical roll is positive when the right wing points upward on the airframe",
    "alpha_m": "mechanism angle of attack is positive when the balance deflects downward in the tunnel",
    "beta_m": "mechanism sideslip is positive when the balance deflects right in the tunnel",
    "gamma_m": "mechanism roll is opposite the right-hand rule about the positive absolute model axis (forward)",
    "placement": "installation is inverted",
    "alpha_e": "elastic angle of attack is positive when the axis bends downward on the balance",
    "beta_e": "elastic sideslip is positive when the axis bends right on the balance",
    "gamma_e": "elastic roll is positive when the lateral axis moves upward on the balance",
    "alpha_v": "flow-offset angle α_V is positive when the airspeed points upward on the airframe",
    "beta_v": "flow-offset angle β_V is positive when the airspeed points left on the airframe",
    "gamma_tp": "balance installation angle is opposite the right-hand rule about the positive absolute balance axis (forward)",
}
SCALAR_LABEL = {
    "alpha": "theoretical angle of attack",
    "beta": "theoretical sideslip",
    "gamma": "theoretical roll",
    "alpha_m": "mechanism angle of attack",
    "beta_m": "mechanism sideslip",
    "gamma_m": "mechanism roll",
    "placement": "installation",
    "alpha_e": "elastic angle of attack",
    "beta_e": "elastic sideslip",
    "gamma_e": "elastic roll",
    "alpha_v": "flow-offset angle α_V",
    "beta_v": "flow-offset angle β_V",
    "gamma_tp": "balance installation angle",
}
FRAMES = (
    ("body_frame", "body", "XTZ", "YTZ", True, "body-axis frame"),
    ("wind_frame", "tunnel", "XFD", "YFD", False, "wind frame"),
    ("balance_frame", "balance", "XTP", "YTP", False, "balance frame"),
)
FRAME_KEYS = [item[0] for item in FRAMES]
FACTORS = list(SCALAR_FLIP) + FRAME_KEYS
POINTS = {
    "axial": ("forward", "rearward"),
    "normal": ("up", "down"),
    "lateral": ("left", "right"),
}
AXES = ("axial", "normal", "lateral")


def _layouts(owner: str, x_name: str, y_name: str, with_order: bool, standard: dict) -> list[dict]:
    found = []
    for chirality in ("right", "left"):
        for x_axis in AXES:
            for x_sign in (0, 1):
                for y_axis in AXES:
                    if y_axis == x_axis:
                        continue
                    for y_sign in (0, 1):
                        spec = {
                            "chirality": chirality,
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
                        if spec != standard:
                            found.append(spec)
    if len(found) != 47:
        raise RuntimeError(f"{owner} has {len(found)} non-standard frames, expected 47.")
    return found


def _frame_phrase(title: str, spec: dict, x_name: str, y_name: str) -> str:
    x_axis = spec[x_name]
    y_axis = spec[y_name]
    return (
        f"{title} is changed to {HAND_ZH[spec['chirality']]}; "
        f"{x_name} lies along {AXIS_ZH[x_axis['along']]} and points {DIRECTION_ZH[x_axis['points']]}; "
        f"{y_name} lies along {AXIS_ZH[y_axis['along']]} and points {DIRECTION_ZH[y_axis['points']]}"
    )


def _change_text(tier: int, changed: list[str], spec: dict) -> str:
    phrases = []
    for key in changed:
        if key in SCALAR_ZH:
            phrases.append(SCALAR_ZH[key])
        else:
            _key, _owner, x_name, y_name, _order, title = next(item for item in FRAMES if item[0] == key)
            phrases.append(_frame_phrase(title, spec[key], x_name, y_name))
    kept = (
        "The model installation angle γ_model stays the same as the right-hand rule "
        "about the positive absolute airframe axis (forward) and is not one of these 16 items."
    )
    return (
        f"Definition-change tier {tier}%: {len(changed)} of the 16 variable definitions are changed. "
        + "; ".join(phrases)
        + f". The other counted variable definitions stay at the common default. {kept} "
        "Euler order is fixed by the angle mechanism and is not a variable definition."
    )


def _samples() -> list[dict]:
    if len(FACTORS) != 16:
        raise RuntimeError(f"Expected 16 factors, found {len(FACTORS)}.")
    layouts = {}
    for key, owner, x_name, y_name, with_order, _title in FRAMES:
        layouts[key] = _layouts(owner, x_name, y_name, with_order, STANDARD_SPEC[key])
    rng = random.Random(SEED)
    seen = set()
    samples = []
    for tier in TIERS:
        width = 16 * tier // 100
        pool = list(itertools.combinations(FACTORS, width))
        rng.shuffle(pool)
        made = 0
        cursor = 0
        while made < PER_TIER:
            chosen = list(pool[cursor % len(pool)])
            cursor += 1
            spec = deepcopy(STANDARD_SPEC)
            for key in chosen:
                if key in SCALAR_FLIP:
                    spec[key] = deepcopy(SCALAR_FLIP[key])
                else:
                    spec[key] = deepcopy(rng.choice(layouts[key]))
            signature = json.dumps(spec, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                if cursor > len(pool) * 20:
                    raise RuntimeError(f"Could not draw a unique spec for tier {tier}.")
                continue
            changed = [key for key in FACTORS if spec[key] != STANDARD_SPEC[key]]
            if changed != chosen and set(changed) != set(chosen):
                raise RuntimeError(f"Tier {tier} changed {changed}, expected {chosen}.")
            if spec["gamma_model"] != STANDARD_SPEC["gamma_model"]:
                raise RuntimeError("gamma_model must stay at the standard value.")
            seen.add(signature)
            made += 1
            sample_id = f"{tier:03d}-{made:03d}"
            samples.append({
                "id": sample_id,
                "tier": tier,
                "changed_count": len(chosen),
                "changed": chosen,
                "title": f"definition change {tier}%",
                "change": _change_text(tier, chosen, spec),
                "spec": spec,
            })
    return samples


def _slim(sample: dict, record: dict) -> dict:
    questions = []
    for item in record["questions"]:
        questions.append({
            "process": item["process"],
            "id": item["id"],
            "local_id": item["local_id"],
            "options": item["options"],
            "format_code": item["format_code"],
            "label": item["label"],
            "answer": item["answer"],
        })
    return {
        "id": sample["id"],
        "tier": sample["tier"],
        "changed_count": sample["changed_count"],
        "changed": sample["changed"],
        "title": sample["title"],
        "change": sample["change"],
        "spec": sample["spec"],
        "declaration": record["declaration"],
        "questions": questions,
    }


def _pilot(samples: list[dict]) -> list[str]:
    rng = random.Random(PILOT_SEED)
    picked = []
    for tier in TIERS:
        ids = [sample["id"] for sample in samples if sample["tier"] == tier]
        picked.extend(rng.sample(ids, PILOT_COUNTS[tier]))
    return picked


def _standard_sample() -> dict:
    return {
        "id": "000-001",
        "tier": 0,
        "changed_count": 0,
        "changed": [],
        "title": "common default",
        "change": "Common default. None of the 16 variable definitions is changed.",
        "spec": deepcopy(STANDARD_SPEC),
    }


def _index(pilot: list[str]) -> str:
    lines = [
        "# One common-default row and 100 definition-change rows",
        "",
        "000-001 is the common default, tier 0. Against it, the rows are split by how many definitions change, four tiers of 25. The 16 variable definitions are: theoretical angle of attack, theoretical sideslip, theoretical roll, mechanism angle of attack, mechanism sideslip, mechanism roll, installation, elastic angle of attack, elastic sideslip, elastic roll, flow-offset angle α_V, flow-offset angle β_V, balance installation angle, and the body-axis frame, the wind frame, and the balance frame.",
        "",
        "A frame counts as changed when it is not the legal axis set of the common default. Each frame has 48 legal declarations, and a change is drawn from the other 47. The model installation angle γ_model enters no blank and stays at the default in every tier. Euler order is fixed by the angle mechanism and is not a variable definition.",
        "",
        "| Tier | Changed | Rows | Ids |",
        "| --- | --- | --- | --- |",
        "| 0% | 0/16 | 1 | 000-001 |",
        "| 25% | 4/16 | 25 | 025-001 through 025-025 |",
        "| 50% | 8/16 | 25 | 050-001 through 050-025 |",
        "| 75% | 12/16 | 25 | 075-001 through 075-025 |",
        "| 100% | 16/16 | 25 | 100-001 through 100-025 |",
        "",
        "The sample seed is 20261102. The change combinations in the 25%, 50%, and 75% tiers do not repeat; 100% has only one combination, and its 25 rows are distinguished by different declarations of the three frames. A changed frame is drawn from the 47 non-default declarations. The rows are in samples.jsonl, one per line, with the declaration and the gold answers of "
        + str(BLANK_COUNT)
        + " blanks. Removed blank numbers are not shifted. cloze_eval reads this whole file by default, including 000-001.",
        "",
        "pilot_10.json is only an optional list of 10 ids, not the entry point. The four tiers are drawn 3, 3, 2, 2: " + ", ".join(pilot) + ".",
        "",
    ]
    return "\n".join(lines)


def write_standard_only() -> None:
    """Insert 000-001 without rebuilding the 100 perturbed rows."""
    sample = _standard_sample()
    record = build_record(sample)
    if len(record["questions"]) != BLANK_COUNT:
        raise RuntimeError(f"000-001 has {len(record['questions'])} questions, expected {BLANK_COUNT}.")
    path = OUT_DIR / "samples.jsonl"
    existing = path.read_text(encoding="utf-8")
    first = existing.split("\n", 1)[0]
    row = json.dumps(_slim(sample, record), ensure_ascii=False)
    if '"id": "000-001"' not in first:
        path.write_text(row + "\n" + existing, encoding="utf-8")
    (OUT_DIR / "00_index.md").write_text(_index(_pilot(_samples())), encoding="utf-8")
    print(f"standard {sample['id']} questions {len(record['questions'])}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    samples = [_standard_sample()] + _samples()
    path = OUT_DIR / "samples.jsonl"
    partial = OUT_DIR / "samples.jsonl.partial"
    with partial.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            record = build_record(sample)
            if len(record["questions"]) != BLANK_COUNT:
                raise RuntimeError(
                    f"{sample['id']} has {len(record['questions'])} questions, expected {BLANK_COUNT}."
                )
            handle.write(json.dumps(_slim(sample, record), ensure_ascii=False) + "\n")
            if index == 1 or index % 10 == 0:
                print(f"wrote {index}/{len(samples)}", flush=True)
    partial.replace(path)
    pilot = _pilot(samples)
    (OUT_DIR / "pilot_10.json").write_text(
        json.dumps({"ids": pilot, "counts": PILOT_COUNTS, "seed": PILOT_SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "00_index.md").write_text(_index(pilot), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUT_DIR}")
    print("Pilot: " + ", ".join(pilot))


if __name__ == "__main__":
    main()
