# -*- coding: utf-8 -*-
"""Ten computer-vision samples in the same four tiers as the wind-tunnel pilot.

Ten definitions enter the cloze: four frames and six angle directions. A tier
of k percent changes exactly 10*k//100 of them. Composition order is fixed
and is not a definition. Each tier's samples use different definitions.

Run::

    python -m domains.computer_vision.batch_generate
"""
from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import SETS
from domains.computer_vision import cv_generate as cv
from domains.computer_vision.cv_generate import (
    ANGLE_FLIP,
    ANGLE_ZH,
    FACTORS,
    FRAME_NAMES,
    FRAME_TITLE,
    STANDARD_SPEC,
    alternate_frames,
)

SEED = 20261103
TIERS = (25, 50, 75, 100)
TIER_COUNTS = {25: 3, 50: 3, 75: 2, 100: 2}
ROLE_EN = {"axial": "axial", "normal": "normal", "lateral": "lateral"}


def _width(tier: int) -> int:
    return len(FACTORS) * tier // 100


def _changed(spec: dict) -> list[str]:
    found = []
    for key in FACTORS:
        if key in ANGLE_FLIP:
            if spec["angles"][key] != STANDARD_SPEC["angles"][key]:
                found.append(key)
        elif spec[key] != STANDARD_SPEC[key]:
            found.append(key)
    return found


def _frame_phrase(key: str, spec: dict) -> str:
    names = FRAME_NAMES[key]
    axes = cv.cm.build_frame(key, names, spec)
    parts = []
    for name in names:
        vector = axes[name]
        role = cv.cm.role_of(vector)
        pointing = cv.DIR_NAME[vector]
        parts.append(f"{name} is {ROLE_EN[role]} and points {pointing}")
    hand = "left-handed" if spec["chirality"] == "left" else "right-handed"
    body = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"The {FRAME_TITLE[key]} is {hand}: {body}."


def _change_text(tier: int, changed: list[str], spec: dict) -> str:
    phrases = []
    for key in changed:
        if key in ANGLE_ZH:
            phrases.append(ANGLE_ZH[key])
        else:
            phrases.append(_frame_phrase(key, spec[key]))
    cleaned = [phrase.rstrip(".") for phrase in phrases]
    return (
        f"Definition change tier {tier}%: {len(changed)} of 10 variable definitions changed: "
        + "; ".join(cleaned)
        + "."
    )


def _samples() -> list[dict]:
    if len(FACTORS) != 10 or sum(TIER_COUNTS.values()) != 10:
        raise RuntimeError("The four tiers must cover 10 definitions and 10 samples.")
    if [_width(tier) for tier in TIERS] != [2, 5, 7, 10]:
        raise RuntimeError("Tier widths drifted from 2, 5, 7, 10.")
    layouts = {key: alternate_frames(key) for key in FRAME_NAMES}
    rng = random.Random(SEED)
    seen = set()
    samples = []
    for tier in TIERS:
        width = _width(tier)
        pool = list(itertools.combinations(FACTORS, width))
        rng.shuffle(pool)
        made = 0
        cursor = 0
        while made < TIER_COUNTS[tier]:
            chosen = list(pool[cursor % len(pool)])
            cursor += 1
            spec = deepcopy(STANDARD_SPEC)
            for key in chosen:
                if key in ANGLE_FLIP:
                    spec["angles"][key] = ANGLE_FLIP[key]
                else:
                    spec[key] = deepcopy(rng.choice(layouts[key]))
            signature = json.dumps(spec, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                if cursor > len(pool) * 40:
                    raise RuntimeError(f"Could not draw a unique spec for tier {tier}.")
                continue
            changed = _changed(spec)
            if set(changed) != set(chosen):
                raise RuntimeError(f"Tier {tier} changed {changed}, expected {chosen}.")
            seen.add(signature)
            made += 1
            samples.append({
                "id": f"{tier:03d}-{made:03d}",
                "tier": tier,
                "changed_count": len(chosen),
                "changed": chosen,
                "title": f"stereo pose definition change {tier}% sample {made}",
                "change": _change_text(tier, chosen, spec),
                "spec": spec,
            })
    return samples


def _wind_signature(spec_wind: dict) -> tuple:
    axes = cv.cm.build_frame("wind", FRAME_NAMES["wind"], spec_wind)
    return tuple(
        (name, cv.cm.role_of(axes[name]), cv.DIR_NAME[axes[name]])
        for name in FRAME_NAMES["wind"]
    )


def _repair_wind_frame(spec_wind: dict, used: set[tuple]) -> dict:
    """Keep Xw on the forward axial axis. Reuse a free legal frame for Yw and Zw."""
    signature = _wind_signature(spec_wind)
    if signature[0][1:] == ("axial", "forward"):
        return spec_wind
    axes = cv.cm.build_frame("wind", FRAME_NAMES["wind"], spec_wind)
    old_dirs = {
        (cv.cm.role_of(axes[name]), cv.DIR_NAME[axes[name]])
        for name in FRAME_NAMES["wind"]
        if cv.cm.role_of(axes[name]) != "axial"
    }
    candidates = []
    for alt in alternate_frames("wind"):
        if alt["chirality"] != spec_wind["chirality"]:
            continue
        alt_signature = _wind_signature(alt)
        if alt_signature in used:
            continue
        prefer = (alt["Yw"]["along"], alt["Yw"]["points"]) in old_dirs
        candidates.append((0 if prefer else 1, alt_signature, alt))
    if not candidates:
        raise RuntimeError("No remaining world frame can separate the other two axes after Xw is fixed.")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def refresh_locked_wind() -> None:
    """Rewrite the stored pilot from its current specs, with Xw fixed forward.

    Other changed factors stay as they were. This does not draw a new sample.
    """
    path = SETS["cross.wind_vision"]["samples"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    used = set()
    for row in rows:
        signature = _wind_signature(row["spec"]["wind"])
        if signature[0][1:] == ("axial", "forward"):
            used.add(signature)
    for row in rows:
        row["spec"]["wind"] = _repair_wind_frame(row["spec"]["wind"], used)
        used.add(_wind_signature(row["spec"]["wind"]))
        changed = _changed(row["spec"])
        if set(changed) != set(row["changed"]):
            raise RuntimeError(f"{row['id']} changed factors moved from {row['changed']} to {changed}.")
        row["change"] = _change_text(row["tier"], row["changed"], row["spec"])
    declarations = []
    written = []
    for row in rows:
        record = cv.generate(row["spec"])
        if len(record["questions"]) != 165:
            raise RuntimeError(f"{row['id']} has {len(record['questions'])} questions.")
        declarations.append(record["declaration"])
        row["declaration"] = record["declaration"]
        row["questions"] = record["questions"]
        written.append(row)
        wind = _wind_signature(row["spec"]["wind"])
        print(f"refreshed {row['id']}: {wind}", flush=True)
    if len(set(declarations)) != len(written):
        raise RuntimeError("Two samples share one declaration.")
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written),
        encoding="utf-8",
    )
    print(f"wrote {path}")


def main() -> None:
    samples = _samples()
    declarations = []
    path = SETS["cross.wind_vision"]["samples"]
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = cv.generate(sample["spec"])
            if len(record["questions"]) != 165:
                raise RuntimeError(f"{sample['id']} has {len(record['questions'])} questions.")
            declarations.append(record["declaration"])
            row = {
                "id": sample["id"],
                "tier": sample["tier"],
                "changed_count": sample["changed_count"],
                "changed": sample["changed"],
                "title": sample["title"],
                "change": sample["change"],
                "spec": sample["spec"],
                "declaration": record["declaration"],
                "questions": record["questions"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"wrote {sample['id']}: {sample['changed_count']}/10 {sample['changed']}", flush=True)
    if len(set(declarations)) != len(samples):
        raise RuntimeError("Two samples share one declaration.")
    for tier in TIERS:
        group = [sample for sample in samples if sample["tier"] == tier]
        specs = {json.dumps(sample["spec"], ensure_ascii=False, sort_keys=True) for sample in group}
        if len(specs) != len(group):
            raise RuntimeError(f"Tier {tier} still has identical definitions.")
    ids = [sample["id"] for sample in samples]
    SETS["cross.wind_vision"]["pilot"].write_text(
        json.dumps({
            "ids": ids,
            "counts": TIER_COUNTS,
            "seed": SEED,
            "factors": len(FACTORS),
            "widths": {str(tier): _width(tier) for tier in TIERS},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("pilot " + ", ".join(ids))


if __name__ == "__main__":
    main()
