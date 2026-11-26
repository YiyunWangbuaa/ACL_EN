# -*- coding: utf-8 -*-
"""Main-experiment vision set: one standard long manuscript and nine single factors.

Theory is not included. Each file is one Object and stereo injection of 165 blanks.
The two-stage review in data/vision/review/scene is left unchanged.

Run::

    python -m domains.computer_vision.single_generate
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from domain_layout import DATA
from domains.computer_vision.batch_generate import _frame_phrase
from domains.computer_vision.cv_generate import (
    ANGLE_FLIP,
    ANGLE_ZH,
    STANDARD_SPEC,
    alternate_frames,
    build,
)
from domains.computer_vision.cv_vision_context import apply_record, leftover
from domains.computer_vision.en_surface import english_text
from domains.computer_vision.review_export import _declaration, _process

OUT = DATA / "vision" / "single"
# Nine definitions. Mechanism roll stays at the standard value in every row.
FACTORS = (
    ("wind", "world frame"),
    ("body", "object frame"),
    ("camera1", "camera-1 frame"),
    ("camera2", "camera-2 frame"),
    ("pan", "camera azimuth"),
    ("tilt", "camera pitch"),
    ("roll", "camera roll"),
    ("attack", "mechanism pitch"),
    ("sideslip", "mechanism azimuth"),
)
FACTOR_EN = {name: name for _key, name in FACTORS}
FACTOR_EN["none"] = "none"


def _spec(key: str) -> dict:
    spec = deepcopy(STANDARD_SPEC)
    if key in ANGLE_FLIP:
        spec["angles"][key] = ANGLE_FLIP[key]
        return spec
    for layout in alternate_frames(key):
        spec[key] = deepcopy(layout)
        return spec
    raise RuntimeError(f"{key} has no non-default declaration.")


def _phrase(key: str, spec: dict) -> str:
    if key in ANGLE_ZH:
        return ANGLE_ZH[key]
    return _frame_phrase(key, spec[key])


def _answers(record: dict) -> list[str]:
    return [str(item["answer"]) for item in record["questions"]]


def _markdown(index: int, factor: str, row: dict) -> str:
    titles = {
        "frame": "Absolute directions and coordinate axes",
        "camera_attitude": "Camera attitude",
        "body_attitude": "Object attitude",
        "projection": "Projection of the object nose",
        "axis_wind": "Direction of an image line",
        "extrinsic": "Stereo extrinsics",
        "epipolar": "Epipolar geometry and triangulation",
    }
    lines = [
        f"# Data {index:02d}",
        f"Single-definition perturbation: {factor}",
        "",
        "This is the long manuscript of the vision main experiment. It contains only the object-and-stereo task and is injected once. It does not include the basic-theory stage.",
        "",
        row["change"],
        "",
        "# Object and stereo",
        "",
        "## Declaration",
        "",
        _declaration(row["declaration"]),
        "",
        "## Full process",
        "",
        _process(row["questions"], titles),
    ]
    return "\n".join(lines)


def _check_words(row: dict) -> None:
    chunks = [row["declaration"], row["change"]]
    for item in row["questions"]:
        chunks.append(str(item.get("prompt") or ""))
        chunks.append(str(item.get("answer") or ""))
        chunks.extend(str(option) for option in item.get("options") or [])
    found = leftover("\n".join(chunks))
    if found:
        raise RuntimeError("wind-tunnel wording remains after rewrite: " + ", ".join(found))


def main() -> None:
    if len(FACTORS) != 9 or len({key for key, _name in FACTORS}) != 9:
        raise RuntimeError("The vision main experiment needs 9 distinct single definitions.")
    print("build standard", flush=True)
    built = build(STANDARD_SPEC)
    standard = apply_record({
        "id": "01",
        "title": "shared default",
        "change": "Single-definition perturbation: none. None of the 10 variable definitions is changed.",
        "declaration": built["declaration"],
        "questions": built["questions"],
    })
    if len(standard["questions"]) != 165:
        raise RuntimeError(f"the standard manuscript has {len(standard['questions'])} blanks, expected 165.")
    _check_words(standard)
    base = _answers(standard)
    rows = [("01", "none", standard)]
    for index, (key, name) in enumerate(FACTORS, start=2):
        print(f"build {key}", flush=True)
        spec = _spec(key)
        record = build(spec)
        if len(record["questions"]) != 165:
            raise RuntimeError(f"{key} has {len(record['questions'])} blanks.")
        if _answers(record) == base:
            raise RuntimeError(f"{key} did not change any gold answer.")
        phrase = _phrase(key, spec)
        if not phrase.endswith("."):
            phrase += "."
        row = apply_record({
            "id": f"{index:02d}",
            "title": name,
            "change": f"Single-definition perturbation: {name}. {phrase}",
            "declaration": record["declaration"],
            "questions": record["questions"],
        })
        _check_words(row)
        rows.append((f"{index:02d}", name, row))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id, factor, row in rows:
        text = _markdown(int(sample_id), factor, row)
        if text.count("{ID:") != 165:
            raise RuntimeError(f"{sample_id} does not have 165 slots.")
        (OUT / f"sample_{sample_id}.md").write_text(english_text(text), encoding="utf-8")
        manifest.append({
            "id": sample_id,
            "factor": FACTOR_EN[factor],
            "change": english_text(row["change"]),
        })
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    lines = [
        "# Vision main experiment",
        "",
        "One standard definition and nine single-definition perturbations. Each item is one object-and-stereo manuscript, 165 blanks, injected once.",
        "The basic-theory stage is not included. The older two-stage manuscripts in data/vision/review/scene stay on disk and are not the entry for this experiment.",
        "The single-definition perturbation is written at the start of each manuscript and in factors.json.",
        "",
    ]
    for item in manifest:
        lines.append(f"- Data {item['id']}: single-definition perturbation: {item['factor']}.")
    lines.append("")
    (OUT / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "factors.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
