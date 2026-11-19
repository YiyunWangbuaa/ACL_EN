# -*- coding: utf-8 -*-
"""One standard flight sample plus the first sample of each definition tier."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from domains.aerodynamics import body_air_dataset, dataset_generate, ground_track_air_dataset
from domains.aerodynamics.pilot_generate import _combine, _slim
from domain_layout import SETS
from flight_frame_model import STANDARD_SPEC

ROOT = Path(__file__).resolve().parent
PILOT = SETS["basic.flight"]["pilot"] / "samples.jsonl"
WANTED = ("025-001", "050-001", "075-001", "100-001")


def main() -> None:
    picked = {}
    for line in PILOT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in WANTED:
            picked[row["id"]] = row
    missing = [item for item in WANTED if item not in picked]
    if missing:
        raise SystemExit("缺少样本：" + "、".join(missing))
    standard = {
        "id": "000-001",
        "tier": 0,
        "changed_count": 0,
        "changed": [],
        "title": "统一默认",
        "change": "定义变更档0%：13项可变定义均保持标准定义，八个角度符号和五个坐标系都不改。",
        "spec": deepcopy(STANDARD_SPEC),
    }
    declaration, questions, _sections = _combine(standard)
    rows = [_slim(standard, declaration, questions)]
    rows.extend(picked[item] for item in WANTED)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (ROOT / "ids.json").write_text(
        json.dumps({"ids": [row["id"] for row in rows]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sentence_count = (
        len(dataset_generate._sentence_groups())
        + len(ground_track_air_dataset._sentence_groups())
        + len(body_air_dataset._sentence_groups())
    )
    print(f"sentences_per_sample={sentence_count}")
    for row in rows:
        print(f"{row['id']} tier={row['tier']} changed={row['changed_count']} blanks={len(row['questions'])}")


if __name__ == "__main__":
    main()
