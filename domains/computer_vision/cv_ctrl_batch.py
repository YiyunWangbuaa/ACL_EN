# -*- coding: utf-8 -*-
"""Generate 10 samples per tier for CV-CTRL (对照组)."""
from __future__ import annotations

import json
from pathlib import Path

from domains.computer_vision import cv_ctrl_generate

HERE = Path(__file__).resolve().parent


def main() -> None:
    out = HERE / "cv_ctrl_samples.jsonl"
    rows = []
    for tier in (25, 50, 75, 100):
        for index in range(1, 11):
            sample = cv_ctrl_generate.build(tier)
            sample["id"] = f"cvctrl-{tier:03d}-{index:03d}"
            sample["title"] = f"机器视觉对照（相机↔世界系）{tier}% 档 样本{index}"
            sample["group"] = "control"
            rows.append(sample)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
