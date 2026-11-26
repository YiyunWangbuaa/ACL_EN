# -*- coding: utf-8 -*-
"""Generate 10 samples per tier for the flight-mechanics control."""
from __future__ import annotations

import json

from domain_layout import SETS
from domains.flight_ctrl import fm_ctrl_generate


def main() -> None:
    out = SETS["reconstruction.flight"]["samples"]
    rows = []
    for tier in (25, 50, 75, 100):
        for index in range(1, 11):
            sample = fm_ctrl_generate.build(tier)
            sample["id"] = f"fmctrl-{tier:03d}-{index:03d}"
            sample["title"] = (
                "Flight-mechanics control (mechanism angles to actual angle of attack and sideslip) "
                f"{tier}% tier sample {index}"
            )
            sample["group"] = "control"
            rows.append(sample)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples -> {out}")


if __name__ == "__main__":
    main()
