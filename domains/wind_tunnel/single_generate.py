# -*- coding: utf-8 -*-
"""Main-experiment wind-tunnel set: one standard row and nine single-factor rows.

The 100-row batch is left in place. This file writes only data/wind_tunnel/single.
Each perturbed row changes exactly one of the first nine scalar definitions and
records that definition in factor.

Run::

    python -m domains.wind_tunnel.single_generate
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from domain_layout import DOMAINS
from domains.wind_tunnel.batch_generate import (
    BLANK_COUNT,
    SCALAR_FLIP,
    SCALAR_LABEL,
    SCALAR_ZH,
    _slim,
    _standard_sample,
)
from domains.wind_tunnel.dataset_generate import build_record
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

OUT = Path(DOMAINS["wind_tunnel"]["root"]) / "single"
KEYS = (
    "alpha",
    "beta",
    "gamma",
    "alpha_m",
    "beta_m",
    "gamma_m",
    "placement",
    "alpha_e",
    "beta_e",
)


def _answers(row: dict) -> dict[str, str]:
    return {str(item["id"]): str(item["answer"]) for item in row["questions"]}


def _row(sample: dict) -> dict:
    record = build_record(sample)
    if len(record["questions"]) != BLANK_COUNT:
        raise RuntimeError(f"{sample['id']} has {len(record['questions'])} blanks; expected {BLANK_COUNT}.")
    row = _slim(sample, record)
    row["factor"] = sample["factor"]
    return row


def main() -> None:
    if len(KEYS) != 9 or len(set(KEYS)) != 9:
        raise RuntimeError("The wind-tunnel main experiment needs 9 distinct single definitions.")
    standard = _standard_sample()
    standard["factor"] = "none"
    standard["change"] = "Single-definition perturbation: none. Common default. None of the 16 variable definitions is changed."
    rows = [_row(standard)]
    base = _answers(rows[0])
    for index, key in enumerate(KEYS, start=1):
        spec = deepcopy(STANDARD_SPEC)
        spec[key] = deepcopy(SCALAR_FLIP[key])
        sample = {
            "id": f"S{index:02d}",
            "tier": 0,
            "changed_count": 1,
            "changed": [key],
            "factor": key,
            "title": SCALAR_LABEL[key],
            "change": (
                f"Single-definition perturbation: {SCALAR_LABEL[key]}. "
                f"{SCALAR_ZH[key]}. The other 15 definitions stay at the common default."
            ),
            "spec": spec,
        }
        row = _row(sample)
        if row["changed"] != [key]:
            raise RuntimeError(f"The changed list of {key} is {row['changed']}.")
        if _answers(row) == base:
            raise RuntimeError(f"{key} does not change any standard answer.")
        rows.append(row)
        print(f"{row['id']} {row['factor']} blanks {len(row['questions'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    target = OUT / "samples.jsonl"
    partial = OUT / "samples.jsonl.partial"
    partial.write_text(payload, encoding="utf-8")
    partial.replace(target)
    lines = [
        "# Wind-tunnel main experiment",
        "",
        "One common-default row and nine single-definition perturbations. Each perturbed row changes exactly one of the 16 variable definitions, and factor records that definition.",
        "The 100-row tiered set remains in data/wind_tunnel/batch and is not the entry for this main experiment.",
        "",
        f"- 000-001: single-definition perturbation: none. {BLANK_COUNT} blanks.",
    ]
    for row in rows[1:]:
        lines.append(f"- {row['id']}: single-definition perturbation: {row['title']}. factor `{row['factor']}`.")
    lines.append("")
    (OUT / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
