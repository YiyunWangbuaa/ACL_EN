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
        raise RuntimeError(f"{sample['id']} 有 {len(record['questions'])} 个空，应为 {BLANK_COUNT}。")
    row = _slim(sample, record)
    row["factor"] = sample["factor"]
    return row


def main() -> None:
    if len(KEYS) != 9 or len(set(KEYS)) != 9:
        raise RuntimeError("风洞主实验要 9 个互不相同的单定义。")
    standard = _standard_sample()
    standard["factor"] = "无"
    standard["change"] = "单定义扰动：无。统一默认。16 项可变定义都不改。"
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
            "title": SCALAR_ZH[key].split("改为", 1)[0],
            "change": f"单定义扰动：{SCALAR_ZH[key].split('改为', 1)[0]}。{SCALAR_ZH[key]}。其余 15 项保持统一默认。",
            "spec": spec,
        }
        row = _row(sample)
        if row["changed"] != [key]:
            raise RuntimeError(f"{key} 的变更列表是 {row['changed']}。")
        if _answers(row) == base:
            raise RuntimeError(f"{key} 没有改变任何标准答案。")
        rows.append(row)
        print(f"{row['id']} {row['factor']} blanks {len(row['questions'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    target = OUT / "samples.jsonl"
    partial = OUT / "samples.jsonl.partial"
    partial.write_text(payload, encoding="utf-8")
    partial.replace(target)
    lines = [
        "# 风洞主实验",
        "",
        "1 条统一默认加 9 条单定义扰动。每条只改 16 项可变定义中的一项，factor 记录这一项。",
        "100 条分档数据仍在 data/wind_tunnel/batch，不作为这次主实验的入口。",
        "",
        f"- 000-001：单定义扰动：无。{BLANK_COUNT} 个空。",
    ]
    for row in rows[1:]:
        lines.append(f"- {row['id']}：单定义扰动：{row['title']}。factor `{row['factor']}`。")
    lines.append("")
    (OUT / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
