# -*- coding: utf-8 -*-
"""Strong perturbations of the ground-to-body subprocess.

Every attitude sign leaves the standard value. The ground frame and the body
frame are both outside the standard declaration. Six layouts share that reversal
and differ in which legal axes are declared.

Run::

    python -m domains.aerodynamics.stress_generate
"""

from __future__ import annotations

import json
from copy import deepcopy

from domain_layout import process_dirs
from flight_frame_model import PHI, PSI, STANDARD_SPEC, THETA
from domains.aerodynamics.dataset_generate import PROCESS, _render_markdown, build_record
from domains.aerodynamics.ground_body_generate import BLANK_COUNT, legal_body_frames, legal_ground_frames


OUT_DIR = process_dirs("aerodynamics", "ground_to_body")["stress"]


def _samples() -> list[dict]:
    grounds = [item for item in legal_ground_frames() if item != STANDARD_SPEC["ground_frame"]]
    bodies = legal_body_frames()
    # Spread across chirality and axis role instead of taking the first six duplicates.
    ground_pick = [grounds[index] for index in (0, 2, 5, 8, 11, 14)]
    body_pick = [bodies[index] for index in (7, 15, 23, 31, 39, 47)]
    samples = []
    for index, (ground, body) in enumerate(zip(ground_pick, body_pick), start=1):
        spec = deepcopy(STANDARD_SPEC)
        spec["ground_frame"] = deepcopy(ground)
        spec["body_frame"] = deepcopy(body)
        spec[PSI] = {"sign": "negative"}
        spec[THETA] = {"sign": "negative"}
        spec[PHI] = {"sign": "negative"}
        samples.append({
            "id": f"{index:02d}",
            "title": f"全因子偏离 {index:02d}",
            "change": "偏航、俯仰、滚转三个符号都离开默认，地面系和机体系都换成另一套合法坐标轴。",
            "spec": spec,
        })
    return samples


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    standard = build_record({
        "id": "00",
        "title": "统一默认",
        "change": "",
        "spec": deepcopy(STANDARD_SPEC),
    })
    standard_answers = {item["id"]: item["answer"] for item in standard["questions"]}
    records = []
    lines = [
        "# 空气动力学强扰动样本",
        "",
        "子过程：地面系到机体系。每条同时改掉三个姿态角符号、地面系和机体系。气流系、稳定性坐标系、航迹系以及另外五个角度保持当前默认，因为这一子过程的空不读取它们。",
        "",
        "| 编号 | 与默认不同的空 |",
        "| --- | --- |",
    ]
    for sample in samples:
        record = build_record(sample)
        answers = {item["id"]: item["answer"] for item in record["questions"]}
        differ = sum(answers[key] != value for key, value in standard_answers.items())
        lines.append(f"| {sample['id']} | {differ}/{BLANK_COUNT} |")
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(_render_markdown(record), encoding="utf-8")
        records.append(record)
        print(f"sample {sample['id']}: {differ}/{BLANK_COUNT} answers differ from the standard")
    lines.append("")
    (OUT_DIR / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} stress samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
