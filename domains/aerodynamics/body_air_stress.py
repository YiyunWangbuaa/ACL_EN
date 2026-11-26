# -*- coding: utf-8 -*-
"""Strong perturbations of the body frame to the airflow frame.

Both airflow-angle signs leave the standard value. The body frame, the
stability frame, and the airflow frame are all outside the standard
declaration. Six layouts share that reversal and differ in which legal axes
are declared.

Run::

    python -m domains.aerodynamics.body_air_stress
"""

from __future__ import annotations

import json
from copy import deepcopy

from domain_layout import process_dirs
from domains.aerodynamics.body_air_dataset import _render_markdown, build_record
from domains.aerodynamics.body_air_generate import legal_stability_frames
from domains.aerodynamics.ground_body_generate import legal_body_frames
from domains.aerodynamics.ground_track_air_generate import legal_airflow_frames
from flight_frame_model import ALPHA, BETA, STANDARD_SPEC


OUT_DIR = process_dirs("aerodynamics", "body_to_airflow")["stress"]


def _samples() -> list[dict]:
    bodies = legal_body_frames()
    stables = legal_stability_frames()
    airs = legal_airflow_frames()
    body_pick = [bodies[index] for index in (7, 15, 23, 31, 39, 47)]
    stable_pick = [stables[index] for index in (3, 11, 19, 27, 35, 43)]
    air_pick = [airs[index] for index in (4, 12, 20, 28, 36, 44)]
    samples = []
    for index, (body, stability, airflow) in enumerate(zip(body_pick, stable_pick, air_pick), start=1):
        spec = deepcopy(STANDARD_SPEC)
        spec["body_frame"] = deepcopy(body)
        spec["stability_frame"] = deepcopy(stability)
        spec["airflow_frame"] = deepcopy(airflow)
        spec[ALPHA] = {"sign": "negative"}
        spec[BETA] = {"sign": "negative"}
        samples.append({
            "id": f"{index:02d}",
            "title": f"full-factor departure {index:02d}",
            "change": "The signs of angle of attack and sideslip both leave the standard definition, and the airframe, the stability frame, and the airflow frame are each replaced by another legal set of axes.",
            "spec": spec,
        })
    return samples


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    standard = build_record({
        "id": "00",
        "title": "common default",
        "change": "",
        "spec": deepcopy(STANDARD_SPEC),
    })
    standard_answers = {item["id"]: item["answer"] for item in standard["questions"]}
    records = []
    lines = [
        "# Flight-mechanics stress samples",
        "",
        "Subprocess: body to airflow. Each sample changes angle of attack, sideslip, the airframe, the stability frame, and the airflow frame together. The ground frame, the track frame, and the attitude angles and track angles stay on the current standard definition, because the blanks of this subprocess do not read them.",
        "",
        "| id | blanks that differ from the standard |",
        "| --- | --- |",
    ]
    for sample in samples:
        record = build_record(sample)
        answers = {item["id"]: item["answer"] for item in record["questions"]}
        differ = sum(answers[key] != value for key, value in standard_answers.items())
        lines.append(f"| {sample['id']} | {differ}/35 |")
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(_render_markdown(record), encoding="utf-8")
        records.append(record)
        print(f"sample {sample['id']}: {differ}/35 answers differ from the standard")
    lines.append("")
    (OUT_DIR / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} stress samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
