# -*- coding: utf-8 -*-
"""Strong perturbations of ground frame to track frame and airflow frame.

Every track-angle sign leaves the standard value. The ground frame, the track
frame, and the airflow frame are all outside the standard declaration. Six
layouts share that reversal and differ in which legal axes are declared.

Run::

    python -m domains.aerodynamics.ground_track_air_stress
"""

from __future__ import annotations

import json
from copy import deepcopy

from domain_layout import process_dirs
from domains.aerodynamics.ground_body_generate import legal_ground_frames
from domains.aerodynamics.ground_track_air_dataset import PROCESS, _render_markdown, build_record
from domains.aerodynamics.ground_track_air_generate import legal_airflow_frames, legal_track_frames
from flight_frame_model import GAMMA, MU, STANDARD_SPEC, VARPHI


OUT_DIR = process_dirs("aerodynamics", "ground_to_track_air")["stress"]


def _samples() -> list[dict]:
    grounds = [item for item in legal_ground_frames() if item != STANDARD_SPEC["ground_frame"]]
    tracks = legal_track_frames()
    airs = legal_airflow_frames()
    ground_pick = [grounds[index] for index in (0, 2, 5, 8, 11, 14)]
    track_pick = [tracks[index] for index in (7, 15, 23, 31, 39, 47)]
    air_pick = [airs[index] for index in (4, 12, 20, 28, 36, 44)]
    samples = []
    for index, (ground, track, airflow) in enumerate(zip(ground_pick, track_pick, air_pick), start=1):
        spec = deepcopy(STANDARD_SPEC)
        spec["ground_frame"] = deepcopy(ground)
        spec["track_frame"] = deepcopy(track)
        spec["airflow_frame"] = deepcopy(airflow)
        spec[VARPHI] = {"sign": "negative"}
        spec[MU] = {"sign": "negative"}
        spec[GAMMA] = {"sign": "negative"}
        samples.append({
            "id": f"{index:02d}",
            "title": f"full-factor departure {index:02d}",
            "change": "The signs of track azimuth, track inclination, and track roll all leave the standard definition, and the ground frame, the track frame, and the airflow frame are each replaced by another legal set of axes.",
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
        "Subprocess: ground to track and airflow. Each sample changes the three track-angle signs, the ground frame, the track frame, and the airflow frame together. The airframe, the stability frame, and the attitude angles, angle of attack, and sideslip stay on the current standard definition, because the blanks of this subprocess do not read them.",
        "",
        "| id | blanks that differ from the standard |",
        "| --- | --- |",
    ]
    for sample in samples:
        record = build_record(sample)
        answers = {item["id"]: item["answer"] for item in record["questions"]}
        differ = sum(answers[key] != value for key, value in standard_answers.items())
        lines.append(f"| {sample['id']} | {differ}/41 |")
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(_render_markdown(record), encoding="utf-8")
        records.append(record)
        print(f"sample {sample['id']}: {differ}/41 answers differ from the standard")
    lines.append("")
    (OUT_DIR / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} stress samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
