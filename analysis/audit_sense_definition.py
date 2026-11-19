# -*- coding: utf-8 -*-
"""Sense audit with a definition-based rule, independent of the model's kind flag.

expected sense = sign( cross(axis, moving) . toward ), toward flipped for a
negative declaration. The axis is the coordinate axis of the source frame that
lies on the angle's absolute direction.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

from flight_frame_model import (  # noqa: E402
    ALPHA, ANGLE_SPEC, BETA, GAMMA, MU, PHI, PSI, THETA, VARPHI, FlightFrameModel,
)
from domains.aerodynamics.body_air_generate import generate as gen_bta  # noqa: E402
from domains.aerodynamics.ground_body_generate import generate as gen_gtb  # noqa: E402
from domains.aerodynamics.ground_track_air_generate import generate as gen_gta  # noqa: E402

TURN = {
    "ground_to_body": (gen_gtb, {"13": PSI, "17": THETA, "21": PHI}),
    "ground_to_track_air": (gen_gta, {"17": VARPHI, "21": MU, "33": GAMMA}),
    "body_to_airflow": (gen_bta, {"16": ALPHA, "27": BETA}),
}
PATHS = {
    "ground_to_body": "data/flight/basic/ground_to_body/batch/samples.jsonl",
    "ground_to_track_air": "data/flight/basic/ground_to_track_air/batch/samples.jsonl",
    "body_to_airflow": "data/flight/basic/body_to_airflow/batch/samples.jsonl",
}


def sense(model, angle):
    spec = ANGLE_SPEC[angle]
    frame = spec.pair[0]
    record = model.binding(angle)
    axis = model.frames[frame][record.associated_axis]
    axis_vector = model.geometry[frame].vector[axis.points_to]
    moving = model._vector(spec.moving)
    toward = model._vector(spec.toward)
    if model._sign[angle] == "negative":
        toward = tuple(-x for x in toward)
    value = sum(a * b for a, b in zip(
        (axis_vector[1] * moving[2] - axis_vector[2] * moving[1],
         axis_vector[2] * moving[0] - axis_vector[0] * moving[2],
         axis_vector[0] * moving[1] - axis_vector[1] * moving[0]),
        toward))
    return "正向" if value > 0 else "逆向"


def options_text(item):
    letters = "ABCDEFGHIJ"
    return dict(zip(letters, item["options"]))


for name, (gen, blanks) in TURN.items():
    rows = [json.loads(line) for line in
            (ROOT / PATHS[name]).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = []
    for row in rows:
        spec = json.loads(json.dumps(row["spec"]))
        model = FlightFrameModel().declare(spec)
        payload = {item["manuscript_id"]: item for item in gen(spec)["items"]}
        for blank, angle in blanks.items():
            item = payload[blank]
            got = str(item["answer"])
            want = sense(model, angle)
            if got != want:
                bad.append((row["id"], blank, angle, got, want))
    print(f"{name}: {len(bad)} mismatches of {len(rows)} samples")
    for item in bad[:8]:
        print("   ", item)
