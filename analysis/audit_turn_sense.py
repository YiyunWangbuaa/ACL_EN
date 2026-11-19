# -*- coding: utf-8 -*-
"""Independent sense audit of every 正向/逆向 blank in the aerodynamics data.

The expected sense follows straight from the definition: a positive angle moves
the named vector toward the declared side, so the sign is the dot product of
(cross(axis, moving), toward). The axis is the source frame's coordinate axis on
the angle's absolute direction.
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

from flight_frame_model import ANGLE_SPEC, FlightFrameModel  # noqa: E402

GEOM = {"forward": (1, 0, 0), "rearward": (-1, 0, 0), "right": (0, 1, 0),
        "left": (0, -1, 0), "down": (0, 0, 1), "up": (0, 0, -1)}
NORMAL = {"ground": "up", "body": "down", "airflow": "down", "stability": "down", "track": "down"}
AXES = {"ground": ("Xg", "Yg", "Zg"), "body": ("Xb", "Yb", "Zb"),
        "airflow": ("Xa", "Ya", "Za"), "stability": ("Xs", "Ys", "Zs"),
        "track": ("Xk", "Yk", "Zk")}
TURNS = {
    "ground_to_body": {"13": ("psi", "ground"), "17": ("theta", "ground"), "21": ("phi", "ground")},
    "ground_to_track_air": {"17": ("varphi", "ground"), "21": ("mu", "ground"), "33": ("gamma", "track")},
    "body_to_airflow": {"16": ("alpha", "body"), "27": ("beta", "stability")},
}


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def neg(v):
    return tuple(-x for x in v)


def build_frame(owner, spec):
    normal = GEOM[NORMAL[owner]]
    axial = GEOM["forward"]
    lateral = cross(normal, axial)
    vectors = {f"{owner}_forward": axial, f"{owner}_rearward": neg(axial),
               f"{owner}_up": GEOM["up"], f"{owner}_down": GEOM["down"],
               f"{owner}_right": lateral, f"{owner}_left": neg(lateral)}
    names = AXES[owner]
    out = {name: vectors[spec[name]["points"]] for name in names if name in spec}
    if owner == "ground" and "Xg" not in out:
        out["Xg"] = GEOM["forward"]
    missing = [n for n in names if n not in out]
    if len(missing) == 1:
        index = names.index(missing[0])
        vector = cross(out[names[(index + 1) % 3]], out[names[(index + 2) % 3]])
        if spec["chirality"] == "left":
            vector = neg(vector)
        out[missing[0]] = vector
    return out


def expected_sense(sample, angle, frame):
    spec = sample["spec"]
    model = FlightFrameModel().declare(json.loads(json.dumps(spec)))
    record = model.binding(angle)
    axis = build_frame(frame, spec[f"{frame}_frame"])[record.associated_axis]
    angle_spec = ANGLE_SPEC[angle]
    sign = spec[angle]["sign"]
    toward = angle_spec.toward if sign == "positive" else None
    if toward is None:
        toward = model._vector(angle_spec.toward)
        toward = neg(toward)
    else:
        toward = model._vector(angle_spec.toward)
    moving = model._vector(angle_spec.moving)
    value = sum(a * b for a, b in zip(cross(axis, moving), toward))
    return "正向" if value > 0 else "逆向"


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


for name, blanks in TURNS.items():
    path = ROOT / "data" / "flight" / "basic" / name / "batch" / "samples.jsonl"
    rows = load(path)
    bad = []
    for row in rows:
        answers = {q["id"]: q["answer"] for q in row["questions"]}
        options = {q["id"]: q.get("options") or [] for q in row["questions"]}
        for blank, (angle, frame) in blanks.items():
            want = expected_sense(row, angle, frame)
            answer = answers[blank]
            got = options[blank][ord(answer) - ord("A")] if options[blank] else answer
            if got != want:
                bad.append((row["id"], blank, angle, got, want))
    print(f"{name}: {len(bad)} mismatches")
    for item in bad[:10]:
        print("   ", item)
