# -*- coding: utf-8 -*-
"""Debug why sample 025-001 gives a negative alpha rotation."""
import json
import sys

ROOT = r"C:\Users\minec\Desktop\ACL\workflow"
sys.path.insert(0, ROOT)

from flight_frame_model import ALPHA, BETA, FlightFrameModel  # noqa: E402

PATH = ROOT + r"\data\flight\basic\pilot\samples.jsonl"
rows = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]
row = next(r for r in rows if r["id"] == "025-001")
spec = row["spec"]

model = FlightFrameModel().declare(json.loads(json.dumps(spec)))
print("alpha spec:", spec["alpha"], "beta:", spec["beta"])
for owner in ("body", "stability"):
    print(owner, {n: (a.absolute_axis, a.points_to) for n, a in model.frames[owner].items()})
rec = model.binding(ALPHA)
print("binding(alpha):", rec.frame, rec.associated_axis, rec.absolute_axis, rec.rotation_when_positive)
for item in model.conversion("body", "stability"):
    print("leg:", item.angle, item.frame, item.absolute_axis, item.axis, item.rotation_when_positive)
