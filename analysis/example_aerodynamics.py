# -*- coding: utf-8 -*-
"""Two worked examples: answers before the binding fix and after it.

Old answers come from report_build/aero_before_fix, new ones from the regenerated
samples.jsonl. The attitude example also compares against a Rodrigues DCM built
without touching the model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from audit_aerodynamics import AXES, build_frame, coords, mat  # noqa: E402

AERO = ROOT / "data" / "flight" / "basic"
BACKUP = ROOT / "report_build" / "aero_before_fix"


def load(path):
    return {row["id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def answers(row):
    return {q["id"]: q["answer"] for q in row["questions"]}


def rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


print("== 地面系 → 机体系，样本 1-01，纯俯仰 10° ==")
old = load(BACKUP / "ground_to_body_batch_samples.jsonl")["1-01"]
new = load(AERO / "ground_to_body/batch/samples.jsonl")["1-01"]
frames = {owner: build_frame(owner, new["spec"][f"{owner}_frame"]) for owner in AXES}
lateral_ground = coords(frames["ground"], "ground", "right")
lateral_body = coords(frames["body"], "body", "right")
print("地面侧向坐标:", lateral_ground.T.tolist(), " 机体侧向坐标:", lateral_body.T.tolist())
pure = {"ψ": 0, "Φ": 0, "θ": sp.pi / 18}
old_s = mat(answers(old)["27"]).subs(pure).evalf(4)
new_s = mat(answers(new)["28"]).subs(pure).evalf(4)
print("修复前 S:", old_s.tolist())
print("  映射侧向 →", (old_s * lateral_ground).evalf(4).T.tolist(), "（应", lateral_body.T.tolist(), "）")
print("修复后 S:", new_s.tolist())
print("  映射侧向 →", (new_s * lateral_ground).evalf(4).T.tolist())

G = np.array([frames["ground"][n] for n in AXES["ground"]], dtype=float)
B = np.array([frames["body"][n] for n in AXES["body"]], dtype=float)
independent = (rodrigues((0, 1, 0), np.deg2rad(10)) @ B.T).T @ G.T
print("独立 Rodrigues:", np.round(independent, 4).tolist())
print("max|修复后-独立| =", float(np.abs(np.array(new_s, dtype=float) - independent).max()))

print("\n== 体轴系 → 气流系，样本 1-01，α=β=10° ==")
old = load(BACKUP / "body_to_airflow_batch_samples.jsonl")["1-01"]
new = load(AERO / "body_to_airflow/batch/samples.jsonl")["1-01"]
frames = {owner: build_frame(owner, new["spec"][f"{owner}_frame"]) for owner in AXES}
body_lateral = coords(frames["body"], "body", "right")
stab_lateral = coords(frames["stability"], "stability", "right")
subs = {"α": sp.pi / 18, "β": sp.pi / 18}
old_a = mat(answers(old)["21"]).subs(subs).evalf(4)
new_a = mat(answers(new)["90"]).subs(subs).evalf(4)
print("修复前 Sα:", old_a.tolist())
print("  映射机体侧向 →", (old_a * body_lateral).evalf(4).T.tolist())
print("修复后 Sα:", new_a.tolist())
print("  映射机体侧向 →", (new_a * body_lateral).evalf(4).T.tolist(), "（应", stab_lateral.T.tolist(), "）")
print("修复前 Sαβ:", mat(answers(old)["35"]).subs(subs).evalf(4).tolist())
print("修复后 Sαβ:", mat(answers(new)["104"]).subs(subs).evalf(4).tolist())
