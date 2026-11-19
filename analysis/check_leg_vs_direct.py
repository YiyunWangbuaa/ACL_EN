# -*- coding: utf-8 -*-
"""Compare the two readings of the body-to-airflow composite on a permuted sample.

  legs : A2 * R_beta(stability) * A1 * R_alpha(body)
  direct: R_beta(body) * R_alpha(body)
"""
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

from flight_frame_model import ROTATION_ANGLE, FlightFrameModel  # noqa: E402
from analysis.audit_aerodynamics import AXES, build_frame  # noqa: E402


def alignment(source, target, frames):
    src = [frames[source][n] for n in AXES[source]]
    dst = [frames[target][n] for n in AXES[target]]
    return sp.Matrix([[sum(a * b for a, b in zip(x, y)) for y in src] for x in dst])

rows = [json.loads(l) for l in
        (ROOT / "data/flight/basic/body_to_airflow/batch/samples.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
alpha, beta = sp.Symbol("α"), sp.Symbol("β")
t = ROTATION_ANGLE


def passive(slot, chirality, angle):
    sine = sp.sin(angle) * (-1 if chirality == "left" else 1)
    cosine = sp.cos(angle)
    if slot == 0:
        return sp.Matrix([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]])
    if slot == 1:
        return sp.Matrix([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    return sp.Matrix([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])


for sample in rows[:8]:
    spec = sample["spec"]
    frames = {o: build_frame(o, spec[f"{o}_frame"]) for o in AXES}
    chir = {o: spec[f"{o}_frame"]["chirality"] for o in AXES}
    model = FlightFrameModel().declare(spec)
    a1 = alignment("body", "stability", frames)
    a2 = alignment("stability", "airflow", frames)

    # slots: body lateral (alpha), stability normal (beta), body normal (beta)
    def slot_of(owner, direction):
        for i, name in enumerate(AXES[owner]):
            v = frames[owner][name]
            if abs(sum(a * b for a, b in zip(v, direction))) == 1:
                return i
        raise KeyError

    b_lat = slot_of("body", (0, 1, 0))
    s_norm = slot_of("stability", (0, 0, 1))
    b_norm = slot_of("body", (0, 0, 1))

    legs = sp.simplify(a2 * passive(s_norm, chir["stability"], beta) * a1 * passive(b_lat, chir["body"], alpha))
    direct = sp.simplify(passive(b_norm, chir["body"], beta) * passive(b_lat, chir["body"], alpha))
    equal = sp.simplify(legs - direct) == sp.zeros(3, 3)
    print(sample["id"], "legs == direct:", equal)
    if not equal:
        print("  legs  ", legs.tolist())
        print("  direct", direct.tolist())
