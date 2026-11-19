# -*- coding: utf-8 -*-
"""Check the bidirectional bindings on freshly generated answers (no patching)."""
import json
import sys
from copy import deepcopy
from pathlib import Path

import sympy as sp

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

from domains.aerodynamics.body_air_generate import generate as gen_body_air  # noqa: E402
from domains.aerodynamics.ground_body_generate import generate as gen_gtb  # noqa: E402
from domains.aerodynamics.ground_track_air_generate import generate as gen_gta  # noqa: E402
from analysis.audit_aerodynamics import (  # noqa: E402
    AXES, build_frame, coords, DATASETS, SYMBOLS, mat, load,
)

GENS = {"ground_to_body": gen_gtb, "ground_to_track_air": gen_gta,
        "body_to_airflow": gen_body_air}


def answers(sample):
    return {item["manuscript_id"]: item["answer"] for item in sample["items"]}


bad = 0
for name, spec in DATASETS.items():
    rows = load(spec["path"])
    counts = {}
    for label, blank, symbol, src_owner, dst_owner, direction in spec["invariants"]:
        counts[label] = 0
        for row in rows:
            got = answers(GENS[name](deepcopy(row["spec"])))
            matrix = mat(got[blank]).subs({SYMBOLS[s]: 0 for s in SYMBOLS if s != symbol})
            if sp.simplify(matrix * coords(row["frames"][src_owner], src_owner, direction)
                           - coords(row["frames"][dst_owner], dst_owner, direction)) != sp.zeros(3, 1):
                counts[label] += 1
                bad += 1
    print(f"{name}: " + ", ".join(f"{k}={v}/{len(rows)}" for k, v in counts.items()))
print("total failures", bad)
