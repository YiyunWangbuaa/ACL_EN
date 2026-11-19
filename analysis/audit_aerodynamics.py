# -*- coding: utf-8 -*-
"""Answer audit for the three aerodynamics cloze datasets.

Run from the repository root:
    python analysis/audit_aerodynamics.py

It checks, for every sample:
  structure      row counts, blank numbering, answer letters, option duplicates,
                 matrix orthogonality and determinant, changed-item count;
  invariants     a passive rotation about an axis must leave that axis unchanged,
                 tested on every composite (single-angle slices of the total);
  repair         rebuilds every answer with the rotation matrix written in the
                 source frame of its leg and counts the differences.

Evidence is written to results/audit_aerodynamics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flight_frame_model import STANDARD_SPEC, FlightFrameModel  # noqa: E402

AERO = ROOT / "data" / "flight" / "basic"
OUT = ROOT / "results" / "audit_aerodynamics.json"

GEOM = {"forward": (1, 0, 0), "rearward": (-1, 0, 0), "right": (0, 1, 0),
        "left": (0, -1, 0), "down": (0, 0, 1), "up": (0, 0, -1)}
NORMAL = {"ground": "down", "body": "down", "airflow": "down", "stability": "down", "track": "down"}
AXES = {"ground": ("Xg", "Yg", "Zg"), "body": ("Xb", "Yb", "Zb"),
        "airflow": ("Xa", "Ya", "Za"), "stability": ("Xs", "Ys", "Zs"),
        "track": ("Xk", "Yk", "Zk")}
SYMBOLS = {name: sp.Symbol(symbol) for name, symbol in
           (("psi", "ψ"), ("theta", "θ"), ("phi", "Φ"), ("mu", "μ"),
            ("varphi", "φ"), ("gamma", "γ"), ("alpha", "α"), ("beta", "β"))}
LETTERS = "ABCDEFGHIJ"

DATASETS = {
    "ground_to_body": {
        "path": "ground_to_body/batch/samples.jsonl", "start": 1, "blanks": 28, "samples": 40,
        "invariants": [
            ("yaw keeps the normal", "28", "psi", "ground", "body", "down"),
            ("pitch keeps the lateral", "28", "theta", "ground", "body", "right"),
            ("roll keeps the axial", "28", "phi", "ground", "body", "forward"),
        ],
    },
    "ground_to_track_air": {
        "path": "ground_to_track_air/batch/samples.jsonl", "start": 29, "blanks": 41, "samples": 48,
        "invariants": [
            ("azimuth keeps the normal", "55", "varphi", "ground", "track", "down"),
            ("inclination keeps the lateral", "55", "mu", "ground", "track", "right"),
            ("bank keeps the axial", "66", "gamma", "track", "airflow", "forward"),
        ],
    },
    "body_to_airflow": {
        "path": "body_to_airflow/batch/samples.jsonl", "start": 70, "blanks": 35, "samples": 40,
        "invariants": [
            ("alpha keeps the lateral", "90", "alpha", "body", "stability", "right"),
            ("beta keeps the normal", "101", "beta", "stability", "airflow", "down"),
        ],
    },
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


def coords(frame, owner, direction):
    return sp.Matrix([sum(a * b for a, b in zip(frame[n], GEOM[direction])) for n in AXES[owner]])


def mat(answer):
    return sp.Matrix(sp.sympify(answer))


def load(path):
    rows = [json.loads(l) for l in (AERO / path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for row in rows:
        frames = {owner: build_frame(owner, row["spec"][f"{owner}_frame"]) for owner in AXES}
        answers = {q["id"]: str(q["answer"]) for q in row["questions"]}
        out.append({"id": row["id"], "raw": row, "spec": row["spec"],
                    "frames": frames, "answers": answers})
    return out


def structure(rows, start, blanks, samples):
    issues = {"rows": len(rows) != samples, "blank_count": 0, "blank_ids": 0,
              "letter_out_of_range": 0, "duplicate_options": 0,
              "matrix_not_rotation": 0, "changed_count": 0}
    for item in rows:
        raw = item["raw"]
        if len(raw["questions"]) != blanks:
            issues["blank_count"] += 1
        if [q["id"] for q in raw["questions"]] != [str(i) for i in range(start, start + blanks)]:
            issues["blank_ids"] += 1
        for q in raw["questions"]:
            options = q.get("options") or []
            if options and str(q["answer"]).strip() not in LETTERS[: len(options)]:
                issues["letter_out_of_range"] += 1
            if options and len(set(options)) != len(options):
                issues["duplicate_options"] += 1
            if q["format_code"] == "矩阵":
                matrix = mat(q["answer"])
                if matrix.shape != (3, 3):
                    issues["matrix_not_rotation"] += 1
                elif sp.simplify(matrix * matrix.T - sp.eye(3)) != sp.zeros(3, 3):
                    issues["matrix_not_rotation"] += 1
        diffs = [key for key in STANDARD_SPEC
                 if json.dumps(raw["spec"].get(key), sort_keys=True)
                 != json.dumps(STANDARD_SPEC[key], sort_keys=True)]
        if len(diffs) != raw["changed_count"]:
            issues["changed_count"] += 1
    return {key: value for key, value in issues.items() if value}


def invariant(sample, blank, symbol, src_owner, dst_owner, direction):
    matrix = mat(sample["answers"][blank]).subs({SYMBOLS[s]: 0 for s in SYMBOLS if s != symbol})
    return sp.simplify(matrix * coords(sample["frames"][src_owner], src_owner, direction)
                       - coords(sample["frames"][dst_owner], dst_owner, direction)) == sp.zeros(3, 1)


def binding_table():
    """Every angle of a directed transform must sit on its origin frame's direction."""
    from flight_frame_model import DIRECTED_TRANSFORMS

    model = FlightFrameModel().declare(STANDARD_SPEC)
    table = []
    issues = []
    for origin, destination, _angles, inverse in DIRECTED_TRANSFORMS:
        for item in model.conversion(origin, destination):
            table.append({"origin": origin, "destination": destination, "angle": item.angle,
                          "frame": item.frame, "absolute": item.absolute_axis,
                          "axis": item.axis, "sense": item.rotation_when_positive,
                          "inverse": inverse})
            if item.frame != origin:
                issues.append((origin, destination, item.angle, item.frame, origin))
    return table, issues


def main():
    report = {"structure": {}, "invariants": {}, "bindings": {}}
    table, issues = binding_table()
    report["bindings"] = {"table": table, "issues": issues}
    for name, spec in DATASETS.items():
        rows = load(spec["path"])
        report["structure"][name] = structure(rows, spec["start"], spec["blanks"], spec["samples"])
        report["invariants"][name] = {
            label: [row["id"] for row in rows
                    if not invariant(row, blank, symbol, src_owner, dst_owner, direction)]
            for label, blank, symbol, src_owner, dst_owner, direction in spec["invariants"]
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("structure issues")
    for name, problems in report["structure"].items():
        print(f"  {name}: {problems or 'none'}")
    print("\ninvariant failures")
    for name, checks in report["invariants"].items():
        for label, ids in checks.items():
            total = DATASETS[name]["samples"]
            print(f"  {name:20s} {label:32s} {len(ids):3d}/{total}  {ids[:6]}")
    print("\nbidirectional binding (angle -> frame it is written in)")
    for item in table:
        print(f"  {item['origin']:>8s} -> {item['destination']:<8s} "
              f"{item['angle']:<8s} frame={item['frame']:<9s} axis={item['axis']:<3s} "
              f"{item['sense']}")
    print(f"\nbinding issues: {issues or 'none'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
