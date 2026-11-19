# -*- coding: utf-8 -*-
"""Compare the aerodynamics answers before and after the binding fix."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
AERO = ROOT / "data" / "flight" / "basic"
BACKUP = ROOT / "report_build" / "aero_before_fix"

SETS = {
    "ground_to_body": ("ground_to_body/batch/samples.jsonl", "ground_to_body_batch_samples.jsonl"),
    "ground_to_track_air": ("ground_to_track_air/batch/samples.jsonl", "ground_to_track_air_batch_samples.jsonl"),
    "body_to_airflow": ("body_to_airflow/batch/samples.jsonl", "body_to_airflow_batch_samples.jsonl"),
}


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    for name, (rel, backup_name) in SETS.items():
        old = {row["id"]: row for row in load(BACKUP / backup_name)}
        new = {row["id"]: row for row in load(AERO / rel)}
        same_ids = set(old) == set(new)
        spec_same = all(json.dumps(old[key]["spec"], sort_keys=True) == json.dumps(new[key]["spec"], sort_keys=True)
                        for key in old)
        changed = {}
        for key in old:
            old_q = {q["id"]: q["answer"] for q in old[key]["questions"]}
            new_q = {q["id"]: q["answer"] for q in new[key]["questions"]}
            for blank in old_q:
                if str(old_q[blank]) != str(new_q[blank]):
                    changed[blank] = changed.get(blank, 0) + 1
        total = sum(changed.values())
        print(f"\n{name}: ids {len(old)}->{len(new)} same={same_ids}, specs identical={spec_same}")
        print(f"  changed answers: {total} of {len(old) * len(old_q)}")
        for blank in sorted(changed, key=int):
            print(f"    blank {blank:>2s}: {changed[blank]:3d} samples")


if __name__ == "__main__":
    main()
