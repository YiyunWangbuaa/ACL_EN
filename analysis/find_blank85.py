# -*- coding: utf-8 -*-
"""Locate blank id 85 across the datasets and print its context."""
import json
import os
import sys

ROOT = r"C:\Users\minec\Desktop\ACL\workflow"
FILES = [
    "data/wind_tunnel/batch/samples.jsonl",
    "data/wind_tunnel/batch/pilot_10.json",
    "data/wind_tunnel/review/samples.json",
    "data/wind_tunnel/stress/samples.json",
    "data/flight/basic/pilot/samples.jsonl",
]

for rel in FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(rel, "missing")
        continue
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
    try:
        sample = json.loads(first)
    except Exception as exc:  # noqa: BLE001
        print(rel, "parse error", exc)
        continue
    questions = sample.get("questions") or sample.get("items") or []
    ids = [str(q.get("id", q.get("manuscript_id"))) for q in questions]
    has = "85" in ids
    print(f"{rel}: keys={list(sample)[:8]} n={len(questions)} has85={has}")
    if has:
        for q in questions:
            if str(q.get("id", q.get("manuscript_id"))) == "85":
                print("   ", json.dumps(q, ensure_ascii=False)[:300])
