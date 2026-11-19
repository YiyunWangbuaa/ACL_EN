# -*- coding: utf-8 -*-
"""Compare body_air_generate.generate with the pilot record for one sample."""
import json
import sys

ROOT = r"C:\Users\minec\Desktop\ACL\workflow"
sys.path.insert(0, ROOT)

from domains.aerodynamics.body_air_generate import generate  # noqa: E402
from domains.aerodynamics.body_air_dataset import build_record  # noqa: E402

PATH = ROOT + r"\data\flight\basic\pilot\samples.jsonl"
rows = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]
row = next(r for r in rows if r["id"] == "025-001")

payload = generate(json.loads(json.dumps(row["spec"])))
for item in payload["items"]:
    if item["manuscript_id"] in {"15", "16", "17"}:
        print("generate:", item["manuscript_id"], item["answer"], item["options"])

record = build_record(row)
for item in record["questions"]:
    if item["local_id"] in {"15", "16", "17"}:
        print("dataset :", item["local_id"], item["answer"], item["options"])

for q in row["questions"]:
    if q["local_id"] in {"15", "16", "17"}:
        print("pilot   :", q["local_id"], q["answer"], q["options"])
