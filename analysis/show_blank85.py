# -*- coding: utf-8 -*-
"""Show the pilot sample that owns blank 85 and the local blank around it."""
import json

PATH = r"C:\Users\minec\Desktop\ACL\workflow\data\flight\basic\pilot\samples.jsonl"

rows = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]
for row in rows:
    target = [q for q in row["questions"] if q["id"] == "85"]
    if not target:
        continue
    q = target[0]
    print("sample", row["id"], "tier", row.get("tier"), "changed", row.get("changed_count"))
    print("spec:", json.dumps(row["spec"], ensure_ascii=False, indent=1)[:1200])
    print("blank 85:", json.dumps(q, ensure_ascii=False))
    for other in row["questions"]:
        if other["id"] in {"83", "84", "85", "86", "87"}:
            print("  ", other["id"], other["label"], other["answer"], other.get("options"))
