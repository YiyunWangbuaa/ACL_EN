# -*- coding: utf-8 -*-
"""单条数据：从多采样结果中抽某个样本的准确率（用于补齐基础层级单条）。"""
import json
import os
from collections import defaultdict

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def per_sample(sub):
    p = os.path.join(BASE, sub, "records.jsonl")
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        sid = r.get("sample_id")
        for b in r.get("blanks") or []:
            agg[sid][0] += 1
            if b.get("normalized"):
                agg[sid][1] += 1
    return {k: (v[1], v[0], round(100 * v[1] / v[0], 1) if v[0] else None) for k, v in agg.items()}


for sub in ["flight_flash_think/pilot_gold-trunc", "flight_flash_think/pilot_final-e2e",
            "flight_flash_think/def_probe_A", "flight_flash_think/standard_final-e2e"]:
    r = per_sample(sub)
    print("===", sub)
    if r is None:
        print("  (no records)")
        continue
    for k in sorted(r):
        print("  ", k, "->", r[k])
    print()
