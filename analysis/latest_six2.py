# -*- coding: utf-8 -*-
"""最新六组（20261112）：同时给严格与归一化正确率，以及成本。"""
import json
import os
from collections import Counter

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def stats(top, sub):
    d = os.path.join(BASE, top, sub)
    m = os.path.join(d, "meta.json")
    r = os.path.join(d, "records.jsonl")
    if not (os.path.exists(m) and os.path.exists(r)):
        return None
    meta = json.load(open(m, encoding="utf-8"))
    recs = [json.loads(l) for l in open(r, encoding="utf-8") if l.strip()]
    rt = lat = nb = ex = nm = 0
    usd = 0.0
    for rec in recs:
        u = rec.get("usage") or {}
        rt += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        usd += (rec.get("cost") or {}).get("usd") or 0.0
        lat += rec.get("latency_ms") or 0
        for b in rec.get("blanks") or []:
            nb += 1
            if b.get("exact"):
                ex += 1
            if b.get("normalized"):
                nm += 1
    return dict(blanks=nb, exact=ex, norm=nm, acc_nm=round(100 * nm / nb, 1) if nb else None,
                rt_per_blank=round(rt / nb, 0) if nb else None,
                lat_per_call=round(lat / len(recs), 0) if recs else None, usd=round(usd, 4))


GROUPS = [
    ("风洞×飞行力学", "交叉", "wind_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "重构", "flight_ctrl_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "基础", "flight_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "交叉", "cv_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "重构", "recon_vision_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "基础", "cv_flash_think", "20261112_basic-01_A", "20261112_basic-01_final-e2e"),
]
for domain, level, top, g, e in GROUPS:
    gs = stats(top, g)
    es = stats(top, e)
    print(f"{domain} | {level}")
    print(f"  gold: 严格 {gs['exact']}/{gs['blanks']} | 归一 {gs['norm']}/{gs['blanks']} = {gs['acc_nm']}% | {gs['rt_per_blank']} tok/空 | {gs['lat_per_call']} ms/次")
    print(f"  e2e : 严格 {es['exact']}/{es['blanks']} | 归一 {es['norm']}/{es['blanks']} = {es['acc_nm']}% | {es['rt_per_blank']} tok/空 | {es['lat_per_call']} ms/次")
