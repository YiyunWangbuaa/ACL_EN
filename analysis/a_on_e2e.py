# -*- coding: utf-8 -*-
"""A 组（gold-trunc）中，与 final-e2e 目标空同 ID 的那些空的正确率。"""
import json
import os

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def load(top, sub):
    p = os.path.join(BASE, top, sub, "records.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def blank_ids(top, sub):
    ids = []
    seen = set()
    for r in load(top, sub):
        for b in r.get("blanks") or []:
            bid = b.get("blank_id")
            if bid not in seen:
                seen.add(bid)
                ids.append(bid)
    return ids


def a_map(top, sub):
    out = {}
    for r in load(top, sub):
        for b in r.get("blanks") or []:
            out[b.get("blank_id")] = (bool(b.get("exact")), bool(b.get("normalized")))
    return out


GROUPS = [
    ("风洞×飞行力学", "交叉", "wind_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "重构", "flight_ctrl_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "基础", "flight_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "交叉", "cv_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "重构", "recon_vision_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "基础", "cv_flash_think", "20261112_basic-01_A", "20261112_basic-01_final-e2e"),
]
for domain, level, top, g, e in GROUPS:
    amap = a_map(top, g)
    eids = blank_ids(top, e)
    present = [i for i in eids if i in amap]
    missing = [i for i in eids if i not in amap]
    ex = sum(1 for i in present if amap[i][0])
    nm = sum(1 for i in present if amap[i][1])
    print(f"{domain} | {level} | e2e目标 {len(eids)} 空 | A组命中 {len(present)} 空 | A组严格 {ex}/{len(present)} | A组归一 {nm}/{len(present)}")
    if missing:
        print("    A组缺失这些ID:", missing)
