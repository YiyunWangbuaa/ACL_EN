# -*- coding: utf-8 -*-
"""汇总 20261112 最新六组单条数据：正确率 + 推理成本。"""
import json
import os

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def stats(top, sub):
    d = os.path.join(BASE, top, sub)
    m = os.path.join(d, "meta.json")
    r = os.path.join(d, "records.jsonl")
    if not (os.path.exists(m) and os.path.exists(r)):
        return None
    meta = json.load(open(m, encoding="utf-8"))
    recs = [json.loads(l) for l in open(r, encoding="utf-8") if l.strip()]
    rt = tt = lat = nb = 0
    usd = 0.0
    for rec in recs:
        u = rec.get("usage") or {}
        rt += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        tt += u.get("total_tokens") or 0
        usd += (rec.get("cost") or {}).get("usd") or 0.0
        lat += rec.get("latency_ms") or 0
        nb += len(rec.get("blanks") or [])
    rec_n = len(recs)
    return dict(blanks=nb, calls=rec_n, norm=meta.get("normalized"),
                acc=round(100 * (meta.get("normalized") or 0) / nb, 1) if nb else None,
                rt=rt, rt_per_blank=round(rt / nb, 0) if nb else None,
                lat_per_call=round(lat / rec_n, 0) if rec_n else None,
                usd=round(usd, 4))


ROWS = [
    ("风洞×飞行力学", "交叉", "wind_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "重构", "flight_ctrl_flash_think", "20261112_025-003_A", "20261112_025-003_final-e2e"),
    ("风洞×飞行力学", "基础", "flight_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "交叉", "cv_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "重构", "recon_vision_flash_think", "20261112_025-001_A", "20261112_025-001_final-e2e"),
    ("风洞×CV", "基础", "cv_flash_think", "20261112_basic-01_A", "20261112_basic-01_final-e2e"),
]
for domain, level, top, gsub, esub in ROWS:
    g = stats(top, gsub)
    e = stats(top, esub)
    print(f"{domain} | {level} | gold {g['acc']}% ({g['norm']}/{g['blanks']}, {g['calls']}次) | e2e {e['acc']}% ({e['norm']}/{e['blanks']}, {e['calls']}次)")
    print(f"    gold: {g['rt_per_blank']} tok/空, {g['lat_per_call']}ms/次, ${g['usd']}")
    print(f"    e2e : {e['rt_per_blank']} tok/空, {e['lat_per_call']}ms/次, ${e['usd']}")
