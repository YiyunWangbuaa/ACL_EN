# -*- coding: utf-8 -*-
"""汇总六组单条数据的正确率与推理成本。"""
import json
import os

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def stats(sub):
    d = os.path.join(BASE, sub)
    m = os.path.join(d, "meta.json")
    r = os.path.join(d, "records.jsonl")
    if not (os.path.exists(m) and os.path.exists(r)):
        return None
    meta = json.load(open(m, encoding="utf-8"))
    recs = [json.loads(l) for l in open(r, encoding="utf-8") if l.strip()]
    rt = tt = 0
    usd = 0.0
    lat = 0
    nb = nc = 0
    for rec in recs:
        u = rec.get("usage") or {}
        rt += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        tt += u.get("total_tokens") or 0
        usd += (rec.get("cost") or {}).get("usd") or 0.0
        lat += rec.get("latency_ms") or 0
        nb += len(rec.get("blanks") or [])
        nc += 1
    return dict(blanks=nb, calls=nc, exact=meta.get("exact"), norm=meta.get("normalized"),
                acc=round(100 * (meta.get("normalized") or 0) / nb, 1) if nb else None,
                rt=rt, rt_per_blank=round(rt / nb, 0) if nb else None,
                lat_per_call=round(lat / nc, 0) if nc else None,
                usd=round(usd, 4))


ROWS = [
    ("风洞×飞行力学 交叉", "wind_flash_think/one_025-003_gold-trunc_v3"),
    ("风洞×飞行力学 交叉 (e2e)", "wind_flash_think/one_025-003_final-e2e"),
    ("风洞×飞行力学 重构", "flight_ctrl_flash_think/one_025-003_gold-trunc_v3"),
    ("风洞×飞行力学 重构 (e2e)", "flight_ctrl_flash_think/one_025-003_final-e2e"),
    ("风洞×CV 交叉", "cv_flash_think/one_025-001_gold-trunc"),
    ("风洞×CV 交叉 (e2e)", "cv_flash_think/one_025-001_final-e2e"),
    ("风洞×CV 重构", "recon_vision_flash_think/one_025-001_gold-trunc"),
    ("风洞×CV 重构 (e2e)", "recon_vision_flash_think/one_025-001_final-e2e"),
    ("风洞×CV 基础", "cv_flash_think/one_basic-01_gold-trunc"),
    ("风洞×CV 基础 (e2e)", "cv_flash_think/one_basic-01_final-e2e"),
]
for title, sub in ROWS:
    s = stats(sub)
    if not s:
        print(title, "| (缺)")
        continue
    print(f"{title} | 空{s['blanks']} | 正确率 {s['acc']}% | 思考token/空 {s['rt_per_blank']} | 延迟ms/次 {s['lat_per_call']} | 美元 {s['usd']}")
