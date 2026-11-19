# -*- coding: utf-8 -*-
"""汇总 CV 与飞行力学各组单条结果的正确率。"""
import json
import os

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results"


def meta(sub):
    p = os.path.join(BASE, sub, "meta.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def show(top, subs=None):
    p = os.path.join(BASE, top)
    if not os.path.isdir(p):
        print("=== missing:", top)
        return
    print("===", top)
    for s in sorted(os.listdir(p)):
        d = os.path.join(p, s)
        if not os.path.isdir(d):
            continue
        if subs is not None and s not in subs:
            continue
        m = meta(top + "/" + s)
        if not m:
            print("  ", s, "| (no meta)")
            continue
        b = m.get("blanks")
        n = m.get("normalized")
        e = m.get("exact")
        acc = round(100 * n / b, 1) if b else None
        print("  ", s, "| blanks", b, "| exact", e, "| norm", n, "|", acc, "% | cond", m.get("condition"), "| samples", m.get("samples"))


for top in ["cv_flash_think", "recon_vision_flash_think", "wind_flash_think", "flight_flash_think", "flight_ctrl_flash_think"]:
    show(top)
    print()
