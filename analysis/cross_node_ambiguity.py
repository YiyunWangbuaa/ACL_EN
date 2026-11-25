# -*- coding: utf-8 -*-
"""风洞二值可达集里，写入的空不同、可达集却相同的模糊组。"""
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

from domains.wind_tunnel import signature as SG  # noqa: E402


def node_set(injector):
    return frozenset(b for b, _v in injector)


def classify(group):
    injectors = [m.instance.injector for m in group]
    nodes = {node_set(i) for i in injectors}
    values = {i for i in injectors}
    if len(values) == 1:
        return "同一注入"
    if len(nodes) == 1:
        return "同空不同值"
    return "节点集不同"


for sample in ("body-rule", "roll-rule"):
    results = [r for r in SG.evaluate(sample) if r.det == 1]
    by_support = defaultdict(list)
    for r in results:
        by_support[frozenset(r.entries)].append(r)
    print(f"\n=== {sample}：det=1 共 {len(results)} 行，支撑集 {len(by_support)} 个 ===")
    harmful = []
    for support, group in sorted(by_support.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 2:
            continue
        kind = classify(group)
        if kind == "节点集不同":
            harmful.append((support, group))
    print(f"节点集不同（有害）组：{len(harmful)} 个\n")
    for support, group in harmful:
        print(f"  支撑集 {len(support)} 个测点，{len(group)} 行：")
        for m in sorted(group, key=lambda x: x.instance.id):
            inj = ", ".join(f"{b}={v}" for b, v in m.instance.injector)
            print(f"    {m.instance.id:<40} [{m.instance.klass}] 注入 {inj}")
        print()
