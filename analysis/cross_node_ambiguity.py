# -*- coding: utf-8 -*-
"""Ambiguity groups in the wind-tunnel binary reachable sets: different written blanks, the same reachable set."""
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
        return "same injection"
    if len(nodes) == 1:
        return "same blank, different value"
    return "different node set"


for sample in ("body-rule", "roll-rule"):
    results = [r for r in SG.evaluate(sample) if r.det == 1]
    by_support = defaultdict(list)
    for r in results:
        by_support[frozenset(r.entries)].append(r)
    print(f"\n=== {sample}: det=1 has {len(results)} rows, {len(by_support)} support sets ===")
    harmful = []
    for support, group in sorted(by_support.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 2:
            continue
        kind = classify(group)
        if kind == "different node set":
            harmful.append((support, group))
    print(f"different node set (harmful) groups: {len(harmful)}\n")
    for support, group in harmful:
        print(f"  support set of {len(support)} probes, {len(group)} rows:")
        for m in sorted(group, key=lambda x: x.instance.id):
            inj = ", ".join(f"{b}={v}" for b, v in m.instance.injector)
            print(f"    {m.instance.id:<40} [{m.instance.klass}] injected {inj}")
        print()
