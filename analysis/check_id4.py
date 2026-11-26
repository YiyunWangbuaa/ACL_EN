# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

from domains.aerodynamics.error_graph import nodes  # noqa: E402
from domains.aerodynamics.signed_lists import T_FINAL  # noqa: E402

G = nodes()
n = G.get("4")
print("ID4 depends_on:", n.depends_on if n else None)

desc = set()
if n:
    pending = ["4"]
    seen = set()
    while pending:
        x = pending.pop()
        if x in seen:
            continue
        seen.add(x)
        for child, item in G.items():
            if x in item.depends_on and child not in seen:
                pending.append(child)
    desc = {c for c in seen if c != "4"}

print("ID4 direct children:", sorted((c for c in G if "4" in G[c].depends_on), key=int))
print("ID4 descendants:", sorted(desc, key=int))
print("T_FINAL:", T_FINAL)
print("ID4 descendants intersect T_FINAL:", sorted(desc & set(T_FINAL), key=int))
