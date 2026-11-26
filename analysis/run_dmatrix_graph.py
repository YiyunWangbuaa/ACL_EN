# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

from domains.wind_tunnel import dmatrix_graph as DG  # noqa: E402

for s in ("body-rule", "roll-rule"):
    v = DG.render(s)
    print(
        f"{s}: rows {len(v['rows'])}  multi-member groups {v['n_ambiguity']}"
        f" (label-repeat {v['n_label_repeat']}, true-ambiguity {v['n_true']})  max {v['max_group']}"
        f"  shared blanks {len(v['multi_class_blanks'])}"
    )
    for g in v["groups"]:
        if g["kind"] != "true-ambiguity":
            continue
        print(f"    [true-ambiguity] {len(g['members'])} members: {', '.join(g['members'])}")
    print()
