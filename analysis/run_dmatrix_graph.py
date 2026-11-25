# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

from domains.wind_tunnel import dmatrix_graph as DG  # noqa: E402

for s in ("body-rule", "roll-rule"):
    v = DG.render(s)
    print(
        f"{s}: 行 {len(v['rows'])}  多成员组 {v['n_ambiguity']}"
        f"（类标签重复 {v['n_label_repeat']}，真模糊 {v['n_true']}）  最大 {v['max_group']}"
        f"  共写空 {len(v['multi_class_blanks'])}"
    )
    for g in v["groups"]:
        if g["kind"] != "真模糊":
            continue
        print(f"    [真模糊] {len(g['members'])} 人: {', '.join(g['members'])}")
    print()
