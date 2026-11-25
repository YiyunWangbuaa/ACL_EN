from domains.fault_class import ALG, DIR, EXE, RSN
# -*- coding: utf-8 -*-
"""Render the D matrix so it can be looked at: row catalog, sparse rows, and one dense excerpt."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

from domains.wind_tunnel import measurement as ME  # noqa: E402

SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "body-rule"
OUT = None
if "--out" in sys.argv:
    OUT = Path(sys.argv[sys.argv.index("--out") + 1])
    sys.stdout = open(OUT, "w", encoding="utf-8")
payload = json.loads((ROOT / "analysis/dmatrix" / f"{SAMPLE}.json").read_text(encoding="utf-8"))
rows = payload["rows"]
catalog = {m.id: m for m in ME.catalog()}


def node_sort(mid: str) -> tuple:
    node = catalog[mid].node if mid in catalog else "999"
    return (int(node) if node.isdigit() else 999, mid)


print(f"# D 矩阵实图：{SAMPLE}（二值图可达）\n")
print(f"行 {len(rows)}，列 {len(catalog)}，可达节点 {sum(len(r['reach_nodes']) for r in rows.values())}\n")

print("## 一、行目录（行标 + 写入空）\n")
by_class = defaultdict(list)
for iid, row in rows.items():
    by_class[row["klass"]].append((iid, row))
for klass in (DIR, RSN, EXE, ALG):
    items = sorted(by_class[klass], key=lambda kv: kv[0])
    print(f"### {klass}：{len(items)} 行\n")
    for iid, row in items[:4]:
        blanks = ", ".join(f"ID{b}" for b in row["blanks"])
        print(f"- `{iid}`")
        print(f"  - 写入 {blanks}｜可达 {len(row['reach_nodes'])} 个节点")
    if len(items) > 4:
        print(f"- …其余 {len(items) - 4} 行同类，id 形如 `{items[-1][0]}`")
    print()

print("## 二、稀疏行视图（一行 = 一个故障；列出它可达的节点）\n")
picks = []
for klass in (DIR, RSN, EXE, ALG):
    live = [iid for iid, _row in by_class[klass]]
    live.sort(key=lambda i: -len(rows[i]["reach_nodes"]))
    picks.extend(live[:1])
for iid in picks:
    row = rows[iid]
    print(f"**`{iid}`**（可达 {len(row['reach_nodes'])} 个节点）")
    ordered = sorted((f"ID{node}" for node in row["reach_nodes"]), key=node_sort)
    for mid in ordered[:14]:
        print(f"  - `{mid}`")
    if len(ordered) > 14:
        print(f"  - …另 {len(ordered) - 14} 个节点")
    print()

print("## 三、密集切片（行 × 节点，1 = 图可达）\n")
sub = [iid for iid in picks if len(rows[iid]["reach_nodes"]) >= 2]
col_hits = Counter()
for iid in sub:
    for node in rows[iid]["reach_nodes"]:
        col_hits[f"ID{node}"] += 1
cols = [mid for mid, _c in col_hits.most_common(14)]
cols = sorted(cols, key=node_sort)
print("| 行 | " + " | ".join(f"`{c.replace('ID','')}`" for c in cols) + " |")
print("|---|" + "---|" * len(cols))
for iid in sub:
    got = {f"ID{node}" for node in rows[iid]["reach_nodes"]}
    cells = ["1" if column in got else "·" for column in cols]
    print(f"| `{iid[:34]}` | " + " | ".join(cells) + " |")
print("\n（`·` = 不可达）")
