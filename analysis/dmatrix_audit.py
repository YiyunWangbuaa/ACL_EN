# -*- coding: utf-8 -*-
"""Dump the graph-reachability D matrix to JSON, then look for anomalies.

The dump reads FaultInstance.potential. It does not inject a wrong answer.

Usage:
  python analysis/dmatrix_audit.py --build
  python analysis/dmatrix_audit.py
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "analysis" / "dmatrix"
SAMPLES = ("body-rule", "roll-rule")


def build() -> None:
    from domains.wind_tunnel.dmatrix_graph import render

    OUT.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        view = render(sample)
        payload = {
            "sample": sample,
            "source": "graph-reachability",
            "n_label_repeat": view["n_label_repeat"],
            "n_true": view["n_true"],
            "rows": {
                row["fault"]: {
                    "klass": row["klass"],
                    "blanks": row["blanks"],
                    "reach_nodes": row["reach_nodes"],
                }
                for row in view["rows"]
            },
            "groups": [
                {
                    "kind": group["kind"],
                    "members": list(group["members"]),
                    "classes": list(group["classes"]),
                    "blanks": list(group["blanks"]),
                    "shared": list(group["shared"]),
                }
                for group in view["groups"]
            ],
        }
        path = OUT / f"{sample}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  built {path.name}  rows={len(payload['rows'])}  KB={path.stat().st_size//1024}")


def analyse(sample: str) -> None:
    from domains.wind_tunnel import measurement as ME

    payload = json.loads((OUT / f"{sample}.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    catalog = {m.id: m for m in ME.catalog()}
    print(f"\n{'='*70}\n{sample}: 行 {len(rows)}，测点 {len(catalog)}")

    hit = Counter()
    per_col_inst = defaultdict(set)
    for row in rows.values():
        for node in row["reach_nodes"]:
            mid = f"ID{node}"
            hit[mid] += 1
            per_col_inst[mid].add(row["klass"])

    node_columns = [mid for mid, item in catalog.items() if item.kind == "node"]
    untouched = [mid for mid in node_columns if hit[mid] == 0]
    print(f"\n--- 1. 没有任何行可达的节点列：{len(untouched)} / {len(node_columns)} ---")
    for mid in untouched[:40]:
        print(f"    {mid:<28} node={catalog[mid].node}")

    print(f"\n--- 2. 被最多行可达的节点（低区分度）---")
    for mid, count in hit.most_common(12):
        print(f"    {mid:<28} 被 {count:<4} 行到达")

    signature = defaultdict(list)
    for mid, inst in per_col_inst.items():
        signature[frozenset(inst)].append(mid)
    duplicates = [group for group in signature.values() if len(group) > 1]
    print(f"\n--- 3. 到达行的类集合完全相同的节点组：{len(duplicates)} 组 ---")
    for group in sorted(duplicates, key=len, reverse=True)[:8]:
        print(f"    {len(group)} 个：{sorted(group)[:8]}")

    groups = payload.get("groups") or []
    counts = Counter(group["kind"] for group in groups)
    print(
        f"\n--- 4. 模糊组：类标签重复 {counts.get('类标签重复', 0)}"
        f"，同注入同标签 {counts.get('同注入同标签', 0)}"
        f"，真模糊 {counts.get('真模糊', 0)} ---"
    )
    for group in groups:
        if group["kind"] != "真模糊":
            continue
        print(f"    {len(group['members'])} 人 共有空 {','.join(group['shared']) or '无'}: {', '.join(group['members'])}")

    print("\n--- 5. 每行可达节点数分布 ---")
    sizes = Counter(len(row["reach_nodes"]) for row in rows.values())
    print(f"    {dict(sorted(sizes.items()))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--sample", default="")
    args = parser.parse_args()
    if args.build:
        build()
        return
    for sample in (args.sample,) if args.sample else SAMPLES:
        analyse(sample)


if __name__ == "__main__":
    main()
