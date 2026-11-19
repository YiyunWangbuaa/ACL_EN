# -*- coding: utf-8 -*-
"""B-condition gold repair at one priority node, for every definition manuscript.

The repaired node's own score stays the model answer. Later sentences in that
subprocess receive the gold answer in the prefix.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = ROOT / "results" / "nothink_pair" / "repair"
BASE = ROOT / "results" / "nothink_pair"
PY = sys.executable

GROUPS = (
    ("standard", "standard-default-unmarked"),
    ("standard-order", "standard-order-unmarked"),
    ("alpha", "alpha-unmarked"),
    ("roll", "roll-unmarked"),
    ("order", "order-unmarked"),
    ("mount", "mount-unmarked"),
    ("body", "body-unmarked"),
)
NODES = ("24", "47", "48", "32", "41", "126")
LIMIT = 4


def complete(path: Path, node: str) -> bool:
    meta_path = path / "meta.json"
    if not meta_path.is_file() or not (path / "records.jsonl").is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        meta.get("status") == "complete"
        and meta.get("thinking") == "disabled"
        and meta.get("condition") == "self-trunc"
        and meta.get("gold_repaired") == [node]
    )


def launch(group: str, sample: str, node: str) -> subprocess.Popen:
    out = OUT / f"ID{node}" / group
    out.mkdir(parents=True, exist_ok=True)
    ids = BASE / "ids" / f"{group}.json"
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text(json.dumps([sample], ensure_ascii=False), encoding="utf-8")
    log = open(out / "run.log", "w", encoding="utf-8")
    command = [
        PY, "cloze_eval.py",
        "--domain", "wind_tunnel",
        "--model", "deepseek-flash",
        "--thinking", "disabled",
        "--condition", "self-trunc",
        "--gold-nodes", node,
        "--runs", "1",
        "--workers", "1",
        "--temperature", "0",
        "--seed", "20261102",
        "--data", str(DATA),
        "--ids-file", str(ids),
        "--out", str(out),
    ]
    print(f"start ID{node} {group}", flush=True)
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def run_all() -> None:
    pending = []
    for node in NODES:
        for group, sample in GROUPS:
            path = OUT / f"ID{node}" / group
            if complete(path, node):
                print(f"skip ID{node} {group}", flush=True)
                continue
            pending.append((group, sample, node))
    active: list[tuple[str, str, subprocess.Popen]] = []
    while pending or active:
        while pending and len(active) < LIMIT:
            group, sample, node = pending.pop(0)
            active.append((group, node, launch(group, sample, node)))
        if not active:
            break
        group, node, process = active.pop(0)
        code = process.wait()
        print(f"done ID{node} {group} exit={code}", flush=True)
        if code != 0 and not complete(OUT / f"ID{node}" / group, node):
            raise SystemExit(f"ID{node} {group} failed with {code}")


def _wrong(path: Path) -> dict[str, bool]:
    found = {}
    for line in (path / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for blank in row.get("blanks") or []:
            found[str(blank["blank_id"])] = not bool(blank.get("normalized"))
    return found


def write_effect() -> None:
    from domains.wind_tunnel.error_graph import NODE_BY_ID, _ensure_bound

    _ensure_bound()
    rows = []
    for node in NODES:
        graph = NODE_BY_ID[node]
        descendants = set(graph.descendants)
        for group, _sample in GROUPS:
            base = _wrong(BASE / group / "B")
            repair = _wrong(OUT / f"ID{node}" / group)
            shared = set(base) & set(repair)
            downstream = sorted(descendants & shared, key=int)
            base_wrong = [item for item in downstream if base[item]]
            repair_wrong = [item for item in downstream if repair[item]]
            fixed = [item for item in base_wrong if not repair[item]]
            new_wrong = [item for item in downstream if (not base[item]) and repair[item]]
            rows.append({
                "node": f"ID{node}",
                "group": group,
                "reachable_descendants": len(descendants),
                "observed_descendants": len(downstream),
                "baseline_wrong": len(base_wrong),
                "repair_wrong": len(repair_wrong),
                "intervention_effect": len(base_wrong) - len(repair_wrong),
                "fixed": [f"ID{item}" for item in fixed],
                "new_wrong": [f"ID{item}" for item in new_wrong],
            })
    (OUT / "effect.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 节点修复的下游错误变化",
        "",
        "B 组原运行与对应修复运行比较。修复节点自身仍按模型原答案计分，不计入下游。干预影响 = 修复前下游错误数 − 修复后下游错误数。",
        "",
        "| 节点 | 定义稿 | 可达后代 | 本次观察到的后代 | 修复前错误 | 修复后错误 | 干预影响 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "standard": "统一默认",
        "standard-order": "球腕顺序",
        "alpha": "理论迎角正负",
        "roll": "机构滚转方向",
        "order": "机构角顺序",
        "mount": "正装反装",
        "body": "体轴系手性",
    }
    for item in rows:
        lines.append(
            f"| {item['node']} | {labels[item['group']]} | {item['reachable_descendants']} | "
            f"{item['observed_descendants']} | {item['baseline_wrong']} | {item['repair_wrong']} | "
            f"{item['intervention_effect']} |"
        )
    lines.append("")
    (OUT / "effect.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "effect.md", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_all()
    write_effect()
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
