# -*- coding: utf-8 -*-
"""频率、传播深度、危害度三张对照榜。

危害度使用已经算完的最小修复集，不再重跑反事实：

    C_i = P(E_i) × Σ_j w_j × [E_i 属于终点 j 的最小修复集]

P 是该故障在两份自答里作为本地故障出现的份数 / 2。
终点自身的直接命中也算进它自己的最小修复集。
全部上游仍不能修好的终点没有最小修复集，不给其中任何一个故障加分。
权重：真实角和体轴力/力矩 3，弹性补偿 2，等效力矩 1。
同一份答卷里，二元集合的两个成员都拿到该终点的全部权重。
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.error_features import classify_wrongs  # noqa: E402
from domains.wind_tunnel.error_graph import NODE_BY_ID  # noqa: E402
from domains.wind_tunnel.eval_scope import scope_of  # noqa: E402
from domains.wind_tunnel.t_final import T_FINAL, final_nodes  # noqa: E402

def _known_grade(grade: str) -> bool:
    if grade in {"单因素", "无上游", "单修失败", "全上游仍失败", "≥3"}:
        return True
    head = grade[:-1]
    return grade.endswith("元") and head.isdigit() and int(head) >= 2


FIXES = ROOT / "analysis" / "CR最小修复集.md"
OUT = ROOT / "analysis" / "危害度榜.md"
SAMPLES = ("body-rule", "roll-rule")
WEIGHT = {}
for _node in ("80", "81", "111", "112", "113", "114", "115", "116"):
    WEIGHT[_node] = 3
for _node in ("176", "177", "178"):
    WEIGHT[_node] = 2
for _node in ("195", "196", "197", "198", "199", "200"):
    WEIGHT[_node] = 1


def _ids(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"ID(\d+)", text))


def _load_fixes(path: Path) -> dict[str, dict[str, dict]]:
    sample = ""
    current = ""
    found: dict[str, dict[str, dict]] = {name: {} for name in SAMPLES}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            sample = line[3:].strip()
            current = ""
            continue
        if sample not in SAMPLES:
            continue
        if line.startswith("### ID"):
            current = line.split()[1].removeprefix("ID")
            continue
        if line.startswith("上游：") and current:
            found[sample][current]["upstream"] = _ids(line.split("。", 1)[0])
            continue
        if not line.startswith("| ID"):
            continue
        cells = [item.strip() for item in line.strip("|").split("|")]
        if len(cells) < 5 or not _known_grade(cells[2]):
            continue
        target = cells[0].removeprefix("ID")
        groups = []
        if cells[3] != "—":
            for piece in cells[3].split("或"):
                groups.append(_ids(piece))
        row = found[sample].get(target, {})
        row.update({
            "grade": cells[2],
            "minimal": tuple(groups),
            "direct": cells[4] == "直接命中",
            "upstream": row.get("upstream", ()),
        })
        found[sample][target] = row
    return found


def _locals() -> dict[str, dict[str, str]]:
    found = {}
    for sample in SAMPLES:
        verdicts = classify_wrongs(sample, "self-trunc")
        found[sample] = {
            node: hit.fault for node, hit in verdicts.items() if hit.fault in {DIR, RSN, EXE, ALG}
        }
    return found


def _reach(fault: str) -> tuple[str, ...]:
    node = NODE_BY_ID[fault]
    reached = [
        target for target in final_nodes()
        if target == fault or target in node.descendants
    ]
    return tuple(sorted(reached, key=int))


def _boards(fixes: dict[str, dict[str, dict]], faults: dict[str, dict[str, str]]) -> str:
    nodes = sorted({node for sample in faults.values() for node in sample}, key=int)
    credited: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    unresolved: dict[str, list[str]] = defaultdict(list)
    for sample, rows in fixes.items():
        for target, row in rows.items():
            members = {fault for group in row["minimal"] for fault in group}
            if row["direct"]:
                members.add(target)
            for fault in members:
                credited[fault].append((sample, target, WEIGHT[target]))
    for sample, rows in fixes.items():
        short = "机体" if sample.startswith("body") else "滚转"
        for target, row in rows.items():
            if row["grade"] not in {"≥3", "单修失败", "全上游仍失败"}:
                continue
            for fault in row.get("upstream", ()):
                unresolved[fault].append(f"{short} ID{target}")

    lines = [
        "# 契约错误的频率、深度与危害度",
        "",
        "本地故障只计 DIR、RSN、EXE、ALG。传播不单独成行。",
        "频率 P 是两份自答里出现的份数 / 2。",
        "危害度 C = P × Σ w。真实角和体轴力/力矩的权重是 3，弹性补偿是 2，等效力矩是 1。",
        "只有写进最小修复集的故障才计入，包含终点自身的直接命中。",
        "同一组最小修复里的每个故障都拿该终点的全部权重。全部上游仍不能修好的终点没有最小修复集，C 不加分，另列在「未解开」。",
        "",
        "## 频率榜",
        "",
        "| 故障 | 层 | 题目 | 份数 | P |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    by_freq = sorted(nodes, key=lambda node: (-sum(node in faults[sample] for sample in SAMPLES), int(node)))
    for node in by_freq:
        count = sum(node in faults[sample] for sample in SAMPLES)
        layer = next(faults[sample][node] for sample in SAMPLES if node in faults[sample])
        lines.append(
            f"| ID{node} | {layer} | {NODE_BY_ID[node].role} | {count}/2 | {count / 2:.2f} |"
        )
    lines.extend([
        "",
        "## 深度榜",
        "",
        "下游跨度是最深终点的依赖深度减去故障自己的深度。跨度大的排在前面。",
        "",
        "| 故障 | 层 | 自身深度 | 最深终点 | 下游跨度 | 能到达的终点 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ])

    def depth_key(node: str) -> tuple:
        reached = _reach(node)
        deepest = max((NODE_BY_ID[item].dependency_depth for item in reached), default=NODE_BY_ID[node].dependency_depth)
        own = NODE_BY_ID[node].dependency_depth
        return (-(deepest - own), -deepest, own, int(node))

    for node in sorted(nodes, key=depth_key):
        reached = _reach(node)
        own = NODE_BY_ID[node].dependency_depth
        deepest = max((NODE_BY_ID[item].dependency_depth for item in reached), default=own)
        shown = "、".join(f"ID{item}" for item in reached) or "—"
        layer = next(faults[sample][node] for sample in SAMPLES if node in faults[sample])
        lines.append(f"| ID{node} | {layer} | {own} | {deepest} | {deepest - own} | {shown} |")
    lines.extend([
        "",
        "## 危害度榜",
        "",
        "| 故障 | 层 | P | 计入的终点 | C | 未解开 |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ])

    def harm_of(node: str) -> float:
        count = sum(node in faults[sample] for sample in SAMPLES)
        return (count / 2) * sum(weight for _sample, _target, weight in credited[node])

    for node in sorted(nodes, key=lambda item: (-harm_of(item), int(item))):
        count = sum(node in faults[sample] for sample in SAMPLES)
        hits = credited[node]
        shown = "、".join(
            f"{'机体' if sample.startswith('body') else '滚转'} ID{target}×{weight}"
            for sample, target, weight in hits
        ) or "—"
        pending = "、".join(unresolved[node]) or "—"
        layer = next(faults[sample][node] for sample in SAMPLES if node in faults[sample])
        lines.append(f"| ID{node} | {layer} | {count / 2:.2f} | {shown} | {harm_of(node):.2f} | {pending} |")
    lines.extend([
        "",
        "终点权重来自四个过程：",
        "",
    ])
    for process, group in T_FINAL.items():
        weight = WEIGHT[group[0]]
        lines.append(f"- {process}：{'、'.join(f'ID{node}' for node in group)}，权重 {weight}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not FIXES.exists():
        raise RuntimeError("还没有最小修复集，先运行 analysis.compute_cr。")
    fixes = _load_fixes(FIXES)
    for sample in SAMPLES:
        if not fixes[sample]:
            raise RuntimeError(f"{sample} 没有读到终点修复结果。")
    text = _boards(fixes, _locals())
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
