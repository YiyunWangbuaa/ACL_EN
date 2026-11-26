# -*- coding: utf-8 -*-
"""Three comparison rankings: frequency, propagation depth, and harm.

Harm uses the minimal repair sets that have already been computed. It does not rerun the counterfactual:

    C_i = P(E_i) x sum_j w_j x [E_i belongs to the minimal repair set of terminal j]

P is the number of the two self-answers in which the fault appears as a local fault, divided by 2.
A direct hit on the terminal itself is counted in that terminal's own minimal repair set.
A terminal that still fails after every upstream fault is restored has no minimal repair set, and none of its faults receives a point.
Weights: true angles and body-axis force/moment 3, elastic compensation 2, equivalent moment 1.
In one answer sheet, both members of a pair set receive the full weight of that terminal.
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
    if grade in {"single-factor", "no-upstream", "single-repair-failed", "all-upstream-still-fail", "≥3"}:
        return True
    if not grade.endswith("-factor"):
        return False
    head = grade[: -len("-factor")]
    return head.isdigit() and int(head) >= 2


FIXES = ROOT / "doc" / "CR_minimal_repair_sets.md"
OUT = ROOT / "doc" / "harm_ranking.md"
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
        if line.startswith("Upstream:") and current:
            found[sample][current]["upstream"] = _ids(line.split(".", 1)[0])
            continue
        if not line.startswith("| ID"):
            continue
        cells = [item.strip() for item in line.strip("|").split("|")]
        if len(cells) < 5 or not _known_grade(cells[2]):
            continue
        target = cells[0].removeprefix("ID")
        groups = []
        if cells[3] != "—":
            for piece in cells[3].split(" or "):
                groups.append(_ids(piece))
        row = found[sample].get(target, {})
        row.update({
            "grade": cells[2],
            "minimal": tuple(groups),
            "direct": cells[4] == "direct hit",
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
        short = "body" if sample.startswith("body") else "roll"
        for target, row in rows.items():
            if row["grade"] not in {"≥3", "single-repair-failed", "all-upstream-still-fail"}:
                continue
            for fault in row.get("upstream", ()):
                unresolved[fault].append(f"{short} ID{target}")

    lines = [
        "# Frequency, depth, and harm of contract errors",
        "",
        "Local faults count only DIR, RSN, EXE, and ALG. Propagation is not a row of its own.",
        "Frequency P is the number of the two self-answers in which the fault appears, divided by 2.",
        "Harm C = P x sum w. True angles and body-axis force/moment have weight 3, elastic compensation has weight 2, and equivalent moment has weight 1.",
        "Only a fault written into a minimal repair set is counted, including a direct hit on the terminal itself.",
        "Every fault in one minimal repair set receives the full weight of that terminal. A terminal that still fails after every upstream fault is restored has no minimal repair set, C gets no point, and those faults are listed under unresolved.",
        "",
        "## Frequency ranking",
        "",
        "| fault | layer | question | count | P |",
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
        "## Depth ranking",
        "",
        "Downstream span is the dependency depth of the deepest terminal minus the fault's own depth. A larger span ranks first.",
        "",
        "| fault | layer | own depth | deepest terminal | downstream span | reachable terminals |",
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
        shown = ", ".join(f"ID{item}" for item in reached) or "—"
        layer = next(faults[sample][node] for sample in SAMPLES if node in faults[sample])
        lines.append(f"| ID{node} | {layer} | {own} | {deepest} | {deepest - own} | {shown} |")
    lines.extend([
        "",
        "## Harm ranking",
        "",
        "| fault | layer | P | credited terminals | C | unresolved |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ])

    def harm_of(node: str) -> float:
        count = sum(node in faults[sample] for sample in SAMPLES)
        return (count / 2) * sum(weight for _sample, _target, weight in credited[node])

    for node in sorted(nodes, key=lambda item: (-harm_of(item), int(item))):
        count = sum(node in faults[sample] for sample in SAMPLES)
        hits = credited[node]
        shown = "、".join(
            f"{'body' if sample.startswith('body') else 'roll'} ID{target}x{weight}"
            for sample, target, weight in hits
        ) or "—"
        pending = ", ".join(unresolved[node]) or "—"
        layer = next(faults[sample][node] for sample in SAMPLES if node in faults[sample])
        lines.append(f"| ID{node} | {layer} | {count / 2:.2f} | {shown} | {harm_of(node):.2f} | {pending} |")
    lines.extend([
        "",
        "Terminal weights come from four processes:",
        "",
    ])
    for process, group in T_FINAL.items():
        weight = WEIGHT[group[0]]
        lines.append(f"- {process}: {', '.join(f'ID{node}' for node in group)}, weight {weight}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not FIXES.exists():
        raise RuntimeError("No minimal repair set yet. Run analysis.compute_cr first.")
    fixes = _load_fixes(FIXES)
    for sample in SAMPLES:
        if not fixes[sample]:
            raise RuntimeError(f"{sample} has no terminal repair result.")
    text = _boards(fixes, _locals())
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
