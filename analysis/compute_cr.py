# -*- coding: utf-8 -*-
"""Single-repair contribution, and the smallest upstream repair set of each wrong terminal. The ceiling is every upstream local fault of that terminal."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.error_graph import NODE_BY_ID  # noqa: E402
from domains.wind_tunnel.t_final import (  # noqa: E402
    audit_finals,
    audit_repair_shape,
    minimal_fixes,
)

OUT = ROOT / "doc" / "CR_minimal_repair_sets.md"
SAMPLES = ("body-rule", "roll-rule")


def _ids(nodes: tuple[str, ...]) -> str:
    return ", ".join(f"ID{node}" for node in nodes)


def _sets(groups: tuple[tuple[str, ...], ...]) -> str:
    if not groups:
        return "—"
    return " or ".join("+".join(f"ID{node}" for node in group) for group in groups)


def _role(node: str) -> str:
    item = NODE_BY_ID.get(node)
    return "" if item is None else item.role


def _counts(rows: tuple[tuple[object, str], ...]) -> str:
    found = {"fixed": 0, "refused": 0, "still-wrong": 0}
    for _key, status in rows:
        found[status] = found.get(status, 0) + 1
    return f"fixed {found['fixed']}, refused {found['refused']}, still-wrong {found['still-wrong']}"


def _render(fixes: dict[str, tuple]) -> str:
    lines = [
        "# Minimal repair sets of the terminals",
        "",
        "The denominator is the terminals that are wrong in the self-answer. The gold prefix is not included.",
        "A repair changes only upstream local faults (DIR, RSN, EXE, ALG). The terminal is always recomputed. Its own answer is not replaced by gold.",
        "Local faults that were not selected keep the model text. Search grows from small to large. The ceiling is every upstream local fault of that terminal. The grade is all-upstream-still-fail only when restoring every one of them to gold still fails.",
        "A calculator refusal is counted apart from an answer that is still wrong after recomputation. A refusal means the faults that were left already stop the calculator from producing the terminal.",
        "",
    ]
    pooled: dict[str, dict] = defaultdict(lambda: {"fixed": 0, "refused": 0, "total": 0, "targets": set()})
    for sample in SAMPLES:
        lines.append(f"## {sample}")
        lines.append("")
        lines.append("| terminal | question | grade | minimal repair set | terminal itself | single repair | pair |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in fixes[sample]:
            pair_text = "—" if not item.pairs else _counts(item.pairs)
            lines.append(
                f"| ID{item.target} | {_role(item.target)} | {item.grade} | {_sets(item.minimal)} | "
                f"{'direct hit' if item.direct else '—'} | {_counts(item.single)} | {pair_text} |"
            )
            if item.direct:
                pooled[item.target]["fixed"] += 1
                pooled[item.target]["total"] += 1
                pooled[item.target]["targets"].add(item.target)
            for fault, status in item.single:
                rec = pooled[fault]
                rec["total"] += 1
                rec["targets"].add(item.target)
                if status == "fixed":
                    rec["fixed"] += 1
                elif status == "refused":
                    rec["refused"] += 1
        lines.append("")
        for item in fixes[sample]:
            if item.grade == "single-factor":
                continue
            lines.append(f"### ID{item.target} {_role(item.target)}")
            lines.append("")
            lines.append(f"Upstream: {_ids(item.upstream) if item.upstream else 'none'}.")
            shown = ", ".join(f"ID{node} {status}" for node, status in item.single) or "none"
            lines.append(f"Single repair: {shown}.")
            if item.pairs:
                ok = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "fixed"]
                refused = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "refused"]
                wrong = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "still-wrong"]
                lines.append("Pair fixed: " + (", ".join(ok) if ok else "none") + ".")
                lines.append("Pair refused: " + (", ".join(refused) if refused else "none") + ".")
                lines.append("Pair still wrong: " + (", ".join(wrong) if wrong else "none") + ".")
            lines.append("")
    lines.append("## Pooled single-repair contribution of the two sheets")
    lines.append("")
    lines.append("One repair corresponds to one terminal. The contribution is the number of times the terminal became correct after this fault was repaired alone, divided by the number of times it was used to repair a terminal.")
    lines.append("")
    lines.append("| fault | fixed | refused | queries | CR | terminals |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    ordered = sorted(pooled.items(), key=lambda kv: (-kv[1]["fixed"] / kv[1]["total"] if kv[1]["total"] else 0, int(kv[0])))
    for fault, rec in ordered:
        share = rec["fixed"] / rec["total"] if rec["total"] else 0.0
        targets = ", ".join(f"ID{node}" for node in sorted(rec["targets"], key=int))
        lines.append(f"| ID{fault} | {rec['fixed']} | {rec['refused']} | {rec['total']} | {share:.2f} | {targets} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    audit_finals()
    audit_repair_shape()
    print("Self-check passed (T_FINAL == 17 expression terminals; single-repair and pair-repair shapes are correct)", flush=True)
    fixes = {}
    for sample in SAMPLES:
        fixes[sample] = minimal_fixes(sample, "self-trunc", log=lambda line: print(line, flush=True))
    text = _render(fixes)
    OUT.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
