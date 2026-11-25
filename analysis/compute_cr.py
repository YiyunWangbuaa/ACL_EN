# -*- coding: utf-8 -*-
"""单修贡献率，以及答错终点的最小上游修复集。上界是该终点的全部上游本地故障。"""
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

OUT = ROOT / "analysis" / "CR最小修复集.md"
SAMPLES = ("body-rule", "roll-rule")


def _ids(nodes: tuple[str, ...]) -> str:
    return "、".join(f"ID{node}" for node in nodes)


def _sets(groups: tuple[tuple[str, ...], ...]) -> str:
    if not groups:
        return "—"
    return " 或 ".join("+".join(f"ID{node}" for node in group) for group in groups)


def _role(node: str) -> str:
    item = NODE_BY_ID.get(node)
    return "" if item is None else item.role


def _counts(rows: tuple[tuple[object, str], ...]) -> str:
    found = {"对": 0, "拒绝": 0, "仍错": 0}
    for _key, status in rows:
        found[status] = found.get(status, 0) + 1
    return f"修好 {found['对']}，拒绝 {found['拒绝']}，仍错 {found['仍错']}"


def _render(fixes: dict[str, tuple]) -> str:
    lines = [
        "# 终点的最小修复集",
        "",
        "分母是自答里答错的终点。金标前缀不在内。",
        "修复只改上游的本地故障（DIR、RSN、EXE、ALG）。终点一律重算，不把终点自己的答案直接换成金标。",
        "其它没被选中的本地故障保持模型原文。从小到大搜索，上界是该终点的全部上游本地故障；全部改回金标仍不能修好，才记为全上游仍失败。",
        "计算器拒绝和重算后仍错分开计。拒绝表示留下的故障已经让计算器算不出终点。",
        "",
    ]
    pooled: dict[str, dict] = defaultdict(lambda: {"fixed": 0, "refused": 0, "total": 0, "targets": set()})
    for sample in SAMPLES:
        lines.append(f"## {sample}")
        lines.append("")
        lines.append("| 终点 | 题目 | 级别 | 最小修复集 | 终点自身 | 单修 | 二元 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in fixes[sample]:
            pair_text = "—" if not item.pairs else _counts(item.pairs)
            lines.append(
                f"| ID{item.target} | {_role(item.target)} | {item.grade} | {_sets(item.minimal)} | "
                f"{'直接命中' if item.direct else '—'} | {_counts(item.single)} | {pair_text} |"
            )
            if item.direct:
                pooled[item.target]["fixed"] += 1
                pooled[item.target]["total"] += 1
                pooled[item.target]["targets"].add(item.target)
            for fault, status in item.single:
                rec = pooled[fault]
                rec["total"] += 1
                rec["targets"].add(item.target)
                if status == "对":
                    rec["fixed"] += 1
                elif status == "拒绝":
                    rec["refused"] += 1
        lines.append("")
        for item in fixes[sample]:
            if item.grade == "单因素":
                continue
            lines.append(f"### ID{item.target} {_role(item.target)}")
            lines.append("")
            lines.append(f"上游：{_ids(item.upstream) if item.upstream else '无'}。")
            shown = "、".join(f"ID{node} {status}" for node, status in item.single) or "无"
            lines.append(f"单修：{shown}。")
            if item.pairs:
                ok = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "对"]
                refused = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "拒绝"]
                wrong = ["+".join(f"ID{node}" for node in pair) for pair, status in item.pairs if status == "仍错"]
                lines.append("二元修好：" + ("、".join(ok) if ok else "无") + "。")
                lines.append("二元拒绝：" + ("、".join(refused) if refused else "无") + "。")
                lines.append("二元仍错：" + ("、".join(wrong) if wrong else "无") + "。")
            lines.append("")
    lines.append("## 两份合计的单修贡献率")
    lines.append("")
    lines.append("一次修复对应一个终点。贡献率是该故障单修后终点变对的次数，除以它被拿去修终点的次数。")
    lines.append("")
    lines.append("| 故障 | 修好 | 拒绝 | 查询数 | CR | 终点 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    ordered = sorted(pooled.items(), key=lambda kv: (-kv[1]["fixed"] / kv[1]["total"] if kv[1]["total"] else 0, int(kv[0])))
    for fault, rec in ordered:
        share = rec["fixed"] / rec["total"] if rec["total"] else 0.0
        targets = "、".join(f"ID{node}" for node in sorted(rec["targets"], key=int))
        lines.append(f"| ID{fault} | {rec['fixed']} | {rec['refused']} | {rec['total']} | {share:.2f} | {targets} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    audit_finals()
    audit_repair_shape()
    print("自检通过（T_FINAL == 17 个表达式终点；单修与二元修复的形状正确）", flush=True)
    fixes = {}
    for sample in SAMPLES:
        fixes[sample] = minimal_fixes(sample, "self-trunc", log=lambda line: print(line, flush=True))
    text = _render(fixes)
    OUT.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
