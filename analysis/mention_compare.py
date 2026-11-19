# -*- coding: utf-8 -*-
"""Compare the unmarked run with the all-rule-question run on the shared blanks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNMARKED = ROOT / "results" / "abcd_body"
MENTION = ROOT / "results" / "abcd_mention"
CODES = ("A", "B", "C", "D")
LABELS = {
    "A": "标准定义，标准前缀",
    "B": "标准定义，模型前缀",
    "C": "左手体轴系，标准前缀",
    "D": "左手体轴系，模型前缀",
}


def load(path: Path) -> dict[str, bool]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for blank in row.get("blanks") or []:
            found[str(blank["blank_id"])] = bool(blank.get("normalized"))
    if not found:
        raise RuntimeError(f"{path} 里没有评分。")
    return found


def main() -> None:
    unmarked = {code: load(UNMARKED / code / "records.jsonl") for code in CODES}
    mention = {code: load(MENTION / code / "records.jsonl") for code in CODES}
    shared = set(unmarked["A"])
    for code in CODES:
        if set(unmarked[code]) != shared:
            raise SystemExit(f"无强调 {code} 的空和无强调 A 不一致。")
        missing = shared - set(mention[code])
        if missing:
            raise SystemExit(f"规则提及 {code} 缺这些共有空：{sorted(missing, key=int)}")
    recalled = sorted(set(mention["A"]) - shared, key=int)
    if len(recalled) != 18:
        raise SystemExit(f"规则询问是 {len(recalled)} 个，应为 18 个。")

    lines = [
        "# 规则询问是否改变准确率",
        "",
        "无强调稿去掉 18 个规则询问。规则提及稿把这 18 个加回去，定义声明和其余 176 个空的标准答案不变。",
        "下面的对比只算这 176 个空。规则询问自己的对错另列，不混进这个准确率。",
        "对错按归一化分数。一次运行，差值是这批题上的题数差，不是抽样区间。",
        "",
        "## 共有 176 题",
        "",
        "| 条件 | 无强调 | 规则提及 | 提及后改对 | 提及后改错 |",
        "| --- | --- | --- | --- | --- |",
    ]
    details = []
    for code in CODES:
        base = unmarked[code]
        full = mention[code]
        hit_base = sum(base[item] for item in shared)
        hit_full = sum(full[item] for item in shared)
        gained = sorted((item for item in shared if (not base[item]) and full[item]), key=int)
        lost = sorted((item for item in shared if base[item] and (not full[item])), key=int)
        lines.append(
            f"| {code} {LABELS[code]} | {hit_base}/176 | {hit_full}/176 | "
            f"{len(gained)} | {len(lost)} |"
        )
        details.append((code, gained, lost))
    lines.extend([
        "",
        "## 18 个规则询问",
        "",
        "这些题只出现在规则提及稿里。",
        "",
        "| 条件 | 正确 |",
        "| --- | --- |",
    ])
    for code in CODES:
        hit = sum(mention[code][item] for item in recalled)
        lines.append(f"| {code} {LABELS[code]} | {hit}/18 |")
    lines.extend(["", f"规则询问：{'、'.join(f'ID{item}' for item in recalled)}。", ""])
    lines.append("## 共有题里发生变化的题")
    lines.append("")
    for code, gained, lost in details:
        lines.append(f"### {code}")
        lines.append("")
        lines.append("提及后改对：" + ("、".join(f"ID{item}" for item in gained) or "无"))
        lines.append("")
        lines.append("提及后改错：" + ("、".join(f"ID{item}" for item in lost) or "无"))
        lines.append("")
    target = MENTION / "compare.md"
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
