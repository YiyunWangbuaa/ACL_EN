# -*- coding: utf-8 -*-
"""Write a human-readable review of the synced flight-mechanics control."""
from __future__ import annotations

import json

from domain_layout import SETS
from domains.flight_ctrl.fm_ctrl_dataset import FORMAT_EXAMPLE, FORMAT_TEXT, _injection, _render_process

OUT = SETS["reconstruction.flight"]["review"]
def _sample_markdown(row: dict, index: int) -> str:
    source = row["spec"]["source_id"]
    by_id = {item["id"]: item for item in row["questions"]}
    lines = [
        f"# 对照 {index:02d}　{row['id']}",
        "",
        f"同步风洞样本 {source}。安装固定正装。提问是一段连续填空，按句号分段。A 组一次填写当前这一句；前面的空写入标准答案原文，后面的段落不出现。",
        "",
        "下面是完整过程。槽里的标签和标准答案只给审阅用，模型看到的是待填槽，不含答案。",
        "",
        "## 定义声明",
        "",
        row["declaration"].strip(),
        "",
        "## 完整过程",
        "",
        _render_process(by_id),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _index(rows: list[dict]) -> str:
    lines = [
        "# 飞行力学对照组审阅",
        "",
        "这 10 条与风洞 pilot_10 同步：地面系对齐风轴系，机体系对齐体轴系，理论角和三个机构角的正方向相同。安装一律正装。每条 81 个空，与风洞「风轴系到体轴系」1–81 同构。选项使用地面系、机体系、气流系及其坐标轴。",
        "",
        "一段里按句号顺承。同一句中的空一次答完。前面各句在下一次输入里填入已有答案。",
        "",
        "正式输入格式见 00_注入格式.md。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [对照 {index:02d}](sample_{index:02d}.md)：{row['id']}，同步风洞 {row['spec']['source_id']}。")
    lines.append("")
    return "\n".join(lines)


def _format_document(example: str) -> str:
    lines = [
        "# 对照组注入格式",
        "",
        "A 组一次填写当前这一句。过程从第 1 段写到这一句的句号。前面的空填标准答案原文，不写选项字母。后面的段落不出现。同一句里的空放在同一个 JSON 对象中。",
        "",
        "答案放在一个 JSON 对象里。选择题的值是字母。矩阵、向量、表达式、arctan、arcsin 用英文双引号包成字符串。",
        "",
        "审阅稿里的标签和标准答案不会出现在模型输入中。",
        "",
        "## 输出格式名",
        "",
    ]
    for code, text in FORMAT_TEXT.items():
        if code in {"无", "矩阵", "向量", "表达式", "arctan", "arcsin"}:
            lines.append(f"- {code}：{text} 示例：{FORMAT_EXAMPLE[code]}")
    lines.extend([
        "",
        "## 一次真实注入",
        "",
        "下面是对照 01、空 5 所在句的 A 组输入。这一句同时填写空 5、空 6 和空 7。前面的空已经填上标准答案。",
        "",
        "```text",
        example,
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    rows = [
        json.loads(line)
        for line in SETS["reconstruction.flight"]["synced"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 10:
        raise RuntimeError(f"同步对照组应为 10 条，实际 {len(rows)} 条。")
    OUT.mkdir(exist_ok=True)
    for index, row in enumerate(rows, start=1):
        (OUT / f"sample_{index:02d}.md").write_text(_sample_markdown(row, index), encoding="utf-8")
    (OUT / "00_索引.md").write_text(_index(rows), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (OUT / "00_注入格式.md").write_text(_format_document(example), encoding="utf-8")
    print(f"wrote 10 reviews -> {OUT}")


if __name__ == "__main__":
    main()
