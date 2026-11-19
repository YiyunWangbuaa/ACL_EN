# -*- coding: utf-8 -*-
"""Write a human-readable review of the computer-vision pilot."""
from __future__ import annotations

import json
from pathlib import Path

from domain_layout import SETS
from domains.computer_vision.cv_dataset import _injection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = SETS["cross.wind_vision"]["review"]
SEGMENT_TITLE = {
    "frame": "绝对方向对应坐标轴",
    "camera_attitude": "相机姿态",
    "body_attitude": "机体姿态",
    "projection": "成像与机头投影",
    "axis_wind": "图像直线的方向向量",
    "extrinsic": "双目外参",
    "epipolar": "极线与三角化",
}


def _options(item: dict) -> str:
    options = item.get("options") or []
    if not options:
        return "无"
    return "；".join(f"{chr(65 + index)}：{option}" for index, option in enumerate(options))


def _slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, 选项:{_options(item)}, 输出格式:{item['format_code']}, "
        f"标签:{item['label']}, 答案:{item['answer']}}}"
    )


def _process(questions: list[dict], titles: dict | None = None) -> str:
    lines = []
    current = None
    paragraph: list[dict] = []

    def flush() -> None:
        if not paragraph:
            return
        number = paragraph[0].get("paragraph")
        text = f"{number}."
        for item in paragraph:
            text += item.get("lead", "") + _slot(item) + item.get("tail", "")
        lines.append(text)
        lines.append("")
        paragraph.clear()

    for item in questions:
        segment = item.get("segment") or ""
        if segment != current:
            flush()
            current = segment
            title = (titles or SEGMENT_TITLE).get(segment, segment or "过程")
            lines.extend(["", f"### {title}", ""])
        if item.get("paragraph"):
            if paragraph and paragraph[0].get("paragraph") != item.get("paragraph"):
                flush()
            paragraph.append(item)
            continue
        flush()
        lines.append(f"{item['id']}.{item['prompt']}{_slot(item)}")
    flush()
    return "\n".join(lines).strip()


def _declaration(text: str) -> str:
    body = text.strip()
    if body.startswith("【定义声明】"):
        body = body[len("【定义声明】"):].strip()
    return body


def _sample_markdown(row: dict, index: int, intro: str | None = None, titles: dict | None = None) -> str:
    lines = [
        f"# 数据 {index:02d}　{row['id']}",
        "",
        intro or (
            "本文件是审阅稿，展示这一条计算机视觉样本的定义声明和全部空。相机姿态按每个相机分成六段，旋转矩阵之后紧接该相机的位姿平移，机体姿态按风洞稿分成九段，成像按公式、内参、机头位置和反对称矩阵分成五段，一段里连续排多个槽；其余空仍单独成题。槽里的标签和标准答案只给审阅用，模型看到的待填槽不含这两项。"
        ),
        "",
        row["change"],
        "",
        "## 定义声明",
        "",
        _declaration(row["declaration"]),
        "",
        "## 完整过程",
        "",
        "下面每个槽都带标签和标准答案，便于直接审阅。这是审阅字段，不是模型在待填空上看到的字段。",
        "",
        _process(row["questions"], titles),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _index(rows: list[dict]) -> str:
    lines = [
        "# 计算机视觉审阅",
        "",
        "这 10 条是已经评过的 pilot：25%、50%、75%、100% 四档，条数 3、3、2、2。机体姿态一段沿用风洞风轴系向机体系的过程，角度定义用机构角。过程名是计算机视觉双目姿态估计。",
        "",
        "正式输入格式见 00_注入格式.md。端到端提问见 [端到端](端到端.md)。改题面或标准答案时，直接改对应样本里的槽。",
        "",
        "相机姿态按每个相机分成六段，旋转矩阵之后紧接该相机的位姿平移。编号相对上一轮已移动。上一轮按旧编号统计的低正确率空不再对应现在的题。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [数据 {index:02d}](sample_{index:02d}.md)：{row['id']}，{row['change']}")
    lines.append("")
    return "\n".join(lines)


def _format_document(example: str) -> str:
    lines = [
        "# 计算机视觉注入格式",
        "",
        "A 组一次填写一个空。输入是定义声明、这一空的题面，以及输出要求。前面的空不带进后一题，因为每题题面自己写全。",
        "",
        "选择题只回答一个选项字母。矩阵写成 [[a, b, c], [d, e, f], [g, h, i]]。向量写成 [a, b, c]，不写成矩阵。",
        "",
        "审阅稿里的标签和标准答案不会出现在模型输入中。",
        "",
        "## 一次真实注入",
        "",
        "下面是数据 01、空 5 的 A 组输入。",
        "",
        "```text",
        example,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = [
        json.loads(line)
        for line in SETS["cross.wind_vision"]["samples"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 10:
        raise RuntimeError(f"计算机视觉 pilot 应为 10 条，实际 {len(rows)} 条。")
    OUT.mkdir(exist_ok=True)
    for index, row in enumerate(rows, start=1):
        if len(row["questions"]) != 165:
            raise RuntimeError(f"{row['id']} 有 {len(row['questions'])} 个空，应为 165。")
        (OUT / f"sample_{index:02d}.md").write_text(_sample_markdown(row, index), encoding="utf-8")
    (OUT / "00_索引.md").write_text(_index(rows), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (OUT / "00_注入格式.md").write_text(_format_document(example), encoding="utf-8")
    print(f"wrote 10 reviews -> {OUT}")


if __name__ == "__main__":
    main()
