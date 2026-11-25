# -*- coding: utf-8 -*-
"""Main-experiment vision set: one standard long manuscript and nine single factors.

Theory is not included. Each file is one 物体与双目 injection of 165 blanks.
The two-stage review in data/vision/review/scene is left unchanged.

Run::

    python -m domains.computer_vision.single_generate
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from domain_layout import DATA
from domains.computer_vision.batch_generate import _change_text, _frame_phrase
from domains.computer_vision.cv_generate import (
    ANGLE_FLIP,
    ANGLE_ZH,
    STANDARD_SPEC,
    alternate_frames,
    build,
)
from domains.computer_vision.cv_vision_context import apply_record, leftover
from domains.computer_vision.review_export import _declaration, _process

OUT = DATA / "vision" / "single"
# Nine definitions. 机构滚转 stays at the standard value in every row.
FACTORS = (
    ("wind", "世界坐标系"),
    ("body", "物体系"),
    ("camera1", "相机1系"),
    ("camera2", "相机2系"),
    ("pan", "相机方位角"),
    ("tilt", "相机俯仰角"),
    ("roll", "相机滚转角"),
    ("attack", "机构俯仰角"),
    ("sideslip", "机构方位角"),
)


def _spec(key: str) -> dict:
    spec = deepcopy(STANDARD_SPEC)
    if key in ANGLE_FLIP:
        spec["angles"][key] = ANGLE_FLIP[key]
        return spec
    for layout in alternate_frames(key):
        spec[key] = deepcopy(layout)
        return spec
    raise RuntimeError(f"{key} 没有可用的非默认声明。")


def _phrase(key: str, spec: dict) -> str:
    if key in ANGLE_ZH:
        return ANGLE_ZH[key]
    return _frame_phrase(key, spec[key])


def _answers(record: dict) -> list[str]:
    return [str(item["answer"]) for item in record["questions"]]


def _markdown(index: int, factor: str, row: dict) -> str:
    titles = {
        "frame": "绝对方向对应坐标轴",
        "camera_attitude": "相机姿态",
        "body_attitude": "物体姿态",
        "projection": "成像与物体前端投影",
        "axis_wind": "图像直线的方向向量",
        "extrinsic": "双目外参",
        "epipolar": "极线与三角化",
    }
    lines = [
        f"# 数据 {index:02d}",
        "",
        f"单定义扰动：{factor}",
        "",
        "本文件是机器视觉主实验的长篇幅稿，只含物体与双目，一次注入。不含基础理论。",
        "",
        row["change"],
        "",
        "# 物体与双目",
        "",
        "## 定义声明",
        "",
        _declaration(row["declaration"]),
        "",
        "## 完整过程",
        "",
        _process(row["questions"], titles),
        "",
    ]
    return "\n".join(lines)


def _check_words(row: dict) -> None:
    chunks = [row["declaration"], row["change"]]
    for item in row["questions"]:
        chunks.append(str(item.get("prompt") or ""))
        chunks.append(str(item.get("answer") or ""))
        chunks.extend(str(option) for option in item.get("options") or [])
    found = leftover("\n".join(chunks))
    if found:
        raise RuntimeError("改写后仍有风洞用语：" + "、".join(found))


def main() -> None:
    if len(FACTORS) != 9 or len({key for key, _name in FACTORS}) != 9:
        raise RuntimeError("机器视觉主实验要 9 个互不相同的单定义。")
    print("build standard", flush=True)
    built = build(STANDARD_SPEC)
    standard = apply_record({
        "id": "01",
        "title": "统一默认",
        "change": "单定义扰动：无。10 项可变定义都不改。",
        "declaration": built["declaration"],
        "questions": built["questions"],
    })
    if len(standard["questions"]) != 165:
        raise RuntimeError(f"标准稿有 {len(standard['questions'])} 个空，应为 165。")
    _check_words(standard)
    base = _answers(standard)
    rows = [("01", "无", standard)]
    for index, (key, name) in enumerate(FACTORS, start=2):
        print(f"build {key}", flush=True)
        spec = _spec(key)
        record = build(spec)
        if len(record["questions"]) != 165:
            raise RuntimeError(f"{key} 有 {len(record['questions'])} 个空。")
        if _answers(record) == base:
            raise RuntimeError(f"{key} 没有改变任何标准答案。")
        change = _change_text(0, [key], spec).replace("定义变更档0%：10项可变定义中变更1项：", "")
        row = apply_record({
            "id": f"{index:02d}",
            "title": name,
            "change": f"单定义扰动：{name}。{change}",
            "declaration": record["declaration"],
            "questions": record["questions"],
        })
        _check_words(row)
        rows.append((f"{index:02d}", name, row))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id, factor, row in rows:
        text = _markdown(int(sample_id), factor, row)
        if text.count("{ID:") != 165:
            raise RuntimeError(f"{sample_id} 槽数不是 165。")
        (OUT / f"sample_{sample_id}.md").write_text(text, encoding="utf-8")
        manifest.append({"id": sample_id, "factor": factor, "change": row["change"]})
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    lines = [
        "# 机器视觉主实验",
        "",
        "1 条标准定义和 9 条单定义扰动。每条都是物体与双目长篇幅稿，165 个空，一次注入。",
        "不含基础理论。data/vision/review/scene 里的两段旧稿保留，不作为这次主实验的入口。",
        "单定义扰动写在每条稿的开头，也写在 factors.json。",
        "",
    ]
    for item in manifest:
        lines.append(f"- 数据 {item['id']}：单定义扰动：{item['factor']}。")
    lines.append("")
    (OUT / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "factors.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
