# -*- coding: utf-8 -*-
"""Build the definition-question ablation from a hand-marked review.

The marked file wraps the reasoning that stays with 【】. Axis names, sign
conventions and other definition-identification questions sit outside those
marks and are left out. Each subprocess becomes one paragraph. Evaluation
scores only the blank ids that remain, which are the overlap with the source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from domain_layout import SETS

ROOT = SETS["cross.wind_flight"]["review"]
SOURCE = ROOT / "sample_02.md"
OUT_DIR = SETS["cross.wind_flight"]["root"] / "ablation"

SLOT_RE = re.compile(
    r"\{ID:(\d+), 选项:(.*?), 输出格式:([^,]*), 标签:([^,]*), (?:规则:[^,]+, )?答案:(.*?)\}",
    re.S,
)
# A kept fragment starts at one of these blanks and needs a short bridge,
# because the definition question that used to introduce it was removed.
BRIDGE = {
    "111": "按定义声明中力与力矩同坐标轴的对应，",
    "158": "定义声明已经给出迎角、侧滑角和滚转角的正方向，",
    "189": "原点上的力会在力矩参考点形成额外力矩。",
}


def _processes(text: str) -> list[tuple[str, str]]:
    body = text.split("## 完整过程", 1)[1]
    found = []
    for part in re.split(r"\n### ", body)[1:]:
        name, _, rest = part.partition("\n")
        found.append((name.strip(), rest.strip()))
    return found


def _passage(rest: str) -> str:
    spans = [re.sub(r"^\d+\.", "", span.strip()) for span in re.findall(r"【(.*?)】", rest, re.S)]
    pieces: list[str] = []
    for span in spans:
        first = SLOT_RE.search(span)
        if first and first.group(1) in BRIDGE:
            pieces.append(BRIDGE[first.group(1)])
        pieces.append(span)
    return "".join(pieces)


def _options(raw: str) -> list[str]:
    if raw.strip() == "无":
        return []
    options = []
    for piece in raw.split("；"):
        piece = piece.strip()
        if not piece:
            continue
        letter, sep, text = piece.partition("：")
        options.append(text if sep else piece)
    return options


def _questions(process: str, passage: str) -> list[dict]:
    questions = []
    for match in SLOT_RE.finditer(passage):
        blank_id, options, format_code, label, answer = match.groups()
        questions.append({
            "process": process,
            "id": blank_id,
            "local_id": blank_id,
            "options": _options(options),
            "format_code": format_code,
            "label": label,
            "answer": answer,
            "prompt": f"{process} 消融空 {blank_id}",
        })
    return questions


def _template(passage: str) -> str:
    return SLOT_RE.sub(lambda match: f"@@{match.group(1)}@@", passage)


def _declaration(text: str) -> str:
    chunk = text.split("## 定义声明", 1)[1].split("## 物理常识", 1)[0]
    return chunk.strip()


def build() -> dict:
    text = SOURCE.read_text(encoding="utf-8")
    source_rows = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
    source = next(row for row in source_rows if str(row["id"]) == "02")
    processes = {}
    questions = []
    kept: list[str] = []
    dropped: list[str] = []
    for name, rest in _processes(text):
        all_ids = [match.group(1) for match in SLOT_RE.finditer(rest)]
        passage = _passage(rest)
        present = [match.group(1) for match in SLOT_RE.finditer(passage)]
        if present != [blank_id for blank_id in all_ids if blank_id in set(present)]:
            raise RuntimeError(f"{name} 消融段落打乱了空的顺序。")
        missing = [blank_id for blank_id in all_ids if blank_id not in present]
        extra = [blank_id for blank_id in present if blank_id not in all_ids]
        if extra or len(present) != len(set(present)):
            raise RuntimeError(f"{name} 消融段落重复或带入了段外的空：{extra}")
        processes[name] = passage
        questions.extend(_questions(name, passage))
        kept.extend(present)
        dropped.extend(missing)
    if len(kept) + len(dropped) != 200:
        raise RuntimeError(f"空的总数是 {len(kept) + len(dropped)}，应为 200。")
    return {
        "id": "02-ablation",
        "source_id": "02",
        "title": "反装·去掉定义识别提问",
        "change": "只保留审阅稿【】中的推理过程。定义识别类提问不进入本题，评测时也不计入正确率。",
        "spec": source["spec"],
        "declaration": _declaration(text),
        "tier": source.get("tier", 0),
        "group": "ablation",
        "process_text": {name: _template(passage) for name, passage in processes.items()},
        "processes": processes,
        "questions": questions,
        "overlap_ids": kept,
        "excluded_ids": dropped,
    }


def _review(record: dict) -> str:
    lines = [
        "# 数据 02 反装·消融",
        "",
        "这份数据从审阅稿 sample_02.md 的【】抽出。定义声明和物理常识仍完整给出。",
        "每个子过程收成一段，只问后续推理；坐标轴识别、正方向复述这类定义识别题已去掉。",
        "评测时只比较本文件里出现的空。这些空与原始数据编号相同。原始数据里多出来的空不统计。",
        "",
        f"保留 {len(record['overlap_ids'])} 个空，排除 {len(record['excluded_ids'])} 个空。",
        "",
        "## 定义声明",
        "",
        record["declaration"],
        "",
        "## 物理常识",
        "",
        "轴向正方向、法向正方向、侧向正方向分别是前方、上方、右侧。模型轴向的正方向、机体轴向的正方向、天平轴向正方向、风洞轴向的正方向都是这个绝对前方，不是坐标轴的正方向。单独变换迎角时，旋转轴是绝对侧向轴。单独变换侧滑角时，旋转轴是绝对法向轴。单独变换滚转角时，旋转轴是绝对轴向轴。各角的正方向以定义声明为准。坐标系转换采用被动变换。姿态角采用动轴内旋。",
        "",
        "## 推理过程",
        "",
        "下面每个槽都带标签和标准答案，便于审阅。模型看到的待填槽不含这两项。",
        "",
    ]
    for name, passage in record["processes"].items():
        lines.extend([f"### {name}", "", passage, ""])
    return "\n".join(lines)


def build_full() -> dict:
    """The same hand-edited manuscript, with definition questions kept."""
    text = SOURCE.read_text(encoding="utf-8")
    source_rows = json.loads((ROOT / "samples.json").read_text(encoding="utf-8"))
    source = next(row for row in source_rows if str(row["id"]) == "02")
    processes = {}
    questions = []
    for name, rest in _processes(text):
        passage = rest.replace("【", "").replace("】", "")
        if "【" in passage or "】" in passage:
            raise RuntimeError(f"{name} 的标注括号没有去干净。")
        present = [match.group(1) for match in SLOT_RE.finditer(passage)]
        if len(present) != len(set(present)):
            raise RuntimeError(f"{name} 有重复的空。")
        processes[name] = passage
        questions.extend(_questions(name, passage))
    if len(questions) != 200:
        raise RuntimeError(f"原始稿有 {len(questions)} 个空，应为 200。")
    return {
        "id": "02",
        "source_id": "02",
        "title": "反装",
        "change": "安装方式改为反装，其余保持默认。提问保留审阅稿全文，含定义识别题。",
        "spec": source["spec"],
        "declaration": _declaration(text),
        "tier": source.get("tier", 0),
        "group": "review",
        "process_text": {name: _template(passage) for name, passage in processes.items()},
        "questions": questions,
    }


def main() -> None:
    record = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sample_02.md").write_text(_review(record), encoding="utf-8")
    stored = {key: value for key, value in record.items() if key != "processes"}
    (OUT_DIR / "samples.jsonl").write_text(
        json.dumps(stored, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "overlap_ids.json").write_text(
        json.dumps({
            "source_id": "02",
            "ablation_id": "02-ablation",
            "overlap_ids": record["overlap_ids"],
            "excluded_ids": record["excluded_ids"],
            "rule": "只统计 overlap_ids。excluded_ids 是定义识别题，不进入消融正确率。",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"kept {len(record['overlap_ids'])} excluded {len(record['excluded_ids'])} -> {OUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
