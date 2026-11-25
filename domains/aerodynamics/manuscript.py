# -*- coding: utf-8 -*-
"""Load gold answers for the full flight manuscript (1–131)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from cloze_score import answer_letter
from domain_layout import DOMAINS
from domains.aerodynamics.flow_generate import STANDARD_SPEC, build, samples
from domains.aerodynamics.pipeline_scope import LAST_ID, NODE_IDS, SAMPLE_ID

REVIEW = DOMAINS["aerodynamics"]["flow"]["review"]
_SLOT = re.compile(
    r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}"
)


@lru_cache(maxsize=1)
def record() -> dict:
    for sample in samples():
        if sample["id"] == SAMPLE_ID:
            return build(sample["spec"])
    raise RuntimeError(f"找不到样本 {SAMPLE_ID}")


@lru_cache(maxsize=1)
def questions() -> dict[str, dict]:
    found = {}
    for item in record()["questions"]:
        node = str(item["id"])
        if node not in NODE_IDS:
            continue
        options = list(item["options"] or [])
        text = str(item["answer"])
        letter = answer_letter(text, options) if options else ""
        found[node] = {
            "id": node,
            "label": item["label"],
            "stem": item["stem"],
            "format": item["format"],
            "options": options,
            "answer_text": text,
            "answer_letter": letter,
            "answer": letter if options else text,
        }
    if set(found) != set(NODE_IDS):
        raise RuntimeError("生成器没有给出完整的 1–131。")
    return found


@lru_cache(maxsize=1)
def gold_texts() -> dict[str, str]:
    return {node: row["answer_text"] for node, row in questions().items()}


@lru_cache(maxsize=1)
def gold_scored() -> dict[str, str]:
    return {node: row["answer"] for node, row in questions().items()}


def manuscript_answers(path: Path | None = None) -> dict[str, str]:
    """Letter or open text as written in the review markdown."""
    target = path or (REVIEW / f"sample_{SAMPLE_ID}.md")
    found = {}
    for match in _SLOT.finditer(target.read_text(encoding="utf-8")):
        node = match.group(1)
        if node in NODE_IDS:
            found[node] = match.group(5).strip()
    return found


def samples_json_answers() -> dict[str, str]:
    rows = json.loads((REVIEW / "samples.json").read_text(encoding="utf-8"))
    sample = next(item for item in rows if item["id"] == SAMPLE_ID)
    return {
        str(item["id"]): str(item["answer"])
        for item in sample["questions"]
        if str(item["id"]) in NODE_IDS
    }
