# -*- coding: utf-8 -*-
"""Load gold from the theory section of the scene manuscript (1–66)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from domain_layout import DOMAINS
from domains.computer_vision.pipeline_scope import NODE_IDS, SAMPLE_ID
from domains.mbd_kernel import question_clauses

REVIEW = Path(DOMAINS["computer_vision"]["scene"]["review"])
_THEORY = "# 前置环节：基础理论"
_SCENE = "# 物体与双目"
_SLOT = re.compile(
    r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}"
)


def _parse_options(raw: str) -> list[str]:
    text = raw.strip()
    if text == "无":
        return []
    found = []
    for part in text.split("；"):
        part = part.strip()
        if "：" in part:
            found.append(part.split("：", 1)[1].strip())
        elif ":" in part:
            found.append(part.split(":", 1)[1].strip())
    return found


def _theory_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if _THEORY not in text or _SCENE not in text:
        raise RuntimeError(f"{path.name} 缺少基础理论前置环节或物体与双目分节")
    return text.split(_THEORY, 1)[1].split(_SCENE, 1)[0]


@lru_cache(maxsize=1)
def questions() -> dict[str, dict]:
    path = REVIEW / f"sample_{SAMPLE_ID}.md"
    body = _theory_text(path)
    clauses = question_clauses(body)
    found = {}
    for match in _SLOT.finditer(body):
        node = match.group(1)
        if node not in NODE_IDS:
            continue
        options = _parse_options(match.group(2))
        raw_answer = match.group(5).strip()
        letter = ""
        answer_text = raw_answer
        if options:
            if len(raw_answer) == 1 and raw_answer.isalpha():
                letter = raw_answer.upper()
                index = ord(letter) - ord("A")
                if index < 0 or index >= len(options):
                    raise RuntimeError(f"ID{node} 选项字母越界")
                answer_text = options[index]
            else:
                for index, option in enumerate(options):
                    if option == raw_answer:
                        letter = chr(ord("A") + index)
                        answer_text = option
                        break
                if not letter:
                    raise RuntimeError(f"ID{node} 无法解析选择题答案：{raw_answer!r}")
        found[node] = {
            "id": node,
            "label": match.group(4).strip(),
            "stem": clauses.get(node, ""),
            "format": match.group(3).strip(),
            "options": options,
            "answer_text": answer_text,
            "answer_letter": letter,
            "answer": letter if options else raw_answer,
        }
    if set(found) != set(NODE_IDS):
        raise RuntimeError(f"缺空：{sorted(set(NODE_IDS) - set(found), key=int)[:8]}")
    return found


@lru_cache(maxsize=1)
def gold_texts() -> dict[str, str]:
    return {node: row["answer_text"] for node, row in questions().items()}


@lru_cache(maxsize=1)
def gold_scored() -> dict[str, str]:
    return {node: row["answer"] for node, row in questions().items()}


def manuscript_path() -> Path:
    return REVIEW / f"sample_{SAMPLE_ID}.md"
