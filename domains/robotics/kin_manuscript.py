# -*- coding: utf-8 -*-
"""Load gold from robotics theory review markdown (F*/I* namespaces)."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from domain_layout import EVAL
from domains.mbd_kernel import question_clauses
from domains.robotics.kin_pipeline_scope import FORWARD_IDS, INVERSE_IDS, NODE_IDS, SAMPLE_ID

REVIEW = EVAL["robotics"]["data"]
_SLOT = re.compile(
    r"\{ID:(\d+), options:(.*?), format:(.*?), label:(.*?), answer:(.*?)\}"
)
_FORWARD_MARK = "# Forward kinematics"
_INVERSE_MARK = "# Inverse kinematics"


def _parse_options(raw: str) -> list[str]:
    text = raw.strip()
    if text == "none":
        return []
    found = []
    for part in re.split(r"[;；]", text):
        part = part.strip()
        if "：" in part:
            found.append(part.split("：", 1)[1].strip())
        elif ":" in part:
            found.append(part.split(":", 1)[1].strip())
    return found


def _section_slots(body: str, prefix: str, expected: tuple[str, ...]) -> dict[str, dict]:
    clauses = question_clauses(body)
    found = {}
    for match in _SLOT.finditer(body):
        public = match.group(1)
        node = f"{prefix}{public}"
        if node not in expected:
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
                    raise RuntimeError(f"{node} option letter out of range: {raw_answer}")
                answer_text = options[index]
            else:
                for index, option in enumerate(options):
                    if option == raw_answer:
                        letter = chr(ord("A") + index)
                        answer_text = option
                        break
                if not letter:
                    raise RuntimeError(f"{node} cannot parse the choice answer: {raw_answer!r}")
        found[node] = {
            "id": node,
            "public_id": public,
            "label": match.group(4).strip(),
            "stem": clauses.get(public, ""),
            "format": match.group(3).strip(),
            "options": options,
            "answer_text": answer_text,
            "answer_letter": letter,
            "answer": letter if options else raw_answer,
        }
    missing = [node for node in expected if node not in found]
    if missing:
        raise RuntimeError(f"{prefix} missing blanks: {missing}")
    return found


@lru_cache(maxsize=1)
def questions() -> dict[str, dict]:
    path = REVIEW / f"sample_{SAMPLE_ID}.md"
    text = path.read_text(encoding="utf-8")
    if _FORWARD_MARK not in text or _INVERSE_MARK not in text:
        raise RuntimeError("The review manuscript is missing the Forward kinematics or Inverse kinematics section")
    forward_body = text.split(_FORWARD_MARK, 1)[1].split(_INVERSE_MARK, 1)[0]
    inverse_body = text.split(_INVERSE_MARK, 1)[1].split("# Cross-layer correspondence", 1)[0]
    found = {}
    found.update(_section_slots(forward_body, "F", FORWARD_IDS))
    found.update(_section_slots(inverse_body, "I", INVERSE_IDS))
    if set(found) != set(NODE_IDS):
        raise RuntimeError(f"node set is incomplete: missing {set(NODE_IDS) - set(found)}")
    return found


@lru_cache(maxsize=1)
def gold_texts() -> dict[str, str]:
    return {node: row["answer_text"] for node, row in questions().items()}


@lru_cache(maxsize=1)
def gold_scored() -> dict[str, str]:
    return {node: row["answer"] for node, row in questions().items()}


def manuscript_path() -> Path:
    return REVIEW / f"sample_{SAMPLE_ID}.md"
