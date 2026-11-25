# -*- coding: utf-8 -*-
"""Answer parsing and equivalence for robotics theory."""
from __future__ import annotations

from cloze_score import normalize, sympy_equal
from domains.robotics.manuscript import questions
from domains.robotics.pipeline_scope import NODE_IDS
from domains.robotics.signed_lists import EQUIVALENT_PAIRS, INEQUIVALENT_PAIRS


def parse_answer(text: str, fmt: str = "无") -> str:
    raw = str(text).strip()
    if not raw:
        raise ValueError("空答案")
    if fmt == "无" and len(raw) == 1 and raw.isalpha():
        return raw.upper()
    return normalize(raw)


def answers_equal(left: str, right: str, fmt: str = "无") -> bool:
    a = str(left).strip()
    b = str(right).strip()
    if a == b:
        return True
    if fmt == "无" and len(a) == 1 and len(b) == 1:
        return a.upper() == b.upper()
    if normalize(a) == normalize(b):
        return True
    if fmt in {"向量", "矩阵", "表达式", "方向角", "关节角", "公式"} or any(ch in a + b for ch in "[]*"):
        return bool(sympy_equal(a, b))
    return False


def check_answer_form() -> None:
    rows = questions()
    for node in NODE_IDS:
        row = rows[node]
        parse_answer(row["answer"], row["format"] if not row["options"] else "无")
        parse_answer(row["answer_text"], row["format"] if not row["options"] else "表达式")
    for left, right in EQUIVALENT_PAIRS:
        if not answers_equal(left, right, "表达式"):
            raise RuntimeError(f"应等价：{left!r} vs {right!r}")
    for left, right in INEQUIVALENT_PAIRS:
        if answers_equal(left, right, "表达式"):
            raise RuntimeError(f"不应等价：{left!r} vs {right!r}")
    print(
        f"阶段2 答案形式：金标解析 {len(NODE_IDS)}/{len(NODE_IDS)}，"
        f"等价对 {len(EQUIVALENT_PAIRS)}，不等价对 {len(INEQUIVALENT_PAIRS)}。",
        flush=True,
    )


if __name__ == "__main__":
    check_answer_form()
