# -*- coding: utf-8 -*-
"""Answer parsing and equivalence for the first flight block."""
from __future__ import annotations

from cloze_score import normalize, sympy_equal
from domains.aerodynamics.manuscript import questions
from domains.aerodynamics.pipeline_scope import NODE_IDS
from domains.aerodynamics.signed_lists import EQUIVALENT_PAIRS, INEQUIVALENT_PAIRS


def parse_answer(text: str, fmt: str = "choice") -> str:
    raw = str(text).strip()
    if not raw:
        raise ValueError("empty answer")
    if fmt == "choice" and len(raw) == 1 and raw.isalpha():
        return raw.upper()
    return normalize(raw)


def answers_equal(left: str, right: str, fmt: str = "choice") -> bool:
    a = str(left).strip()
    b = str(right).strip()
    if a == b:
        return True
    if fmt == "choice" and len(a) == 1 and len(b) == 1:
        return a.upper() == b.upper()
    if normalize(a) == normalize(b):
        return True
    if fmt in {"vector", "matrix", "expression"} or any(ch in a + b for ch in "[]*"):
        return bool(sympy_equal(a, b))
    return False


def check_answer_form() -> None:
    rows = questions()
    for node in NODE_IDS:
        row = rows[node]
        parse_answer(row["answer"], row["format"])
        parse_answer(row["answer_text"], row["format"] if not row["options"] else "expression")
    for left, right in EQUIVALENT_PAIRS:
        if not answers_equal(left, right, "expression"):
            raise RuntimeError(f"Should be equivalent: {left!r} vs {right!r}")
    for left, right in INEQUIVALENT_PAIRS:
        if answers_equal(left, right, "expression"):
            raise RuntimeError(f"Should not be equivalent: {left!r} vs {right!r}")
    print(
        f"Stage 2 answer form: gold parsed {len(NODE_IDS)}/{len(NODE_IDS)}, "
        f"equivalent pairs {len(EQUIVALENT_PAIRS)}, inequivalent pairs {len(INEQUIVALENT_PAIRS)}.",
        flush=True,
    )


if __name__ == "__main__":
    check_answer_form()
