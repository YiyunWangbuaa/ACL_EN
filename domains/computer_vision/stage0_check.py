# -*- coding: utf-8 -*-
"""Stage 0: basic review manuscript covers 1–66; optional generator diff."""
from __future__ import annotations

from domains.computer_vision.cv_basic_generate import STANDARD_SPEC, build
from domains.computer_vision.manuscript import gold_texts, manuscript_path, questions
from domains.computer_vision.pipeline_scope import NODE_IDS
from cloze_score import answer_letter


def check_stage0() -> None:
    rows = questions()
    if set(rows) != set(NODE_IDS):
        raise RuntimeError("node set does not match the range")
    if not manuscript_path().is_file():
        raise RuntimeError("review manuscript is missing")
    built = build(STANDARD_SPEC)
    generated = {}
    for item in built["questions"]:
        node = str(item["id"])
        if node not in NODE_IDS:
            continue
        options = list(item.get("options") or [])
        text = str(item["answer"])
        if options:
            letter = answer_letter(text, options)
            generated[node] = options[ord(letter) - ord("A")] if letter else text
        else:
            generated[node] = text
    gold = gold_texts()
    diff = {
        node: (gold[node], generated.get(node))
        for node in NODE_IDS
        if generated.get(node) != gold[node]
    }
    if diff:
        # Allow sympy-equal open answers; fail only hard mismatches.
        from domains.computer_vision.answer_form import answers_equal

        hard = {}
        for node, pair in diff.items():
            fmt = rows[node]["format"] if not rows[node]["options"] else "expression"
            if pair[1] is None or not answers_equal(pair[0], pair[1], fmt):
                hard[node] = pair
        if hard:
            sample = dict(list(hard.items())[:3])
            raise RuntimeError(f"generator and review manuscript differ: {sample}")
    print(f"stage 0 manuscript: vision basics 1-66 gold diff is empty, {len(NODE_IDS)} blanks.", flush=True)


if __name__ == "__main__":
    check_stage0()
