# -*- coding: utf-8 -*-
"""Stage 0: scene review covers 1–165 and is free of aero cross terms."""
from __future__ import annotations

from domains.computer_vision.scene_manuscript import manuscript_path, questions
from domains.computer_vision.scene_pipeline_scope import NODE_IDS

_AERO = ("schlieren", "wind tunnel", "airspeed", "shock", "fuselage")


def check_stage0() -> None:
    rows = questions()
    if set(rows) != set(NODE_IDS):
        raise RuntimeError("node set does not match the range")
    path = manuscript_path()
    if not path.is_file():
        raise RuntimeError("scene review manuscript is missing")
    text = path.read_text(encoding="utf-8")
    leftover = [term for term in _AERO if term in text]
    if leftover:
        raise RuntimeError(f"scene manuscript still has wind-tunnel cross terms: {leftover}")
    for node in NODE_IDS:
        if not str(rows[node]["answer_text"]).strip():
            raise RuntimeError(f"ID{node} gold is empty")
        if rows[node]["options"] and rows[node]["answer_letter"] not in {
            chr(ord("A") + index) for index in range(len(rows[node]["options"]))
        }:
            raise RuntimeError(f"ID{node} choice gold is not a letter: {rows[node]['answer_letter']!r}")
    print(
        f"stage 0 manuscript: scene 1-165 gold is complete and has no wind-tunnel terms, {len(NODE_IDS)} blanks.",
        flush=True,
    )


if __name__ == "__main__":
    check_stage0()
