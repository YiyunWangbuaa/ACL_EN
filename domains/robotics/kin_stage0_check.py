# -*- coding: utf-8 -*-
"""Stage 0: theory review manuscript covers F1–F13 and I1–I10."""
from __future__ import annotations

from domains.robotics.kin_manuscript import gold_scored, manuscript_path, questions
from domains.robotics.kin_pipeline_scope import NODE_IDS


def check_stage0() -> None:
    rows = questions()
    scored = gold_scored()
    if set(rows) != set(NODE_IDS):
        raise RuntimeError("node set does not match the scope")
    for node in NODE_IDS:
        if not scored[node]:
            raise RuntimeError(f"{node} gold is empty")
        if rows[node]["options"] and scored[node] not in {
            chr(ord("A") + i) for i in range(len(rows[node]["options"]))
        }:
            raise RuntimeError(f"{node} choice gold is not a letter：{scored[node]!r}")
    if not manuscript_path().is_file():
        raise RuntimeError("review manuscript does not exist")
    print(f"stage 0 manuscript: robotics kinematics gold is complete, {len(NODE_IDS)} blanks (F140+I16).", flush=True)


if __name__ == "__main__":
    check_stage0()
