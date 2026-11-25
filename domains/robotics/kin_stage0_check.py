# -*- coding: utf-8 -*-
"""Stage 0: theory review manuscript covers F1–F13 and I1–I10."""
from __future__ import annotations

from domains.robotics.kin_manuscript import gold_scored, manuscript_path, questions
from domains.robotics.kin_pipeline_scope import NODE_IDS


def check_stage0() -> None:
    rows = questions()
    scored = gold_scored()
    if set(rows) != set(NODE_IDS):
        raise RuntimeError("节点集合与范围不一致")
    for node in NODE_IDS:
        if not scored[node]:
            raise RuntimeError(f"{node} 金标为空")
        if rows[node]["options"] and scored[node] not in {
            chr(ord("A") + i) for i in range(len(rows[node]["options"]))
        }:
            raise RuntimeError(f"{node} 选择题金标不是字母：{scored[node]!r}")
    if not manuscript_path().is_file():
        raise RuntimeError("审阅稿不存在")
    print(f"阶段0 稿面：机器人学运动学金标齐全，共 {len(NODE_IDS)} 空（F140+I16）。", flush=True)


if __name__ == "__main__":
    check_stage0()
