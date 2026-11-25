# -*- coding: utf-8 -*-
"""Stage 0: scene review covers 1–165 and is free of aero cross terms."""
from __future__ import annotations

from domains.computer_vision.scene_manuscript import manuscript_path, questions
from domains.computer_vision.scene_pipeline_scope import NODE_IDS

_AERO = ("迎角", "侧滑角", "飞行力学", "正装", "反装", "倒装")


def check_stage0() -> None:
    rows = questions()
    if set(rows) != set(NODE_IDS):
        raise RuntimeError("节点集合与范围不一致")
    path = manuscript_path()
    if not path.is_file():
        raise RuntimeError("scene 审阅稿不存在")
    text = path.read_text(encoding="utf-8")
    leftover = [term for term in _AERO if term in text]
    if leftover:
        raise RuntimeError(f"scene 稿仍含气动交叉用语：{leftover}")
    for node in NODE_IDS:
        if not str(rows[node]["answer_text"]).strip():
            raise RuntimeError(f"ID{node} 金标为空")
        if rows[node]["options"] and rows[node]["answer_letter"] not in {
            chr(ord("A") + index) for index in range(len(rows[node]["options"]))
        }:
            raise RuntimeError(f"ID{node} 选择题金标不是字母：{rows[node]['answer_letter']!r}")
    print(
        f"阶段0 稿面：scene 1–165 金标齐全且无气动用语，共 {len(NODE_IDS)} 空。",
        flush=True,
    )


if __name__ == "__main__":
    check_stage0()
